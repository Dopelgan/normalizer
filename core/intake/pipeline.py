"""
Цепочка приёма одного файла: Data Gateway -> Quality Gate -> нормализатор.

Порядок жёсткий и тот же, что был. Data Gateway отвечает на вопрос
«относится ли это к корпоративным знаниям», Quality Gate — «пригоден ли
документ технически». Смешивать нельзя: личная фотография технически
безупречна и проверку качества прошла бы, поэтому отсев обязан идти
первым.

Изменилось одно: между этапами больше нет HTTP-запроса с таймаутом.
Раньше Data Gateway ходил в Quality Gate по сети и ждал ответа по всему
пакету — на пятидесяти сканах ожидание не укладывалось в 120 секунд, и
потребитель получал «Quality Gate недоступен» вместо вердиктов. Теперь оба
этапа идут в одной задаче, по одному файлу за раз, и ждать некому.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from core import telemetry
from core.config import settings
from core.db.session import SessionLocal
from core.gateway.profile import load_profile
from core.gateway.service import ACCEPT, REJECT, DataGateway
from core.intake import state
from core.models.intake import FileVerdict, IntakeFile
from core.providers.file_locator import FileLocator, SourceFileNotFound
from core.providers.storage import StorageProviderFactory
from core.quality.service import QualityGate
from core.repositories import IntakeRepository
from core.repositories.intake_repo import decision_id

logger = logging.getLogger(__name__)

STAGE_GATEWAY = "data_gateway"
STAGE_QUALITY = "quality_gate"


class IntakePipeline:
    """Оба этапа приёма в одном процессе. Создаётся на задачу."""

    def __init__(
        self,
        storage=None,
        session_factory=None,
        profile=None,
    ):
        self.storage = storage or StorageProviderFactory.default()
        self.session_factory = session_factory or SessionLocal
        self.profile = profile or load_profile()
        self.locator = FileLocator(self.storage)

    # ------------------------------------------------------------- публичное
    def run(self, request_id: str, item: IntakeFile) -> Dict[str, Any]:
        """
        Итог по файлу: исход, вердикты этапов и время каждой операции.

        Ничего не бросает наружу: приём одного файла не должен уводить в
        повтор задачу, которая уже вынесла вердикты по остальным.
        """
        with telemetry.collect() as timings:
            with telemetry.measure(
                "intake.file", request_id=request_id, s3_fileid=item.s3_fileid
            ) as span:
                result = self._run(request_id, item)
                span["outcome"] = result["outcome"]
            result["timings_ms"] = dict(timings)
        return result

    # ------------------------------------------------------------ внутреннее
    def _run(self, request_id: str, item: IntakeFile) -> Dict[str, Any]:
        s3_fileid = item.s3_fileid
        verdicts: List[FileVerdict] = []

        try:
            located = self.locator.locate(s3_fileid)
            uri = located.uri
        except SourceFileNotFound as exc:
            verdicts.append(FileVerdict(
                s3_fileid=s3_fileid, outcome=REJECT, stage=STAGE_GATEWAY,
                layer="G-1", reason=str(exc),
            ))
            return _result(s3_fileid, REJECT, verdicts)
        except Exception as exc:  # noqa: BLE001 — хранилище не обязано отвечать
            logger.error("Файл %s не найден в хранилище: %s", s3_fileid, exc)
            verdicts.append(FileVerdict(
                s3_fileid=s3_fileid, outcome=REJECT, stage=STAGE_GATEWAY,
                layer="G-1", reason=f"Хранилище недоступно: {exc}",
            ))
            return _result(s3_fileid, REJECT, verdicts)

        size = self._size(uri)

        gateway_verdict = self._gateway(s3_fileid, uri, size)
        verdicts.append(gateway_verdict)
        if gateway_verdict.outcome != ACCEPT:
            return _result(s3_fileid, gateway_verdict.outcome, verdicts)

        quality_verdict, file_hash = self._quality(s3_fileid, uri)
        verdicts.append(quality_verdict)
        if quality_verdict.outcome != ACCEPT:
            state.release_content(file_hash or "", s3_fileid)
            return _result(s3_fileid, quality_verdict.outcome, verdicts)

        result = _result(s3_fileid, ACCEPT, verdicts)
        if settings.INTAKE_FORWARD_TO_PARSER:
            forwarded, error = self.forward(request_id, [item])
            result["forwarded"] = forwarded
            result["forward_error"] = error
        return result

    def _size(self, uri: str) -> Optional[int]:
        getter = getattr(self.storage, "size", None)
        if getter is None:
            return None
        try:
            return getter(uri)
        except Exception as exc:  # noqa: BLE001 — неизвестный размер не отказ
            logger.warning("Размер %s не определён: %s", uri, exc)
            return None

    # ------------------------------------------------------- этап G (отсев)
    def _gateway(self, s3_fileid: str, uri: str, size: Optional[int]) -> FileVerdict:
        gateway = DataGateway(profile=self.profile, storage=self.storage)
        result = gateway.evaluate(s3_fileid, uri, size)
        self._record(
            s3_fileid=s3_fileid, stage=STAGE_GATEWAY, outcome=result.outcome,
            reason=result.reason, layer=result.layer, category=result.category,
            confidence=result.confidence, signals=result.signals, source_path=uri,
        )
        return FileVerdict(
            s3_fileid=s3_fileid, outcome=result.outcome, stage=STAGE_GATEWAY,
            layer=result.layer, reason=result.reason, category=result.category,
            confidence=result.confidence,
            decision_id=decision_id(s3_fileid, STAGE_GATEWAY),
        )

    # ------------------------------------------------- этап QG (пригодность)
    def _quality(self, s3_fileid: str, uri: str):
        known_hashes, known_fingerprints = self._known()
        gate = QualityGate(
            profile=self.profile, storage=self.storage,
            known_hashes=known_hashes, known_fingerprints=known_fingerprints,
        )
        result = gate.evaluate(s3_fileid, uri)

        # Заявка на содержимое закрывает гонку внутри пакета: два
        # одинаковых файла обрабатываются параллельно и записей друг друга
        # в базе ещё не видят, поэтому по базе оба проходят как новые.
        if result.outcome == ACCEPT and result.file_hash:
            owner = state.claim_content(result.file_hash, s3_fileid)
            if owner:
                result.outcome = REJECT
                result.reason = (
                    f"Точный дубликат уже принятого файла {owner!r} "
                    f"(совпадает контрольная сумма)."
                )
                result.stage = "QG-2"

        self._record(
            s3_fileid=s3_fileid, stage=STAGE_QUALITY, outcome=result.outcome,
            reason=result.reason, layer=result.stage, confidence=result.confidence,
            signals={**result.signals, "warnings": result.warnings,
                     "routing": result.routing},
            source_path=uri, file_hash=result.file_hash,
        )
        return FileVerdict(
            s3_fileid=s3_fileid, outcome=result.outcome, stage=STAGE_QUALITY,
            layer=result.stage, reason=result.reason, confidence=result.confidence,
            warnings=result.warnings, routing=result.routing,
            decision_id=decision_id(s3_fileid, STAGE_QUALITY),
        ), result.file_hash

    def _known(self):
        """Хеши и отпечатки уже принятого — для поиска дублей."""
        from sqlalchemy import select

        from core.db.models import IntakeDecisionDB

        session = self.session_factory()
        try:
            with session.begin():
                rows = session.execute(
                    select(IntakeDecisionDB)
                    .where(IntakeDecisionDB.outcome == ACCEPT)
                    .order_by(IntakeDecisionDB.created_at.desc())
                    .limit(1000)
                ).scalars()
                hashes = {r.file_hash: r.s3_fileid for r in rows if r.file_hash}
                fingerprints = IntakeRepository(session).fingerprints()
            return hashes, fingerprints
        except Exception as exc:  # noqa: BLE001 — без базы приём всё равно идёт
            logger.error("Известные хеши не прочитаны: %s", exc)
            return {}, []
        finally:
            session.close()

    # ------------------------------------------------------------ журнал
    def _record(self, **fields: Any) -> None:
        """Решение переживает запрос — иначе ни дублей, ни отчёта приёма."""
        session = self.session_factory()
        try:
            with session.begin():
                IntakeRepository(session).record(**fields)
        except Exception as exc:  # noqa: BLE001 — журнал не важнее вердикта
            logger.error(
                "Решение по %s (%s) не записано: %s",
                fields.get("s3_fileid"), fields.get("stage"), exc,
            )
        finally:
            session.close()

    # ------------------------------------------------- передача в разбор
    @staticmethod
    def forward(request_id: str, files: List[IntakeFile]):
        """Только утверждённые файлы уходят в нормализатор."""
        url = f"{settings.PARSER_ENDPOINT.rstrip('/')}/internal/v1/parse/background"
        payload = {
            "request_id": request_id,
            "files": [item.model_dump() for item in files],
        }
        try:
            with telemetry.measure("intake.forward", request_id=request_id):
                response = requests.post(
                    url, json=payload, timeout=settings.INTAKE_TIMEOUT
                )
                response.raise_for_status()
            return True, None
        except requests.RequestException as exc:
            logger.error("Нормализатор не принял запрос %s: %s", request_id, exc)
            return False, str(exc)


def _result(
    s3_fileid: str, outcome: str, verdicts: List[FileVerdict]
) -> Dict[str, Any]:
    return {
        "s3_fileid": s3_fileid,
        "outcome": outcome,
        "verdicts": [v.model_dump(mode="json") for v in verdicts],
        "forwarded": False,
        "forward_error": None,
        "timings_ms": {},
    }


def evaluate_file(request_id: str, item: IntakeFile, **kwargs) -> Dict[str, Any]:
    """Короткий путь для задачи и для тестов."""
    return IntakePipeline(**kwargs).run(request_id, item)
