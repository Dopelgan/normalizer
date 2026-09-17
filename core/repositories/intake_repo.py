"""Хранение решений приёма: карантин, отказы, отчёт по массовому приёму."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.db.models import IntakeDecisionDB


def decision_id(s3_fileid: str, stage: str) -> str:
    """Одно решение на файл и этап: повторный приём перезаписывает своё."""
    digest = hashlib.sha256(f"{stage}:{s3_fileid}".encode("utf-8")).hexdigest()
    return f"intake-{digest[:32]}"


class IntakeRepository:
    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------- запись
    def record(
        self,
        s3_fileid: str,
        stage: str,
        outcome: str,
        reason: str,
        layer: Optional[str] = None,
        category: Optional[str] = None,
        confidence: float = 1.0,
        signals: Optional[Dict[str, Any]] = None,
        source_path: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> IntakeDecisionDB:
        record_id = decision_id(s3_fileid, stage)
        existing = self.session.get(IntakeDecisionDB, record_id)

        if existing is None:
            existing = IntakeDecisionDB(id=record_id, s3_fileid=s3_fileid, stage=stage)
            self.session.add(existing)

        existing.outcome = outcome
        existing.reason = reason
        existing.layer = layer
        existing.category = category
        existing.confidence = float(confidence)
        existing.signals = dict(signals or {})
        existing.source_path = source_path
        existing.file_hash = file_hash
        existing.created_at = datetime.now(timezone.utc).replace(tzinfo=None)
        # Повторный приём — это новое решение, прежний вердикт администратора
        # к нему уже не относится.
        existing.resolved_outcome = None
        existing.resolved_by = None
        existing.resolved_at = None
        self.session.flush()
        return existing

    def resolve(
        self, record_id: str, outcome: str, resolved_by: str
    ) -> Optional[IntakeDecisionDB]:
        """Решение администратора по карантинной записи."""
        record = self.session.get(IntakeDecisionDB, record_id)
        if record is None:
            return None
        record.resolved_outcome = outcome
        record.resolved_by = resolved_by
        record.resolved_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.session.flush()
        return record

    # ------------------------------------------------------------- чтение
    def get(self, record_id: str) -> Optional[IntakeDecisionDB]:
        return self.session.get(IntakeDecisionDB, record_id)

    def quarantine_queue(self, limit: int = 100) -> List[IntakeDecisionDB]:
        """Очередь администратору: только нерассмотренный карантин."""
        return list(
            self.session.execute(
                select(IntakeDecisionDB)
                .where(
                    IntakeDecisionDB.outcome == "quarantine",
                    IntakeDecisionDB.resolved_outcome.is_(None),
                )
                .order_by(IntakeDecisionDB.created_at)
                .limit(limit)
            ).scalars()
        )

    def fingerprints(self, limit: int = 500) -> List[tuple]:
        """(s3_fileid, отпечаток) принятых файлов — для поиска почти-дублей."""
        rows = self.session.execute(
            select(IntakeDecisionDB)
            .where(IntakeDecisionDB.outcome == "accept")
            .order_by(IntakeDecisionDB.created_at.desc())
            .limit(limit)
        ).scalars()
        pairs = []
        for row in rows:
            value = (row.signals or {}).get("fingerprint")
            if value is not None:
                pairs.append((row.s3_fileid, int(value)))
        return pairs

    # ------------------------------------------------------------- отчёты
    def summary(self, stage: Optional[str] = None) -> Dict[str, Any]:
        """
        Отчёт по итогам массового приёма: сколько принято, в карантине,
        отклонено, с примерами по каждой категории.
        """
        query = select(
            IntakeDecisionDB.outcome, func.count(IntakeDecisionDB.id)
        ).group_by(IntakeDecisionDB.outcome)
        if stage:
            query = query.where(IntakeDecisionDB.stage == stage)
        counts = {outcome: int(total) for outcome, total in self.session.execute(query)}

        examples: Dict[str, List[Dict[str, Any]]] = {}
        for outcome in ("accept", "quarantine", "reject"):
            rows = self.session.execute(
                select(IntakeDecisionDB)
                .where(IntakeDecisionDB.outcome == outcome)
                .order_by(IntakeDecisionDB.created_at.desc())
                .limit(5)
            ).scalars()
            examples[outcome] = [
                {
                    "id": row.id,
                    "s3_fileid": row.s3_fileid,
                    "stage": row.stage,
                    "layer": row.layer,
                    "category": row.category,
                    "reason": row.reason,
                }
                for row in rows
            ]

        return {
            "total": sum(counts.values()),
            "accepted": counts.get("accept", 0),
            "quarantined": counts.get("quarantine", 0),
            "rejected": counts.get("reject", 0),
            "examples": examples,
        }

    def rule_suggestions(self, min_cases: int = 3) -> List[Dict[str, Any]]:
        """
        Обучение на решениях администратора: если он раз за разом принимает
        то, что классификатор отправлял в карантин, правило пора править.
        Система именно предлагает — менять правила молча она не должна.
        """
        rows = self.session.execute(
            select(IntakeDecisionDB).where(
                IntakeDecisionDB.outcome == "quarantine",
                IntakeDecisionDB.resolved_outcome.isnot(None),
            )
        ).scalars()

        tally: Dict[tuple, Dict[str, int]] = {}
        for row in rows:
            key = (row.layer or "?", row.category or "?")
            bucket = tally.setdefault(key, {"accept": 0, "reject": 0})
            if row.resolved_outcome in bucket:
                bucket[row.resolved_outcome] += 1

        suggestions: List[Dict[str, Any]] = []
        for (layer, category), bucket in tally.items():
            accepted, rejected = bucket["accept"], bucket["reject"]
            if accepted >= min_cases and accepted > rejected * 2:
                suggestions.append({
                    "layer": layer,
                    "category": category,
                    "observed_accepts": accepted,
                    "observed_rejects": rejected,
                    "suggestion": (
                        f"Администратор принял {accepted} из {accepted + rejected} "
                        f"карантинных файлов категории {category!r} на слое {layer}. "
                        f"Похоже, категорию стоит перенести в accepted_categories "
                        f"профиля приёма."
                    ),
                })
        return suggestions
