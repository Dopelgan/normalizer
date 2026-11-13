"""
Маршрутизатор лестницы стратегий.

Правило подъёма: начинаем с самой дешёвой применимой стратегии и
поднимаемся выше, пока уверенность или структурная полнота ниже порога.
Из этого правила есть исключение: результат-фоллбэк подъём не останавливает,
каким бы высоким ни был его балл. Плоское распознавание физически не может
вернуть таблицу, формулу или поле чертежа, и его высокая оценка говорит
только об объёме текста.

Правило старшинства: структурный результат старше фоллбэка **всегда**, а не
только выше порога. Раньше оговорка про порог оставляла дыру ровно там, где
она дороже всего: на листе с формулами структурный разбор набирает мало
символов и до порога не дотягивает, а сплошное распознавание той же страницы
набирает много — и выигрывало сравнение, отдавая в индекс кашу вида
`д = —fz+ay` с честной оценкой 0.87. Балл меряет объём текста и уверенность,
то есть ровно те величины, по которым каша выглядит хорошо; решать им спор
между разными способами разбора нельзя.

Оговорка одна и она про сохранность: структурный результат получает
старшинство, только если принёс сопоставимый объём текста
(`LADDER_CONTENT_KEEP_RATIO` от лучшего фоллбэка). Разбор, вернувший три
символа со страницы, не должен побеждать распознавание всей страницы только
потому, что он структурный.

У оговорки есть граница применимости (`LADDER_DRAWING_OVER_OCR`): на
листе-картинке, который растровый детектор считает чертежом, число символов
у структурного разбора о потере не говорит. MinerU возвращает такой лист
одной картинкой, а читает его модель — после лестницы. Здесь старшинство
структурного разбора сохраняется, лучший фоллбэк остаётся в запасе, а
сохранность сверяется после чтения листа (`core/ladder/drawing_route.py`).

Правило сравнения: результаты уровней не смешиваются. Каждый уровень даёт
свой результат целиком, они сравниваются оценкой, и наружу уходит лучший —
вместе со сведениями о том, какой стратегией он получен и что пробовали до
этого. Смешение уровней дало бы документ, собранный из кусков разного
качества, в котором уже не разобрать, чему верить.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.ladder import drawing_route
from core.ladder.base import Strategy, text_chars
from core.ladder.context import DocumentContext
from core.ladder.scoring import Score, score_result
from core.ladder.strategies.cad_source import CadSourceStrategy
from core.ladder.strategies.native_text import (
    DocxStrategy,
    PdfTextLayerStrategy,
    PlainTextStrategy,
)
from core.ladder.strategies.recognition import (
    LayoutStrategy,
    PlainOcrStrategy,
    RestoreThenLayoutStrategy,
    VlmEscalationStrategy,
)
from core.ladder.strategies.tabular import SpreadsheetStrategy
from core.models.parse_result import ParseResult
from core.providers.storage import StorageProvider
from core.providers.document_parser import DocumentParserProvider, ParserFailed
from core.providers.tesseract_fallback import ocr_memo

logger = logging.getLogger(__name__)


# Уровни платформы по возрастанию стоимости. Здесь только типы: создавать
# стратегию значит поднимать за собой распознавание или клиента парсера, а
# выключенный уровень платить за это не должен.
STRATEGY_TYPES = (
    CadSourceStrategy,
    PdfTextLayerStrategy,
    DocxStrategy,
    PlainTextStrategy,
    SpreadsheetStrategy,
    PlainOcrStrategy,
    LayoutStrategy,
    RestoreThenLayoutStrategy,
    VlmEscalationStrategy,
)

# Уровни, которые сами ходят в хранилище: им нужен тот же провайдер, что и
# всей задаче, иначе каждый заводит свой клиент S3 со своим пулом.
_TAKES_STORAGE = (PlainOcrStrategy, LayoutStrategy, RestoreThenLayoutStrategy)


def default_strategies(
    storage: Optional[StorageProvider] = None,
    levels: Optional[Sequence[int]] = None,
) -> List[Strategy]:
    """
    Стратегии платформы, по возрастанию стоимости.

    `levels` ограничивает набор до создания: отключённый уровень не должен
    создаваться вовсе. `storage` уезжает внутрь уровней, которые читают и
    пишут файлы, — общий на документ, а не свой у каждого.
    """
    allowed = None if levels is None else set(levels)
    selected = [t for t in STRATEGY_TYPES if allowed is None or t.level in allowed]
    return sorted(
        (
            strategy_type(storage=storage)
            if issubclass(strategy_type, _TAKES_STORAGE)
            else strategy_type()
            for strategy_type in selected
        ),
        key=lambda strategy: strategy.level,
    )


@dataclass
class Attempt:
    """Попытка одного уровня — для отчёта о том, как выбирали."""

    level: int
    strategy: str
    score: Optional[Score] = None
    error: Optional[str] = None
    is_fallback: bool = False
    chars: int = 0

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"level": self.level, "strategy": self.strategy}
        if self.score is not None:
            payload.update(self.score.as_dict())
            payload["is_fallback"] = self.is_fallback
            payload["chars"] = self.chars
        if self.error:
            payload["error"] = self.error
        return payload


class LadderRouter:
    """Выбор стратегии разбора и подъём по лестнице."""

    def __init__(
        self,
        strategies: Optional[Sequence[Strategy]] = None,
        threshold: Optional[float] = None,
        enabled_levels: Optional[Sequence[int]] = None,
        storage: Optional[StorageProvider] = None,
    ):
        allowed = set(
            enabled_levels if enabled_levels is not None else settings.LADDER_ENABLED_LEVELS
        )
        # Именно `is None`: пустой список — это осознанное «стратегий нет»,
        # а не повод молча подставить набор по умолчанию. Набор по умолчанию
        # создаётся сразу отфильтрованным: конструктор уровня не бесплатен.
        if strategies is None:
            self.strategies = default_strategies(storage=storage, levels=allowed)
        else:
            self.strategies = sorted(
                (s for s in strategies if s.level in allowed),
                key=lambda s: s.level,
            )
        self.threshold = (
            settings.LADDER_SCORE_THRESHOLD if threshold is None else float(threshold)
        )

    # ------------------------------------------------------------- публичное
    def parse(self, context: DocumentContext) -> ParseResult:
        # Уровни 4 и 5 читают одни и те же страницы скана: в пределах
        # документа каждая распознаётся один раз (результат тот же).
        with ocr_memo():
            return self._parse(context)

    def _parse(self, context: DocumentContext) -> ParseResult:
        applicable = self._applicable(context)
        if not applicable:
            raise ParserFailed(
                f"Ни одна стратегия не применима к {context.uri} "
                f"(тип {context.file_type!r}, род {context.kind!r})"
            )

        classification = context.classification
        logger.info(
            "Лестница для %s: источник=%s жанр=%s качество=%s, уровни %s",
            context.uri, classification.source, classification.genre,
            classification.raster_quality,
            [s.level for s in applicable],
        )

        attempts: List[Attempt] = []
        best: Optional[ParseResult] = None
        best_score: Optional[Score] = None
        best_strategy: Optional[Strategy] = None
        candidates: List[tuple[ParseResult, Score, Strategy, float]] = []

        # Объём лучшего фоллбэка — мерка сохранности для структурных
        # уровней: фоллбэк читает страницу целиком и годится как нижняя
        # граница того, сколько текста на ней вообще есть.
        fallback_chars = 0
        structural_seen = False
        # Лист прочитает чертёжная ветка: счёт символов у структурного
        # разбора сравнивал бы непрочитанный лист с распознаванием.
        sheet_read_later = drawing_route.deferred(context)

        def candidate_rank(candidate):
            parsed, evaluated, _, unit_coverage = candidate
            structured = not parsed.is_fallback
            retains = (
                self._keeps_content(text_chars(parsed), fallback_chars)
                or (structured and sheet_read_later)
            )
            # Дополнительная страница не оправдывает доказанную потерю уже
            # извлечённого текста. После нового fallback переоцениваем и
            # прежних кандидатов относительно той же мерки сохранности.
            return (int(retains), unit_coverage, int(structured and retains), evaluated.value)

        for strategy in applicable:
            left = deadline.remaining()
            if left is not None and left < settings.LADDER_MIN_REMAINING:
                logger.warning("Уровень %d пропущен: до срока задачи %.0f с", strategy.level, left)
                attempts.append(Attempt(strategy.level, strategy.name,
                                        error="нет бюджета времени задачи"))
                continue
            try:
                result = strategy.run(context)
            except SoftTimeLimitExceeded:
                # Лимит задачи — не «неудача уровня». Если кандидат уже есть,
                # он сохраняется с пометкой; иначе лимит обрабатывает задача.
                if best is None:
                    raise
                note = f"срок задачи истёк на уровне {strategy.level} ({strategy.name})"
                best.degraded.append(note)
                attempts.append(Attempt(strategy.level, strategy.name, error=note))
                logger.warning("Разбор %s: %s — сохраняем лучший результат", context.uri, note)
                break
            except Exception as exc:  # noqa: BLE001 — неудача уровня не фатальна
                logger.warning(
                    "Уровень %d (%s) не справился: %s", strategy.level, strategy.name, exc
                )
                attempts.append(Attempt(strategy.level, strategy.name, error=str(exc)))
                continue

            chars = text_chars(result)
            source_count = context.source_page_count or result.source_page_count
            if source_count is not None:
                result.source_page_count = source_count
                result.page_count = source_count
                covered = (set(result.pages()) | set(result.confirmed_blank_pages)) - set(result.failed_pages)
                coverage = len({page for page in covered if 1 <= page <= source_count}) / source_count
            else:
                # Неизвестная граница будет partial в отчёте; порядок
                # существующих стратегий не заменяется догадкой о числе.
                coverage = 1.0
            references = set(result.pages()) | set(result.confirmed_blank_pages) | set(result.failed_pages)
            units_complete = (source_count is not None and coverage == 1.0 and not result.failed_pages
                              and all(1 <= page <= source_count for page in references))
            structural = not result.is_fallback
            score = score_result(
                result, context, exhaustive=strategy.exhaustive,
                expect_structure=structural_seen and not structural,
            )
            attempts.append(Attempt(
                strategy.level, strategy.name, score=score,
                is_fallback=result.is_fallback, chars=chars,
            ))
            logger.info(
                "Уровень %d (%s): оценка %.3f, символов %d%s",
                strategy.level, strategy.name, score.value, chars,
                f" — {score.reason}" if score.reason else "",
            )

            if structural:
                structural_seen = True
                # Место этих строк важно, и важность его не видна: признак
                # должен быть выставлен до `candidates.append` ниже, потому
                # что `candidate_rank` читает `sheet_read_later` из
                # объемлющей области на каждом вызове `max`. Переставишь —
                # кандидат попадёт в сравнение с прежним значением, и
                # лестница молча вернётся к счёту символов.
                if (not sheet_read_later
                        and not self._keeps_content(chars, fallback_chars)
                        and drawing_route.unread_frame_sheet(context, result)):
                    # Лист с рамкой формата вернулся картинкой без текста:
                    # читать его будет чертёжная ветка, и до её чтения счёт
                    # символов сравнивает непрочитанный лист с
                    # распознаванием. Старшинство такому разбору даётся
                    # авансом — ровно как листу жанра `drawing`.
                    sheet_read_later = True
                    logger.info(
                        "Лист %s: рамка формата есть, уровень %d вернул лист "
                        "картинкой без текста — лист прочитает чертёжная ветка",
                        context.uri, strategy.level,
                    )
            else:
                fallback_chars = max(fallback_chars, chars)

            # Старшинство структурного разбора. Балл здесь не при чём: он
            # меряет объём текста и уверенность, а фоллбэк на странице с
            # формулами набирает и то и другое, возвращая кашу. Единственная
            # проверка — сохранность: разбор, потерявший больше половины
            # текста относительно фоллбэка, старшинства не получает и
            # соревнуется по баллу наравне.
            keeps_text = structural and self._keeps_content(chars, fallback_chars)
            keeps = keeps_text or (structural and sheet_read_later)
            # Третий ключ — разрешение равенства: при одинаковом балле
            # старше тот, у кого есть структура. Но только если он её не
            # купил ценой потерянного текста, иначе оговорка о сохранности
            # обходилась бы через ничью.
            candidates.append((result, score, strategy, coverage))
            best, best_score, best_strategy, _ = max(candidates, key=candidate_rank)

            if score.value >= self.threshold:
                if keeps and not units_complete:
                    logger.info("Уровень %d не подтвердил все исходные единицы — пробуем следующий", strategy.level)
                    continue
                # Останавливает лестницу только измеренная сохранность:
                # старшинство авансом (лист прочитает чертёжная ветка) не
                # повод пропускать следующие уровни.
                if keeps_text and units_complete:
                    break
                if structural:
                    logger.info(
                        "Уровень %d (%s) дотянул до порога, но принёс %d символов "
                        "против %d у фоллбэка — пробуем следующий уровень",
                        strategy.level, strategy.name, chars, fallback_chars,
                    )
                    continue
                # Фоллбэк не повод останавливаться. Плоское распознавание не
                # вернёт ни таблицы, ни формулы, ни полей чертежа, а высокая
                # оценка у него означает всего лишь «символов много»: на
                # странице с формулами Tesseract выдаёт «д = —fz+ay» и по
                # объёму текста выглядит успешным. Уровень остаётся в
                # кандидатах и победит, если следующий не справится.
                logger.info(
                    "Уровень %d (%s) дотянул до порога, но это фоллбэк — "
                    "пробуем следующий уровень",
                    strategy.level, strategy.name,
                )

        if best is None or best_score is None or best_strategy is None:
            raise ParserFailed(
                f"Все применимые уровни не справились с {context.uri}: "
                + "; ".join(f"{a.strategy}: {a.error}" for a in attempts if a.error)
            )

        # Победивший фоллбэк при живом структурном уровне — это деградация,
        # а не нормальный исход. Молчать о ней нельзя: снаружи документ
        # выглядит разобранным, а в индекс уехала каша.
        fallback_won = best.is_fallback and structural_seen
        if fallback_won:
            # Балл победителя пересчитывается честно: структуру на этом
            # документе кто-то вернул, значит её отсутствие здесь — не
            # особенность документа, а неполнота разбора. В отчёте должно
            # стоять то же число, по которому принималось бы решение.
            best_score = score_result(
                best, context, exhaustive=best_strategy.exhaustive, expect_structure=True
            )
            for attempt in attempts:
                if (attempt.level == best_strategy.level
                        and attempt.strategy == best_strategy.name
                        and attempt.score is not None):
                    attempt.score = best_score
            note = (
                f"фоллбэк выиграл сравнение: {best_strategy.name} принёс "
                f"{text_chars(best)} символов без структуры"
            )
            if note not in best.degraded:
                best.degraded.append(note)
            logger.warning("Разбор %s деградировал: %s", context.uri, note)

        # Старшинство выдано авансом: лист ещё не прочитан. Лучший фоллбэк
        # уезжает вместе с результатом, и после чтения листа моделью объём
        # сверяется заново — уже по тому, что чертёжная ветка принесла.
        reserve: Optional[ParseResult] = None
        if (sheet_read_later and not best.is_fallback
                and not self._keeps_content(text_chars(best), fallback_chars)):
            fallbacks = [c for c in candidates if c[0].is_fallback]
            if fallbacks:
                reserve = max(fallbacks, key=lambda c: (text_chars(c[0]), c[1].value))[0]
                best.fallback_reserve = reserve
                logger.info(
                    "Лист-чертёж %s: разбор с %d символами оставлен старшим — лист "
                    "прочитает чертёжная ветка; %s (%d символов) в запасе",
                    context.uri, text_chars(best), reserve.parser_name, text_chars(reserve),
                )

        best.ladder = {
            "chosen_level": best_strategy.level,
            "chosen_strategy": best_strategy.name,
            "threshold": self.threshold,
            "classification": classification.as_dict(),
            "attempts": [a.as_dict() for a in attempts],
            "fallback_won": fallback_won,
            "degraded": list(best.degraded),
        }
        if reserve is not None:
            best.ladder["drawing_deferred"] = {
                "reserve_strategy": reserve.parser_name,
                "reserve_chars": text_chars(reserve),
            }
        logger.info(
            "Выбран уровень %d (%s) с оценкой %.3f из %d попыток",
            best_strategy.level, best_strategy.name, best_score.value, len(attempts),
        )
        return best

    @staticmethod
    def _keeps_content(chars: int, fallback_chars: int) -> bool:
        """Структурный разбор не потерял текст относительно фоллбэка."""
        if fallback_chars <= 0:
            return True
        return chars >= fallback_chars * settings.LADDER_CONTENT_KEEP_RATIO

    # ------------------------------------------------------------ внутреннее
    def _applicable(self, context: DocumentContext) -> List[Strategy]:
        applicable: List[Strategy] = []
        for strategy in self.strategies:
            try:
                if strategy.applicable(context):
                    applicable.append(strategy)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Проверка применимости %s не удалась: %s", strategy.name, exc
                )
        return applicable


class LadderParserProvider(DocumentParserProvider):
    """
    Провайдер разбора поверх лестницы. Подставляется вместо прямого вызова
    MinerU, поэтому задача обработки документа не знает про уровни вовсе.
    """

    def __init__(self, storage=None, router: Optional[LadderRouter] = None):
        self.storage = storage
        # Хранилище одно на задачу: и контекст документа, и уровни лестницы
        # работают через него, а не заводят каждый своё.
        self.router = router or LadderRouter(storage=storage)

    def parse(
        self, uri: str, file_type: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ParseResult:
        context = DocumentContext(
            uri=uri, file_type=file_type, metadata=metadata, storage=self.storage
        )
        return self.router.parse(context)
