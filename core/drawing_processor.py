"""
Разбор чертежа: разметка листа зрением плюс чтение внешней моделью Qwen3-VL.

Как устроена ветка и почему именно так.

1. **Зрение размечает, модель читает.** Геометрия (`drawing_vision`) выправляет
   наклон, находит рамку формата, штамп и таблицы; модель читает лист целиком
   и каждую найденную область отдельно. Прогон шести способов на настоящем
   листе показал, почему нужны оба: сплошное чтение листа теряет таблицу
   целиком (спецификация — ноль строк из пяти), а нарезка по найденной сетке
   поднимает её до полного восстановления структуры.

2. **Модель внешняя.** Веса живут на отдельном хосте с GPU; конвейер ходит
   туда по HTTP. Если адрес не задан или хост молчит, чертёж остаётся
   неразобранным, и это видно: фрагмент не превращается в `drawing`, в логе
   стоит причина. Подменять разбор сплошным OCR по листу больше нечем —
   на живом листе он давал 24% попаданий при 85% мусора.

3. **Правила проверяют модель.** Категорию, допуск, единицу и характер
   размера определяют те же правила ГОСТ, что и для векторного чертежа
   (`vector_drawing`). Совпала категория модели с правилом — уверенность
   высокая; разошлись — поле остаётся, но уверенность честно ниже.

4. **Детектор YOLO и Florence в ветке больше не участвуют.** Стоковые веса
   DOTA не знают категорий чертежа и возвращали ноль областей, из-за чего лист
   доезжал до сплошного распознавания. Их код из проекта удалён; появятся
   веса под машиностроительный чертёж — детектор можно вернуть из истории
   git как источник областей вместо геометрии.

5. **Таблица составных частей не помещается в контракт.** Полей вида
   «строка спецификации» в нём нет, поэтому таблицы уходят в
   `structured_payload.tables` целиком, а в поля — только то, для чего
   категория предусмотрена.

Связи. Вызывается из `ingest.tasks` после нормализации:
`DrawingProcessor.enrich_fragments` заменяет фрагменты `image` на `drawing`
и, если документ — скан или фото чертежа, разбирает листы целиком
(страницы из `raster_drawing_pages` или `drawing_pages` в
`classification.signals`, не больше `DRAWING_MAX_SHEETS`). Векторный чертёж
приходит сюда уже готовым фрагментом `drawing` из текстового слоя
(`vector_drawing`) и не трогается. Лист, который `vector_drawing` признал
текстом в рамке ЕСКД (`text_in_frame`, флаг `DRAWING_TEXT_IN_FRAME_ENABLED`),
чертёжным не считается и полей чертежа не получает.

Вызовы модели (`QwenVisionClient`, `QWEN_ENDPOINT`): `read_sheet` — лист
целиком, затем по области, найденной геометрией (всего не больше
`DRAWING_MAX_REGIONS`): `read_title_block` для штампа, `read_table` для
остальных таблиц. Время на лист ограничено `DRAWING_TIME_BUDGET`. Если до
срока задачи не укладывается даже один вызов (`VLM_TIMEOUT`), фрагмент
остаётся с пометкой `DRAWING_SKIPPED_NO_TIME`, а неразобранные листы скана —
с `DRAWING_SHEETS_SKIPPED`: документ сохраняется, а не теряется целиком.

На листе, весь текст которого переписан моделью, записи распознавания того же
листа правятся её текстом (`core/drawing_transcript.py`,
`DRAWING_TRANSCRIPT_FIXES_OCR`), а повторы и шум распознавания убираются
(`core/drawing_stale_sources.py`, `DRAWING_SUPPRESS_STALE_ON_FULL_READ`).
"""

import contextvars
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import filetypes
from celery.exceptions import SoftTimeLimitExceeded

from core import deadline, outage
from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import (
    block_cleanup, dimension_fidelity, disputed_reread, drawing_text, drawing_vision, field_placement, gpu_gate,
    raster_drawing, reading_reconcile, sense_rules, title_block, vector_drawing,
)
from core.providers.qwen_client import (
    CATEGORIES, GUARD_MIN_TEXT_LINES, REJECT_EMPTY, QwenVisionClient,
)
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.vector_drawing import structure_ratio
from core.drawing_quality import apply_drawing_quality
from core.drawing_stale_sources import suppress_stale_sources
from core.drawing_transcript import apply_transcripts
from core.scan_text_rescan import poor_pages

logger = logging.getLogger(__name__)

# Жанр и источник из классификации лестницы (core.ladder.context). Строками,
# чтобы не тянуть сюда модуль лестницы ради двух констант.
GENRE_DRAWING = "drawing"
SOURCE_RASTER = "raster_only"

MODE_VISION_VLM = "vision_vlm"   # разметка зрением + чтение моделью
MODE_SHEET_ONLY = "sheet_only"   # только чтение листа целиком
MODE_NONE = "none"               # разобрать не удалось

# Способ получения полей -> (метод в контракте, уровень лестницы).
# Внешняя мультимодальная модель — это уровень 7 по определению лестницы.
_PROVENANCE_BY_MODE = {
    MODE_VISION_VLM: ("vision_plus_vlm", 7),
    MODE_SHEET_ONLY: ("vlm_full_page", 7),
}

# Уверенность в поле складывается из согласия модели с правилами ГОСТ.
_AGREED = 0.9            # модель и правило назвали одну категорию
_MODEL_ONLY = 0.75       # правило не знает такой категории (позиция, штамп)
_DISAGREED = 0.6         # разошлись: поле остаётся, но доверия меньше
_FROM_REGION = 0.85      # прочитано в области, найденной геометрией
_HANDWRITTEN = 0.5       # дописано от руки: проверить правилами нечем

# Классификатор «чертёж или текст» (DRAWING_PAGE_CLASSIFIER_ENABLED). В
# текстовом документе картинка меньше этой доли листа — логотип, печать,
# подпись, гриф «Утверждаю», QR-код. Qwen-VL делает из них «основную надпись»
# чертежа («Ген», «202д», «НТА-прибор»). Вид или схема в ТЗ и ПМ занимают
# десятую часть листа и больше и разбираются как чертёж.
_TEXT_DOC_MIN_FIGURE_AREA = 0.06
# Документ текстовый, если в среднем на страницу у него столько символов
# текста и таблиц. Скан сборочного чертежа, который растровый детектор не
# узнал, — это штамп и пара подписей: до порога он не дотягивает.
_TEXT_DOC_MIN_CHARS_PER_PAGE = 300

# Лист не читается: до срока задачи вызов модели не уложится
# (`DrawingProcessor._read_sheets`).
_NO_TIME = object()

# MuPDF не поддерживает работу из нескольких потоков одновременно, даже с
# разными документами, а листы одного документа могут читаться параллельно
# (VLM_SHEET_CONCURRENCY). Рендер страницы — доли секунды против десятков
# секунд модели, поэтому он просто идёт по очереди.
_PDF_RENDER_LOCK = threading.Lock()


class _Either:
    """Отмена, если поставлено любое из событий (`deadline.check_cancelled`)."""

    def __init__(self, *events: Optional[threading.Event]):
        self.events = [event for event in events if event is not None]

    def is_set(self) -> bool:
        return any(event.is_set() for event in self.events)


class _Deferred:
    """Проход, который выполняется там, где берётся его результат."""

    def __init__(self, function):
        self.function = function

    def result(self):
        return self.function()


class _Passes:
    """
    Проходы одного листа (DRAWING_PASSES_PARALLEL). При 1 проход
    выполняется в момент `result()`, то есть в прежнем порядке. Больше 1 —
    проходы стартуют сразу в пуле этого размера, а результаты всё равно
    берутся по одному в фиксированном порядке.

    Отмена: проход в потоке не делает следующего вызова модели, если
    отменена работа листа снаружи (`_read_sheets`) или сам лист закончился
    (`close`) — например, лист оказался пустым и остальные проходы уже не
    нужны, или задача оборвалась, пока лист ждал ответов.
    """

    def __init__(self, workers: int):
        self.abort = threading.Event()
        self.pool: Optional[ThreadPoolExecutor] = None
        if workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="drawing-pass")
            # Отмена снаружи хранится в контексте потока листа; проходам
            # нужна и она, и своя.
            self.cancel = _Either(deadline._cancel.get(), self.abort)

    def add(self, function):
        if self.pool is None:
            return _Deferred(function)
        # Срок задачи, часы бюджета листа и сборщик таймингов живут в
        # contextvars: потоку нужна копия контекста.
        return self.pool.submit(contextvars.copy_context().run, self._run, function)

    def _run(self, function):
        with deadline.cancel_on(self.cancel):
            deadline.check_cancelled()
            return function()

    def wait(self) -> None:
        """Дождаться начатых и стоящих в очереди проходов."""
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    def close(self) -> None:
        """Лист закончен: проходы из очереди не начинаются, начатые не зовут модель."""
        self.abort.set()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)


class DrawingProcessor:
    """Изображение чертежа -> структурированные поля -> фрагмент `drawing`."""

    # Вырезки для дочитывания после сведения D3 (`enrich_fragments`); вне
    # документа — None, и лист дочитывается сам.
    _reread_crops: Optional[Dict[Any, Any]] = None
    # Листы для «Что изображено» (DRAWING_WHAT_DRAWN); живут только на время документа.
    _what_drawn_sheets: Optional[Dict[Any, Dict[str, Any]]] = None

    def __init__(
        self,
        storage: Optional[StorageProvider] = None,
        qwen: Optional[QwenVisionClient] = None,
    ):
        self.storage = storage or StorageProviderFactory.default()
        self.qwen = qwen or QwenVisionClient()
        self.vision_enabled = settings.DRAWING_VISION_ENABLED
        # Таблицы MinerU по страницам: (bbox, строк). Только для строки учёта
        # прохода по таблице (`_log_table_pass`), на разбор не влияет.
        self._mineru_tables: Dict[int, List[Tuple[List[float], int]]] = {}
        # Строк слоя распознавания по страницам: по ним ограждённый проход
        # листа отличает пустой лист от пустого ответа (DRAWING_SHEET_GUARDED).
        self._layer_lines: Dict[int, int] = {}

    # ------------------------------------------------------------- публичное
    def process(
        self, image_ref: str, metadata: Dict[str, Any], page: int = 1,
        transcribe: bool = False, log_page: Optional[int] = None,
        text_only: bool = False, page_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает:
        {'detected_count', 'parsed_fields', 'avg_confidence', 'completeness',
         'mode', 'payload'}

        `log_page` — номер страницы только для лога: у вырезки MinerU номер
        страницы живёт во фрагменте, а не в `page` (там всегда 1, потому что
        вырезка — отдельная картинка). На разбор не влияет.

        `page_fields` — список прочтений ТОЙ ЖЕ страницы листа, накопленный
        прежними вызовами этого метода: страница и нарезанные с неё вырезки
        видов разбираются разными вызовами, а надпись у них одна. Список
        сквозной и пополняется здесь же — прочтения вырезок доходят до
        прохода по странице (`dimension_fidelity.adopt_page_signs`). Без
        него (None) сведение остаётся в точности прежним, внутри вызова.
        """
        # Бюджет листа меряется часами без ожидания мест на видеокарте
        # (GPU_GATE_ENABLED); у каждого листа свои часы.
        with gpu_gate.budget_clock():
            return self._process(image_ref, metadata, page, transcribe, log_page,
                                 text_only, page_fields)

    def _process(
        self, image_ref: str, metadata: Dict[str, Any], page: int,
        transcribe: bool, log_page: Optional[int], text_only: bool,
        page_fields: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not self.qwen.available:
            logger.error(
                "QWEN_ENDPOINT не задан — чертёж %s не разобран. Ветка чертежей "
                "работает только с внешней моделью.", image_ref,
            )
            return _empty(MODE_NONE)

        image = self._open(image_ref, page)
        if image is None:
            return _empty(MODE_NONE)

        started = time.monotonic()
        # Начало бюджета листа: без ожидания мест на видеокарте.
        budget_started = gpu_gate.budget_now()
        sheet = page if log_page is None else log_page
        layout = (
            drawing_vision.analyse(image) if self.vision_enabled
            else drawing_vision.plain(image)
        )

        fields: List[Dict[str, Any]] = []
        payload: Dict[str, Any] = {"layout": layout.as_dict(), "tables": []}
        payload["coordinate_reference"] = "deskewed_page"
        if settings.DRAWING_FIELD_PLACEMENT_V2:
            # Размер картинки до выправления: по нему обратный наклон
            # переводит рамки полей в доли исходной картинки (field_placement).
            payload["layout"]["source_size"] = [int(image.size[0]), int(image.size[1])]
        missing_regions: List[Dict[str, Any]] = []

        # Перечитывание плохого скана (DRAWING_RESCAN_POOR_OCR_PAGES) просит
        # у модели только текст страницы: надписи по категориям, штамп и
        # таблицы на такой странице уже прочитаны своей ветвью, и второй раз
        # платить за них временем незачем.
        #
        # Штамп важнее прочих таблиц: если времени хватит не на все области,
        # прочитан будет он.
        regions = [] if text_only else sorted(
            layout.regions,
            key=lambda r: r.kind != drawing_vision.KIND_TITLE_BLOCK,
        )
        if settings.DRAWING_STAMP_FALLBACK and not text_only and log_page is None:
            # Рамка и сетка штампа не нашлись: штамп — по месту по ГОСТ.
            fallback = drawing_vision.fallback_title_block(layout)
            if fallback is not None:
                regions.insert(0, fallback)
                payload["title_block_fallback"] = True
        gated = self._gate_table_regions(regions, page, sheet, whole_page=log_page is None)
        if gated:
            regions = [region for region in regions if not any(region is g for g in gated)]
            payload["regions_gated"] = len(gated)

        def sheet_pass():
            pass_started = time.monotonic()
            budget = settings.DRAWING_TIME_BUDGET
            # Срок передаётся только когда лист и правда может читаться частями:
            # с выключенным флагом вызов прохода остаётся в точности прежним.
            #
            # Из бюджета вычитается запас на проходы после чтения листа. Иначе
            # дочитывание частями съедает бюджет, частям прохода «весь текст
            # листа» времени не остаётся, и документ уходит человеку из-за
            # transcript_tile — хотя человеку качество чтения листа не адресовано.
            deadline_kwargs = (
                {"until": budget_started + budget - settings.DRAWING_SHEET_TILE_RESERVE}
                if settings.DRAWING_SHEET_TILES and budget else {}
            )
            reading = self.qwen.read_sheet(
                layout.image, hint=layout.hint(), **deadline_kwargs,
            )
            reads = int(settings.DRAWING_CONSENSUS_READS or 1)
            if reading is not None and reads > 1:
                # Режим приёмки (D4): лист читается ещё reads−1 раз, надписи
                # сводятся по значению и месту с пометкой согласия.
                runs = [reading.annotations]
                budget = settings.DRAWING_TIME_BUDGET
                for run in range(2, reads + 1):
                    if budget and gpu_gate.budget_now() + settings.VLM_TIMEOUT > budget_started + budget:
                        break
                    # Номер чтения в подсказке: иначе при нулевой температуре
                    # и кэше ответов повтор — тот же ответ, и сведение
                    # пометило бы согласием всё подряд.
                    more = self.qwen.read_sheet(
                        layout.image, hint=f"{layout.hint()} (чтение {run} из {reads})".strip(),
                        **deadline_kwargs,
                    )
                    if more is not None:
                        runs.append(more.annotations)
                reading.annotations = disputed_reread.consensus(runs)
                reading.raw = {**reading.raw, "consensus_reads": len(runs)}
            self._log_pass(
                sheet, "лист целиком", pass_started,
                self._sheet_produced(reading) if reading is not None
                else "лист не прочитан",
            )
            return reading

        # Весь текст листа отдельным проходом (DRAWING_IMAGE_FULL_READ):
        # разметка надписей по категориям теряет штамп, графы, номер листа
        # и примечания, а без них лист нельзя найти ни по обозначению, ни
        # по требованиям.
        def transcribe_pass():
            pass_started = time.monotonic()
            own_payload: Dict[str, Any] = {}
            own_missing: List[Dict[str, Any]] = []
            lines = self._transcribe(layout.image, own_payload, own_missing, budget_started)
            tiles = int(own_payload.get("transcript_tiles") or 0)
            self._log_pass(
                sheet, "весь текст листа", pass_started,
                ("строк текста %d%s" % (len(lines), ", частей %d" % tiles if tiles else ""))
                if lines else "текст не прочитан",
            )
            return lines, own_payload, own_missing

        def region_pass(region):
            # Срок проверяется, когда проход начинается, а не когда встал в
            # очередь: как и в цикле по очереди.
            if self._out_of_time(budget_started):
                return None
            pass_started = time.monotonic()
            own_fields: List[Dict[str, Any]] = []
            own_payload: Dict[str, Any] = {"tables": []}
            if region.kind == drawing_vision.KIND_TITLE_BLOCK:
                read = self._read_title_block(layout, region, own_fields, own_payload)
                with_side = bool((own_payload.get("side_column") or {}).get("read"))
                self._log_pass(
                    sheet, "штамп", pass_started,
                    ("граф %d%s" % (len(own_fields),
                                    ", с боковой колонкой" if with_side else ""))
                    if read else "штамп не прочитан",
                )
            else:
                piece = drawing_vision.crop(layout.image, region.bbox)
                read = self._read_table(piece, region, own_payload, int(sheet or 1))
                self._log_pass(
                    sheet, "таблица", pass_started,
                    ("строк %d" % len(own_payload["tables"][-1]["rows"])) if read
                    else "таблица не прочитана",
                )
                self._log_table_pass(sheet, page, region, own_payload if read else None,
                                     pass_started)
            return read, own_fields, own_payload

        # Проходы листа друг от друга не зависят: каждый читает свою вырезку
        # и пишет в своё. С DRAWING_PASSES_PARALLEL > 1 они идут разом, а
        # результаты сводятся ниже в прежнем порядке, поэтому запись листа
        # та же, что и по очереди. С 1 проход выполняется ровно там, где
        # берётся его результат, — порядок и сроки прежние.
        passes = _Passes(int(settings.DRAWING_PASSES_PARALLEL))
        try:
            sheet_job = None if text_only else passes.add(sheet_pass)
            transcribe_job = passes.add(transcribe_pass) if transcribe else None
            region_jobs = [passes.add(lambda region=region: region_pass(region)) for region in regions]

            reading = None
            if sheet_job is not None:
                reading = sheet_job.result()
                if reading is not None:
                    self._sheet_record(reading, payload)
            lines = None
            if transcribe_job is not None:
                lines, own_payload, own_missing = transcribe_job.result()
                payload.update(own_payload)
                missing_regions.extend(own_missing)
            if lines:
                payload["transcript_lines"] = lines
            if reading is not None and settings.DRAWING_SHEET_GUARDED:
                # Слой распознавания — только у листа целиком: у вырезки вида
                # `page` всегда 1, и чужая страница решала бы за неё.
                layer = self._layer_lines.get(int(page), 0) if log_page is None else 0
                guarded = self._guard_empty_sheet(
                    layout, reading, max(len(lines or []), layer), budget_started,
                )
                if guarded is not reading:
                    reading = guarded
                    self._sheet_record(reading, payload)
                guard = _guard_note(reading)
                if guard:
                    payload["sheet_guard"] = guard
                    self._sheet_record(reading, payload)
            if reading is None and not lines:
                logger.error("Модель не прочитала лист %s", image_ref)
                self._log_sheet(image_ref, sheet, MODE_NONE, 0, 0, 0, 0, started)
                return _empty(MODE_NONE)
            if reading is not None:
                payload["sheet_type"] = reading.sheet_type
                fields.extend(self._fields_from_annotations(reading.annotations))

            regions_read, skipped = 0, 0
            for region, job in zip(regions, region_jobs):
                outcome = job.result()
                if outcome is None:
                    skipped += 1
                    missing_regions.append({"kind": region.kind, "bbox": region.bbox, "reason": "time_budget"})
                    continue
                read, own_fields, own_payload = outcome
                fields.extend(own_fields)
                for key, value in own_payload.items():
                    if key == "tables":
                        payload["tables"].extend(value)
                    else:
                        payload[key] = value
                regions_read += int(read)
                if not read:
                    missing_regions.append({"kind": region.kind, "bbox": region.bbox, "reason": "region_not_read"})
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            # Обрыв задачи или отмена листа: соседей не ждать (`close`).
            raise
        except Exception:
            # Отказ одного прохода не обрывает соседние: они доходят до конца,
            # и лист падает так же, как упал бы при чтении по очереди.
            passes.wait()
            raise
        finally:
            passes.close()
        if skipped:
            logger.warning(
                "На %d областей листа %s не хватило бюджета времени (%d с)",
                skipped, image_ref, settings.DRAWING_TIME_BUDGET,
            )
            payload["regions_skipped"] = skipped

        fields = _deduplicate(fields)
        if settings.DRAWING_BLOCK_CLEANUP:
            # Подписи граф штампа («Лит.», «Масса») без значения — не поля.
            fields = block_cleanup.drop_caption_fields(fields)
        if settings.DRAWING_DIMENSION_FIDELITY:
            # Один размер, прочитанный и на листе целиком, и на вырезке вида:
            # остаётся прочтение со знаком «±» и со справочной звёздочкой.
            fields = dimension_fidelity.merge_readings(fields)
            if _category_strict():
                # Тот же размер, прочитанный проходом по надписям и
                # транскрипцией листа: «±» теряется в одном проходе и
                # остаётся в другом («1087±3» против «1087+3»), а
                # merge_readings сравнивает только поля между собой.
                dimension_fidelity.adopt_transcript_signs(
                    fields, payload.get("transcript_lines")
                )
        whole_sheet = log_page is None or image_ref in getattr(self, "_whole_images", ())
        if (settings.DRAWING_CORNER_ROUGHNESS and whole_sheet and not text_only
                and reading is not None and reading.sheet_type == "detail"):
            # Знак общей шероховатости в углу листа: проход листа его не видит.
            # Только лист целиком (у вырезки вида угла листа нет) и только
            # чертёж детали: общий знак ставится там, а на листе неизвестного
            # рода (текст, схема без штампа) лишний вызов искал бы знак зря.
            fields.extend(self._corner_roughness(layout, fields, payload, budget_started))
        if (settings.DRAWING_MARGIN_COLUMNS and whole_sheet and not text_only
                and reading is not None and reading.sheet_type in ("detail", "assembly")):
            # Графы поля подшивки и графа 26: повёрнутые надписи, которых проход
            # листа не читает. Только лист целиком (у вырезки вида поля
            # подшивки нет) и только чертёж детали или сборки — у них рамка
            # формы 1 с полем подшивки. Поля правятся на месте
            # (`margin_columns.merge`).
            self._margin_columns(layout, fields, payload, budget_started)
        crops = self._reread_crops
        if settings.DRAWING_DISPUTED_REREAD and fields and crops is not None:
            # Сведение D3 (DRAWING_READINGS_RECONCILED) идёт по всему документу
            # после листов и само решает, какие поля спорные: дочитывание —
            # после него (`_reread_reconciled`). Здесь — только рамки полей,
            # как открыть лист заново и сколько осталось от бюджета листа:
            # рендер в памяти до конца документа не держится.
            budget = settings.DRAWING_TIME_BUDGET
            crops[(image_ref, sheet)] = {
                "image_ref": image_ref, "page": page,
                "angle": layout.angle, "turn": layout.turn,
                "boxes": disputed_reread.boxes(fields),
                "left": budget_started + budget - gpu_gate.budget_now() if budget else None,
            }
        elif settings.DRAWING_DISPUTED_REREAD and fields:
            # Третье свидетельство по спорным числам (D4): вырезки в родном
            # разрешении. Здесь — после сведения прочтений листа и до того,
            # как поля уйдут соседним фрагментам страницы и привязке к виду.
            fields = self._reread_disputed(layout.image, fields, payload, budget_started)
        if settings.DRAWING_CONSENSUS_READS > 1:
            # Согласие N чтений листа нужно только выбору спорных полей выше;
            # в контракт поля оно не выходит (у D3 там свои значения).
            for item in fields:
                item.pop("agreement", None)
        if settings.DRAWING_DIMENSION_FIDELITY:
            if page_fields is not None:
                # Тот же размер, прочитанный в другом вызове `process()` по
                # той же странице: лист целиком и вырезка вида с него — это
                # два фрагмента, и merge_readings их не сводит. Список
                # сквозной, поэтому пополняется уже сведёнными полями.
                dimension_fidelity.adopt_page_signs(fields, page_fields)
                page_fields.extend(fields)
        # Связь величины с видом: заголовок из уже прочитанных надписей,
        # область — из геометрии листа. Поля не меняются, модель не
        # вызывается; выключенный флаг не добавляет ничего. Место важно:
        # `fields` к этой точке уже сведены и слиты по знакам, а привязка
        # ссылается на них по номеру.
        value_links = self._value_links(layout, fields)
        if value_links is not None:
            payload["value_links"] = value_links
        if _dimension_links() and fields and not text_only:
            # Связь размера с тем, что он измеряет: ключ `link` у поля. Место
            # то же, что у привязки к виду, — поля уже сведены.
            payload["dimension_links"] = self._dimension_links(
                image_ref, page, layout, fields, value_links, budget_started,
            )
        if (settings.DRAWING_WHAT_DRAWN and self._what_drawn_sheets is not None and whole_sheet
                and not text_only and fields
                and reading is not None and reading.sheet_type in ("detail", "assembly")):
            # «Что изображено» — после сведения, дочитывания и правил здравого
            # смысла документа (`_what_drawn_pass`), чтобы величины были
            # окончательными. Здесь — как открыть лист заново и сколько
            # осталось от бюджета листа: рендер до конца документа не держится.
            # Срок — по общим часам: время дочитывания D4 и прочих листов
            # между записью и вызовом идёт в счёт бюджета этого листа.
            budget = settings.DRAWING_TIME_BUDGET
            self._what_drawn_sheets[(image_ref, sheet)] = {
                "image_ref": image_ref, "page": page, "angle": layout.angle, "turn": layout.turn,
                "until": (time.monotonic() + budget_started + budget - gpu_gate.budget_now()
                          if budget else None),
            }
        mode = MODE_VISION_VLM if (self.vision_enabled and layout.regions) else MODE_SHEET_ONLY
        if not fields and not payload["tables"] and not payload.get("transcript_lines"):
            self._log_sheet(image_ref, sheet, MODE_NONE, 0, len(lines or []),
                            regions_read, len(regions), started)
            return _empty(MODE_NONE)

        payload["regions_read"] = regions_read
        payload["model"] = self.qwen.model
        self._log_sheet(image_ref, sheet, mode, len(fields), len(lines or []),
                        regions_read, len(regions), started)
        confidence = (
            sum(f["confidence"] for f in fields) / len(fields) if fields else 0.0
        )
        return {
            "detected_count": len(fields),
            "parsed_fields": fields,
            "avg_confidence": round(confidence, 3),
            # Покрытие найденных таблиц — независимый сигнал, но не
            # доказательство чтения всех размеров/видов листа.
            "completeness": None,
            "structure_score": structure_ratio(fields),
            "completeness_mechanisms": {
                "geometry_regions": {
                    "expected": len(regions), "read": regions_read,
                    "coverage": regions_read / len(regions) if regions else None,
                },
                "whole_sheet": {"coverage": None, "reason": "not_independently_measured"},
            },
            "missing_regions": missing_regions,
            "warnings": ["DRAWING_COMPLETENESS_UNKNOWN"],
            "mode": mode,
            "payload": payload,
        }

    # ------------------------------------------------- оборванный ответ листа
    def _guard_empty_sheet(self, layout, reading, text_lines: int, budget_started: float):
        """
        Пустой ответ по надписям при непустом листе (DRAWING_SHEET_GUARDED):
        у прохода «весь текст листа» или слоя распознавания страницы не меньше
        GUARD_MIN_TEXT_LINES строк. Такой ответ — сбой, а не чистый лист: он
        отбраковывается, и лист перечитывается частями один раз.
        """
        if reading.annotations or reading.tiles or text_lines < GUARD_MIN_TEXT_LINES:
            return reading
        logger.warning("Пустой ответ по надписям при %d строках текста листа: лист "
                       "перечитывается частями", text_lines)
        reading.rejected = REJECT_EMPTY
        budget = settings.DRAWING_TIME_BUDGET
        until = budget_started + budget - settings.DRAWING_SHEET_TILE_RESERVE if budget else None
        return self.qwen.reread_sheet_by_tiles(layout.image, reading, hint=layout.hint(), until=until)

    @staticmethod
    def _sheet_produced(reading) -> str:
        """Что дал проход «лист целиком» — для строки лога."""
        produced = "надписей %d" % len(reading.annotations)
        if reading.tiles:
            produced += ", частей %d" % reading.tiles
        if reading.cut and not reading.tiles:
            produced += ", ответ оборван"
        if reading.failed:
            produced += ", не прочитано частей %d" % len(reading.failed)
        if reading.lost:
            produced += ", пропало элементов %d" % reading.lost
        return produced

    @staticmethod
    def _sheet_record(reading, payload) -> None:
        """
        Что случилось с ответом прохода «лист целиком» — в payload листа.
        Это наш учёт, а не повод позвать человека.

        Человека зовут только за тем, чего он один и может решить: кластер,
        имя, конфликт версий. Насколько хорошо прочитан лист — наша забота,
        поэтому ни `warnings`, ни `missing_regions` здесь не появляются и
        документ от обрыва ответа не становится partial: открыть такой
        документ, найти его полным и получить вопрос «что с ним?» — чистый
        расход для заказчика.

        Но и `completed` при настоящей потере лист не изображает: своими
        данными он говорит правду. `sheet_annotations_complete`=`false` —
        прочтения потеряны; `sheet_reparse_candidate`=`true` — лист, который
        имеет смысл перечитать машинно (вот это и есть правильное
        автоматическое действие), с причиной в `sheet_reparse_reason`.

        Ключи пишутся всегда и ни от какого флага не зависят: это факты о
        разборе, а не эскалация. Под флагом учёт означал бы, что посчитать
        обрывы можно только грепом по логам воркера — а считать их нужно по
        сохранённым данным.
        """
        if reading.tiles:
            payload["sheet_tiles"] = reading.tiles
        if reading.cut:
            payload["sheet_answer_cut"] = True
        if reading.lost:
            payload["sheet_annotations_lost"] = reading.lost
        if reading.complete:
            # Ответ обрывался, но лист дочитан: терять нечего и считать
            # нечего. Такой лист в кандидаты на перечитывание не идёт.
            return
        payload["sheet_annotations_complete"] = False
        payload["sheet_reparse_candidate"] = True
        # Причина — та, что случилась: по ней видно, поможет ли перечитывание
        # вообще и что менять, если не поможет.
        if reading.failed:
            payload["sheet_reparse_reason"] = "sheet_tiles_unread"
            payload["sheet_unread_tiles"] = [list(box) for box in reading.failed]
        elif reading.cut:
            payload["sheet_reparse_reason"] = "sheet_answer_cut"
        else:
            payload["sheet_reparse_reason"] = "sheet_annotations_lost"

    # ------------------------------------------------- связь величин с видами
    @staticmethod
    def _value_links(layout, fields) -> Optional[Dict[str, Any]]:
        """
        Блок `value_links` или `None`, если связывать нечего.

        Модуль `core/providers/drawing_views.py` и настройку
        `DRAWING_VALUE_VIEW_LINKS` приносит ветка связывания величин с
        видами (`fix/drawing-value-links-20261114`). В этой сборке обе
        ветки слиты, и оба на месте — автор писал эту функцию в дереве, где
        их не было.

        Импорт всё равно оставлен ленивым и внутри проверки флага, а не
        наверху файла: импорт наверху в дереве без той ветки — это
        ImportError на загрузке модуля, то есть падает не разбор чертежа, а
        весь `core.drawing_processor`, и ни один документ не разбирается
        вовсе. Цена ленивого импорта — один поиск в `sys.modules` на лист,
        и менять его при сборке не за чем.

        Сам `attach` тоже возвращает `None` при выключенном флаге, так что
        отказ держится и структурно, независимо от этого места вызова;
        проверка флага здесь нужна ровно для того, чтобы до импорта дело не
        дошло, пока модуля нет.
        """
        if not _value_view_links():
            return None
        from core.providers import drawing_views

        return drawing_views.attach(layout, fields)

    # ------------------------------------------ связь размера с элементом
    def _dimension_links(self, image_ref, page, layout, fields, value_links, budget_started) -> Dict[str, Any]:
        """
        Ставит `link` полям листа и возвращает учёт для
        `structured_payload["dimension_links"]` (`DRAWING_DIMENSION_LINKS`).

        Геометрия берётся только у страницы векторного PDF: файл читается
        ещё раз (рендер его не хранит), и только под флагом. Вызов модели —
        если на него хватает бюджета листа.
        """
        from core.providers import dimension_links

        started = time.monotonic()
        geometry = self._vector_geometry(image_ref, page, layout)
        views = [view.get("bbox") for view in (value_links or {}).get("views", [])
                 if isinstance(view, dict) and view.get("bbox")]
        budget = settings.DRAWING_TIME_BUDGET

        def has_time() -> bool:
            return not budget or gpu_gate.budget_now() + settings.VLM_TIMEOUT <= budget_started + budget

        try:
            stats = dimension_links.apply(
                fields, image=layout.image,
                forbidden=dimension_links.forbidden_regions(layout),
                geometry=geometry,
                ask=self.qwen.read_dimension_links if self.qwen.available else None,
                views=views, has_time=has_time,
            )
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            raise
        except Exception as exc:  # noqa: BLE001 — связь не роняет лист
            logger.error("Связь размеров с элементами не построилась: %s", exc)
            for item in fields:
                item.pop("link", None)
            return {"error": type(exc).__name__}
        if settings.DRAWING_SENSE_RULES:
            # Связь «радиус» у величины без «R» (или «диаметр» без «⌀») —
            # связь неверного прочтения: значение остаётся, связь снимается.
            wrong = sense_rules.check_links(fields)
            if wrong:
                stats["dropped"]["type_mismatch"] = stats["dropped"].get("type_mismatch", 0) + wrong
                stats["linked"] = max(0, stats.get("linked", 0) - wrong)
        stats["seconds"] = round(time.monotonic() - started, 2)
        logger.info(
            "Связь размеров листа %s: величин %d, по геометрии %d, моделью %d, отказов %d за %.1f с",
            image_ref, stats["dimensions"], stats["linked_vector"], stats["linked_vlm"],
            sum(stats["dropped"].values()), stats["seconds"],
        )
        return stats

    def _vector_geometry(self, image_ref, page, layout):
        """Отрезки и стрелки страницы PDF в координатах выправленного листа или None."""
        data = self._read(image_ref)
        if data[:5] != b"%PDF-":
            return None
        try:
            import fitz
        except ImportError:  # pragma: no cover
            return None
        from core.providers import dimension_links

        with _PDF_RENDER_LOCK:
            try:
                document = fitz.open(stream=data, filetype="pdf")
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF чертежа не открылся для геометрии: %s", exc)
                return None
            try:
                if not 1 <= page <= document.page_count:
                    return None
                pdf_page = document[page - 1]
                frame = dimension_links.PageFrame.from_page(
                    pdf_page, layout.image.size, layout.angle, layout.turn,
                )
                return dimension_links.geometry_from_page(pdf_page, frame)
            finally:
                document.close()

    # ------------------------------------------------------------------- лог
    # Доказательство работы модели на чертеже читается прямо из лога: по
    # строке на каждый вызов Qwen-VL и строка-итог на лист. Содержимого
    # листа в лог не попадает — только номер страницы, вид прохода, счётчики,
    # имя модели и время.
    def _log_pass(self, sheet: Any, kind: str, started: float, produced: str) -> None:
        """Один вызов модели: что за проход, что дал и сколько шёл."""
        logger.info(
            "Qwen-VL: стр. %s, проход «%s», %s, модель %s, %.1f с",
            sheet, kind, produced, self.qwen.model, time.monotonic() - started,
        )

    def _log_table_pass(
        self, sheet: Any, page: int, region: drawing_vision.Region,
        payload: Optional[Dict[str, Any]], started: float,
    ) -> None:
        """
        Учёт прохода по таблице листа: размер области, что геометрия нашла в
        ней, сколько строк у MinerU в той же области и что дал проход. По этим
        строкам за день прода выбирается правило, какие проходы по таблице
        бесполезны (DRAWING_TABLE_PASS_GATE); само правило здесь не решается.
        """
        x1, y1, x2, y2 = (list(region.bbox) + [0.0] * 4)[:4]
        best_rows, best_share = 0, 0.0
        for bbox, rows in self._mineru_tables.get(int(page), []):
            share = _overlap_share(region.bbox, bbox)
            if share > best_share:
                best_rows, best_share = rows, share
        outcome = ("строк %d" % len(payload["tables"][-1]["rows"])) if payload else "пусто"
        logger.info(
            "Qwen-VL таблица: стр. %s, область %.4f листа (%.3f x %.3f), сетка %dx%d, "
            "у MinerU строк %d (перекрытие %.2f), итог %s, %.1f с",
            sheet, region.area, x2 - x1, y2 - y1, region.rows, region.cols,
            best_rows, best_share, outcome, time.monotonic() - started,
        )

    def _log_sheet(
        self, image_ref: str, sheet: Any, mode: str, fields: int, lines: int,
        regions_read: int, regions: int, started: float,
    ) -> None:
        """Итог по листу: один вызов `process` — одна строка."""
        logger.info(
            "Qwen-VL итог: лист %s стр. %s, режим %s, полей %d, строк текста %d, "
            "областей %d из %d, модель %s, %.1f с",
            image_ref, sheet, mode, fields, lines, regions_read, regions,
            self.qwen.model, time.monotonic() - started,
        )

    def build_fragment(
        self,
        drawing_result: Dict[str, Any],
        source: Fragment,
        metadata: Dict[str, Any],
    ) -> Optional[Fragment]:
        """Строит фрагмент типа `drawing` на месте фрагмента-изображения."""
        fields = drawing_result.get("parsed_fields") or []
        payload = dict(drawing_result.get("payload") or {})
        # Текст листа идёт в content.text, а в payload остаётся только число
        # строк: иначе потребитель индексировал бы его дважды.
        transcript = payload.pop("transcript_lines", None) or []
        if transcript:
            payload["transcript_line_count"] = len(transcript)
        if not fields and not payload.get("tables") and not transcript:
            return None

        mode = drawing_result.get("mode", MODE_SHEET_ONLY)
        method, level = _PROVENANCE_BY_MODE.get(mode, ("vlm_full_page", 7))
        payload.update({
            "detected_count": drawing_result.get("detected_count", 0),
            "recognition_mode": mode,
        })
        mechanisms = dict(drawing_result.get("completeness_mechanisms") or {})
        links_stats = payload.pop("dimension_links", None)
        if links_stats is not None:
            # Учёт связи размеров — служебный: в payload его адаптер RAG
            # вывел бы строкой в индекс (`contract_finalize._diagnostics_aside`),
            # поэтому он сразу лежит там, где тот флаг держит служебное.
            diagnostics = dict(mechanisms.get("drawing_diagnostics") or {})
            diagnostics["dimension_links"] = links_stats
            mechanisms["drawing_diagnostics"] = diagnostics
        fragment = Fragment(
            fragment_id=source.fragment_id,
            type="drawing",
            content=Content(
                # Переписанный текст листа — то, по чему лист находят поиском.
                text="\n".join(transcript) or None,
                image_ref=source.content.image_ref,
                structured_drawing_fields=[DrawingField(**f) for f in fields],
                structured_payload=payload,
            ),
            position=source.position.model_copy(deep=True) if source.position else Position(page=1, bbox=[0, 0, 1, 1]),
            section_title=source.section_title,
            section_path=list(source.section_path),
            confidence=round(float(drawing_result.get("avg_confidence", 0.0)), 3),
            completeness=drawing_result.get("completeness"),
            structure_score=drawing_result.get("structure_score"),
            completeness_mechanisms=mechanisms,
            missing_regions=[
                {**region, "page": source.position.page if source.position else 0,
                 "sheet": source.position.sheet if source.position else None}
                for region in drawing_result.get("missing_regions") or []
            ],
            warnings=list(dict.fromkeys(source.warnings + list(drawing_result.get("warnings") or []))),
            provenance=Provenance(method=method, strategy_level=level, source="drawing",
                                  source_unit_id=source.provenance.source_unit_id),
            graph_nodes=list(source.graph_nodes),
            relations=list(source.relations),
        )
        if fragment.position:
            fragment.position.bbox_exact = False
            fragment.position.page_start = fragment.position.page_end = fragment.position.page
        apply_drawing_quality(fragment)
        return fragment

    def enrich_fragments(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> List[Fragment]:
        """
        Заменяет image-фрагменты на drawing-фрагменты там, где получилось, и
        отдельно разбирает лист целиком, если сам документ — растровый
        чертёж. Без второго обработчик молчал на сканах и фотографиях
        чертежей: фрагмента `image` в таком документе нет.
        """
        # Рамки полей листов для дочитывания после сведения D3 (оба флага):
        # по (image_ref, страница) — так их находит фрагмент листа. Живут
        # только на время документа.
        self._reread_crops = (
            {} if settings.DRAWING_DISPUTED_REREAD and settings.DRAWING_READINGS_RECONCILED else None
        )
        # Картинки MinerU во весь лист — только этого документа: обработчик
        # живёт дольше документа, и ссылка прошлого документа не должна
        # делать листом вырезку следующего.
        self._whole_images = set()
        # Листы для «Что изображено» (DRAWING_WHAT_DRAWN): по (image_ref, страница
        # документа), как их находит фрагмент листа.
        self._what_drawn_sheets = {}
        try:
            return self._enrich_fragments(fragments, metadata)
        finally:
            self._reread_crops = None
            self._whole_images = set()
            self._what_drawn_sheets = None

    def _enrich_fragments(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> List[Fragment]:
        result: List[Fragment] = []
        # Прочтения по страницам, накопленные вызовами `process()`: вырезка
        # вида и сама страница читаются разными вызовами, а надпись у них
        # одна. Вырезки идут до прохода по странице, поэтому страница видит
        # их прочтения, а не наоборот.
        page_fields: Dict[int, List[Dict[str, Any]]] = {}
        text_document = settings.DRAWING_PAGE_CLASSIFIER_ENABLED and _is_text_document(fragments, metadata)
        # Страницы, которые ветвь «лист целиком» прочитает сама: мелкие
        # вырезки с них в модель не уходят (DRAWING_SKIP_SMALL_CROPS).
        whole_pages = self._pages_read_whole(metadata)
        for fragment in fragments:
            # Векторный чертёж уже разобран парсером по текстовому слою —
            # там точные значения и углы, читать нечего.
            if fragment.type == "drawing":
                result.append(fragment)
                continue
            if fragment.type != "image" or not fragment.content.image_ref:
                result.append(fragment)
                continue
            if text_document and _is_small_picture(fragment):
                # Логотип или печать в тексте: чертежом не становится и
                # вызова модели не стоит. Фрагмент остаётся картинкой.
                result.append(fragment)
                continue
            if self._crop_covered_by_whole_page(fragment, whole_pages):
                # Мелкая вырезка вида со страницы, которую ветвь «лист
                # целиком» прочитает сама: то же содержимое в более крупном
                # масштабе, новых пикселей в ней нет. Замеренное покрытие:
                # у вырезок с долей площади ниже порога все надписи и все
                # слова нашлись в проходе по странице.
                #
                # Пропускаются только мелкие вырезки. Широкий вариант
                # (пропускать с читаемой целиком страницы все вырезки) ждёт
                # сшивки чтений между вызовами `process()`: `merge_readings`
                # сливает показания внутри одного вызова, поэтому допуски и
                # знаки вида `⌀6,6-0,05`, `10°±1°` прочитались на крупной
                # вырезке, а в проходе по странице остались без формы
                # допуска — то есть потерялись бы.
                result.append(fragment)
                continue
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                # До срока задачи не уложится даже один вызов модели: лучше
                # сохранить документ с пометкой, чем потерять его целиком.
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_SKIPPED_NO_TIME"]))
                result.append(fragment)
                continue
            if ((settings.DRAWING_CORNER_ROUGHNESS or settings.DRAWING_MARGIN_COLUMNS
                    or settings.DRAWING_WHAT_DRAWN) and _covers_page(fragment)):
                # Картинка MinerU на всю страницу — лист целиком, хотя и
                # читается как вырезка: у неё есть угол рамки со знаком
                # общей шероховатости (`_corner_roughness`).
                self._whole_images.add(fragment.content.image_ref)
            try:
                sheet_page = int(fragment.position.page) if fragment.position else 0
                drawing_result = self.process(
                    fragment.content.image_ref, metadata,
                    log_page=fragment.position.page if fragment.position else None,
                    page_fields=page_fields.setdefault(sheet_page, []),
                )
                enriched = self.build_fragment(drawing_result, fragment, metadata)
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — чертёж не должен ронять документ
                logger.error("Обработка чертежа %s не удалась: %s",
                             fragment.content.image_ref, exc)
                enriched = None
            if enriched is None and self._source_is_raster_drawing(metadata):
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_NOT_RECOGNIZED"]))
            result.append(enriched or fragment)

        had_drawings = any(f.type == "drawing" for f in result)
        sheets = self._sheet_drawings(result, metadata, page_fields)
        pdf_full_read = self._full_read_pdf(metadata)
        if sheets and (pdf_full_read or (had_drawings and settings.DRAWING_SHEET_MISSING_PAGES)):
            # Дочитанный лист встаёт перед фрагментами своей страницы, а не
            # в начало документа: порядок страниц сохраняется.
            for sheet in sheets:
                page = sheet.position.page
                index = next((i for i, f in enumerate(result)
                              if f.position is not None and f.position.page >= page), len(result))
                result.insert(index, sheet)
        elif sheets:
            result[0:0] = sheets
        elif not any(f.type == "drawing" for f in result) and self._source_is_raster_drawing(metadata):
            for fragment in result:
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_NOT_RECOGNIZED"]))
        if settings.DRAWING_READINGS_RECONCILED:
            # До правки абзацев распознавания: слой страницы — одно из
            # прочтений, и сводится он в том виде, в каком его дал движок.
            reading_reconcile.apply(result)
            if self._reread_crops is not None:
                # D3 свёл прочтения, D4 дочитывает то, что осталось спорным.
                self._reread_reconciled(result)
        if settings.DRAWING_SENSE_RULES:
            # После сведения и дочитывания: правила смотрят на итог прочтений
            # (толщина из штампа против «50,5*», шероховатость поверх размера,
            # вид величины против вида связи). До размещения полей: рамки
            # вырезок здесь ещё в долях вырезки, как их понимает D3.
            changed = sense_rules.apply(result)
            if any(changed.values()):
                logger.info("Правила здравого смысла: %s", changed)
        if settings.DRAWING_WHAT_DRAWN and self._what_drawn_sheets:
            # После правил: величины листа окончательные, связи `link` проверены.
            # До размещения полей: рамки ещё в долях выправленного листа —
            # того, что уходит в модель.
            self._what_drawn_pass(result)
        if settings.DRAWING_RESCAN_POOR_OCR_PAGES:
            result.extend(self._rescan_poor_pages(result, metadata))
        # Модель не ответила документу ни разу и отказывала как сервис:
        # непрочитанные листы — свойство минуты, а не документа. С
        # PARSE_REQUEUE_ON_OUTAGE документ ставится в очередь (ServiceOutage).
        outage.check_vlm()
        if settings.DRAWING_TRANSCRIPT_FIXES_OCR:
            result = apply_transcripts(result)
        if settings.DRAWING_SUPPRESS_STALE_ON_FULL_READ:
            # После правки абзацев: исправленный абзац к этому моменту —
            # текст модели, и убирается как повтор, а не как расхождение.
            result = suppress_stale_sources(result)
        if settings.DRAWING_FIELD_PLACEMENT_V2:
            # Последним: сведение D3 и дочитывание D4 работают в координатах
            # вырезок и выправленного листа, а наружу поля уходят в долях страницы.
            result = field_placement.apply(result)
        return result

    # ------------------------------------------- перечитывание плохого скана
    def _rescan_poor_pages(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> List[Fragment]:
        """
        Страницы, у которых распознавание испорчено, читаются моделью
        (DRAWING_RESCAN_POOR_OCR_PAGES).

        Берётся тот же проход «весь текст листа», которым читаются листы при
        полном чтении, но без чтения надписей и областей: нужен только текст
        страницы. Дальше абзацы распознавания правит существующий шаг
        `core/drawing_transcript.py`, а повторы убирает
        `core/drawing_stale_sources.py`.

        Страница, текст которой модель уже переписала, повторно не читается.
        """
        source_uri = str(metadata.get("source_path") or "")
        if not source_uri or not self.qwen.available:
            return []
        transcribed = {
            int(fragment.position.page) for fragment in fragments
            if fragment.type == "drawing" and fragment.position is not None
            and (fragment.content.structured_payload or {}).get("transcript_line_count")
        }
        verdicts = poor_pages(
            fragments, settings.DRAWING_RESCAN_OCR_SCORE,
            settings.DRAWING_RESCAN_MIXED_SHARE,
        )
        pages = [page for page in verdicts if page not in transcribed]
        if not pages:
            return []
        sheets: List[Fragment] = []
        for page in pages[: settings.DRAWING_RESCAN_MAX_PAGES]:
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                logger.warning(
                    "Страница %d файла %s не перечитана: до срока задачи %.0f с",
                    page, source_uri, left,
                )
                break
            verdict = verdicts[page]
            logger.info(
                "Страница %d: распознавание испорчено (оценка %.2f, смешанных "
                "слов %.0f%%, признаки %s) — текст читает модель",
                page, verdict.score, verdict.mixed_share * 100,
                ",".join(verdict.reasons),
            )
            try:
                drawing_result = self.process(
                    source_uri, metadata, page, transcribe=True,
                    log_page=page, text_only=True,
                )
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — страница не роняет документ
                logger.error("Перечитывание страницы %d файла %s не удалось: %s",
                             page, source_uri, exc)
                continue
            # Ответ `process` — ещё не фрагмент: строки лежат в
            # `payload.transcript_lines`, а `transcript_line_count` считает
            # уже `build_fragment`. Неполное прочтение (часть листа не
            # прочитана) тоже идёт дальше: пропуски уходят в missing_regions,
            # а `transcript_complete=False` остаётся в payload.
            if not (drawing_result.get("payload") or {}).get("transcript_lines"):
                continue
            anchor = Fragment(
                fragment_id=f"{metadata.get('doc_id') or 'doc'}-rescan-{page:03d}",
                type="image",
                content=Content(image_ref=source_uri),
                position=Position(page=int(page), bbox=[0.0, 0.0, 1.0, 1.0], order=0),
                confidence=0.0, completeness=0.0,
                provenance=Provenance(method="vlm_full_page", strategy_level=7,
                                      source="drawing"),
            )
            sheet = self.build_fragment(drawing_result, anchor, metadata)
            if sheet is None:
                continue
            payload = sheet.content.structured_payload or {}
            payload["rescan_reason"] = verdict.as_dict()
            sheets.append(sheet)
        return sheets

    # -------------------------------------------------- лист-картинка целиком
    def _sheet_drawings(
        self, fragments: List[Fragment], metadata: Dict[str, Any],
        page_fields: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[Fragment]:
        """
        Разбор листов целиком, когда документ — скан или фото чертежа.

        `page_fields` — прочтения по страницам из уже разобранных вырезок
        той же страницы (см. `process`). Без него проход по странице
        сводит прочтения только внутри своего вызова, как раньше.

        У скана PDF разбирается каждый чертёжный лист (по классификации
        лестницы), а не только файл как одна картинка: PDF как растр не
        открывается, и ветка молча вернула бы пустой результат.
        """
        full_read = self._full_read_image(metadata) or self._full_read_pdf(metadata)
        drawings = [fragment for fragment in fragments if fragment.type == "drawing"]
        drawn_pages = {int(f.position.page) for f in drawings if f.position is not None}
        # Нарезанные MinerU картинки листа — это виды, а не лист: штамп,
        # номер листа и примечания остаются за ними. При полном чтении
        # картинки лист читается целиком и в этом случае.
        only_undrawn = False
        if not full_read and drawings:
            # Листы без единого фрагмента drawing (повёрнутый лист, который
            # MinerU принял за таблицу) иначе не читались моделью вовсе.
            # Фрагмент без страницы не даёт сказать, какой лист уже прочитан.
            if (not settings.DRAWING_SHEET_MISSING_PAGES
                    or len(drawn_pages) == 0 or any(f.position is None for f in drawings)
                    or not self._source_is_raster_drawing(metadata)):
                return []
            only_undrawn = True
        if not full_read and not self._source_is_raster_drawing(metadata):
            return []

        source_uri = str(metadata.get("source_path") or "")
        if not source_uri:
            return []

        signals = (metadata.get("classification") or {}).get("signals") or {}
        pages = list(signals.get("raster_drawing_pages") or signals.get("drawing_pages") or [1])
        if only_undrawn:
            pages = [page for page in pages if int(page) not in drawn_pages]
            if not pages:
                return []
        sheets: List[Fragment] = []
        skipped = pages[settings.DRAWING_MAX_SHEETS:]
        chosen = pages[: settings.DRAWING_MAX_SHEETS]
        # Списки прочтений берутся здесь, в порядке страниц: потоки листов
        # получают каждый свой список и общий словарь не трогают.
        fields_of = [
            None if page_fields is None else page_fields.setdefault(int(page), [])
            for page in chosen
        ]
        self._mineru_tables = _tables_by_page(fragments)
        self._layer_lines = _layer_lines_by_page(fragments)
        readings = self._read_sheets(source_uri, metadata, chosen, full_read, fields_of)
        # Листы, начатые и потерянные (исключение листа, таймаут чтения
        # модели, пустой ответ), — DRAWING_LOST_SHEETS_SKIPPED.
        lost: List[Any] = []
        # Фрагменты собираются строго в порядке страниц, в каком бы порядке
        # ни закончились вызовы модели.
        for index, (page, drawing_result) in enumerate(zip(chosen, readings)):
            if drawing_result is _NO_TIME:
                skipped = pages[index:]
                break
            if drawing_result is None:
                lost.append(page)
                continue
            anchor = Fragment(
                # Нулевой номер: лист целиком идёт перед своими фрагментами и не
                # сталкивается с нумерацией нормализатора, которая с единицы.
                fragment_id=f"{metadata.get('doc_id') or 'doc'}-frag-000"
                + ("" if int(page) == 1 else f"-{page}"),
                type="image",
                content=Content(image_ref=source_uri),
                position=Position(page=int(page), bbox=[0.0, 0.0, 1.0, 1.0], order=0),
                confidence=0.0, completeness=0.0,
                provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
            )
            sheet = self.build_fragment(drawing_result, anchor, metadata)
            if sheet is not None:
                sheets.append(sheet)
            else:
                lost.append(page)
        if settings.DRAWING_LOST_SHEETS_SKIPPED and lost:
            # Потерянный лист — тот же непрочитанный лист, что и лист за
            # пределом: без этого он пропадал молча.
            logger.warning("Листы %s стр. %s прочитать не удалось", source_uri,
                           ", ".join(str(page) for page in lost))
            skipped = sorted({int(page) for page in list(lost) + list(skipped)})
        carrier = sheets[0] if sheets else None
        if (carrier is None and skipped and settings.DRAWING_LOST_SHEETS_SKIPPED
                and lost and fragments):
            # Ни один лист не прочитан, но у документа есть другие фрагменты
            # (вырезки MinerU, текст): пометка встаёт на первый из них, иначе
            # потеря листов не доходит до предупреждений документа.
            carrier = fragments[0]
        if skipped and carrier is not None:
            # Неразобранные листы не пропадают молча: пометка ведёт на проверку.
            note = "DRAWING_SHEETS_SKIPPED:" + ",".join(str(page) for page in skipped)
            carrier.review_required = True
            carrier.warnings = list(dict.fromkeys(carrier.warnings + [note]))
        return sheets

    def _read_sheets(
        self, source_uri: str, metadata: Dict[str, Any], pages: List[Any],
        full_read: bool, fields_of: List[Optional[List[Dict[str, Any]]]],
    ) -> List[Any]:
        """
        Ответы `process` по листам в порядке `pages`: результат, `None` —
        лист не разобран, `_NO_TIME` — до срока задачи вызов не уложится,
        и этот лист и все следующие не читаются.

        С VLM_SHEET_CONCURRENCY > 1 листы читаются одновременно, но не
        больше этого числа разом. Результат тот же, что и по очереди: каждый
        лист читается тем же вызовом `process` со своей страницей и своим
        списком прочтений страницы, отказ листа ловится у самого листа и
        соседей не трогает, а ответы возвращаются в порядке страниц.

        Одна и та же страница дважды в списке — по очереди: второй вызов
        видит прочтения первого через общий список, и порядок там важен.
        """
        # Два события, а не одно. `no_time` — лист увидел нехватку срока:
        # следующие листы не начинаются, а уже начатые доходят до конца, как
        # при чтении по очереди. `abort` — задача оборвалась: начатые листы
        # бросают работу перед следующим вызовом модели.
        no_time = threading.Event()
        abort = threading.Event()

        def read(page: Any, fields: Optional[List[Dict[str, Any]]]) -> Any:
            # Срок проверяется, когда лист дождался места, а не когда встал в
            # очередь. Лист, который увидел нехватку времени, останавливает
            # и все следующие, как остановка цикла при чтении по очереди.
            if no_time.is_set() or abort.is_set():
                return _NO_TIME
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                no_time.set()
                logger.warning("Разбор листа %s стр. %d пропущен: до срока задачи %.0f с",
                               source_uri, page, left)
                return _NO_TIME
            try:
                with deadline.cancel_on(abort):
                    return self.process(
                        source_uri, metadata, page, transcribe=full_read, page_fields=fields,
                    )
            except SoftTimeLimitExceeded:
                raise
            except deadline.Cancelled:
                return _NO_TIME
            except Exception as exc:  # noqa: BLE001 — чертёж не должен ронять документ
                logger.error("Разбор листа %s стр. %d не удался: %s", source_uri, page, exc)
                return None

        workers = min(max(1, int(settings.VLM_SHEET_CONCURRENCY)), len(pages))
        if workers <= 1 or len({int(page) for page in pages}) != len(pages):
            readings: List[Any] = []
            for page, fields in zip(pages, fields_of):
                reading = read(page, fields)
                readings.append(reading)
                if reading is _NO_TIME:
                    break
            return readings + [_NO_TIME] * (len(pages) - len(readings))

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="drawing-sheet")
        try:
            # Срок задачи хранится в contextvars: потоку нужна копия контекста.
            futures = [
                pool.submit(contextvars.copy_context().run, read, page, fields)
                for page, fields in zip(pages, fields_of)
            ]
            return [future.result() for future in futures]
        except BaseException:
            # Обрыв задачи (SoftTimeLimitExceeded приходит в главный поток,
            # пока он ждёт ответы): листы из очереди не начинаются, а начатые
            # бросают работу перед следующим вызовом модели
            # (`deadline.check_cancelled` в `_ask_text` и перед повтором в
            # `_post`). Запрос, уже ушедший в vLLM, прервать нечем: он
            # дожидается ответа или своего таймаута, и на этом лист кончается.
            abort.set()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _pages_read_whole(self, metadata: Dict[str, Any]) -> frozenset:
        """
        Номера страниц, которые ветвь «лист целиком» прочитает сама.

        Повторяет выбор страниц из `_sheet_drawings` для случая полного
        чтения: тот же список чертёжных страниц из классификации и тот же
        потолок `DRAWING_MAX_SHEETS`. Пустое множество — когда флаг
        выключен или полного чтения не будет, и тогда вырезки читаются как
        раньше.
        """
        if not settings.DRAWING_SKIP_SMALL_CROPS:
            return frozenset()
        if not (self._full_read_image(metadata) or self._full_read_pdf(metadata)):
            return frozenset()
        if not str(metadata.get("source_path") or ""):
            return frozenset()      # без источника лист целиком не открыть
        signals = (metadata.get("classification") or {}).get("signals") or {}
        pages = list(signals.get("raster_drawing_pages") or signals.get("drawing_pages") or [1])
        return frozenset(int(page) for page in pages[: settings.DRAWING_MAX_SHEETS])

    @staticmethod
    def _crop_covered_by_whole_page(fragment: Fragment, whole_pages: frozenset) -> bool:
        """
        Мелкая ли это вырезка со страницы, которая читается целиком.

        Доля площади берётся из bbox фрагмента в долях листа; без bbox или
        с bbox не в долях размер неизвестен, и вырезка читается как раньше.
        """
        if not whole_pages or fragment.position is None:
            return False
        if int(fragment.position.page) not in whole_pages:
            return False
        share = _page_area_share(fragment)
        return share is not None and share < settings.DRAWING_SKIP_SMALL_CROP_AREA

    def _full_read_image(self, metadata: Dict[str, Any]) -> bool:
        """
        Лист-картинка читается целиком со всем текстом
        (DRAWING_IMAGE_FULL_READ): по классификации это растровый чертёж,
        либо лестница назвала лист текстом, но растровый детектор видит на
        нём рамку формата (лист со штампом, где MinerU нашёл только
        таблицу: таблица и примечания дают много строк, и детектор зовёт
        такой лист страницей текста, но рамка у него есть).
        """
        if not settings.DRAWING_IMAGE_FULL_READ:
            return False
        if not filetypes.is_image(str(metadata.get("file_type") or "")):
            return False
        if self._source_is_raster_drawing(metadata):
            return True
        data = self._read(str(metadata.get("source_path") or ""))
        if not data:
            return False
        verdict = raster_drawing.analyse(data)
        return verdict.is_drawing or bool(verdict.signals.get("frame"))

    def _full_read_pdf(self, metadata: Dict[str, Any]) -> bool:
        """
        Скан чертежа в PDF читается целиком со всем текстом листа
        (DRAWING_PDF_FULL_READ): вырезки видов, которые нарезал MinerU, не
        несут ни штампа, ни технических требований, ни размеров вне вырезок.
        """
        if not settings.DRAWING_PDF_FULL_READ:
            return False
        if str(metadata.get("file_type") or "").lower().lstrip(".") != "pdf":
            return False
        return self._source_is_raster_drawing(metadata)

    def _source_is_raster_drawing(self, metadata: Dict[str, Any]) -> bool:
        """Растровый чертёж по классификации лестницы, иначе — по самому файлу."""
        classification = metadata.get("classification") or {}
        genre = classification.get("genre")
        if genre:
            return genre == GENRE_DRAWING and classification.get("source") == SOURCE_RASTER

        if not filetypes.is_image(str(metadata.get("file_type") or "")):
            return False
        data = self._read(str(metadata.get("source_path") or ""))
        if not data:
            return False
        return raster_drawing.analyse(data).is_drawing

    def _transcribe(self, image, payload, missing_regions, started) -> Optional[List[str]]:
        """
        Весь текст листа. С DRAWING_TRANSCRIBE_TILES плотный лист, текст
        которого не помещается в один ответ, дочитывается частями; часть,
        которую прочитать не удалось, — непрочитанная область листа, и лист
        не считается переписанным целиком (core/drawing_quality.py).
        """
        if not settings.DRAWING_TRANSCRIBE_TILES:
            return self._fidelity(self.qwen.transcribe(image))
        budget = settings.DRAWING_TIME_BUDGET
        transcript = self.qwen.transcribe_sheet(image, until=started + budget if budget else None)
        if transcript is None:
            return None
        if transcript.tiles:
            payload["transcript_tiles"] = transcript.tiles
        if not transcript.complete:
            payload["transcript_complete"] = False
            missing_regions.extend(
                {"kind": "transcript_tile", "bbox": box, "reason": reason}
                for box, reason in transcript.failed
            )
        return self._fidelity(transcript.lines)

    @staticmethod
    def _fidelity(lines: Optional[List[str]]) -> Optional[List[str]]:
        """Строки транскрипции с восстановленными числами и знаками."""
        if lines is None or not settings.DRAWING_DIMENSION_FIDELITY:
            return lines
        return dimension_fidelity.repair_lines(lines)

    def _out_of_time(self, started: float) -> bool:
        budget = settings.DRAWING_TIME_BUDGET
        return bool(budget) and (gpu_gate.budget_now() - started) > budget

    # ------------------------------------------------------------ чтение
    def _read_title_block(
        self,
        layout: drawing_vision.SheetLayout,
        region: drawing_vision.Region,
        fields: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """Штамп: графы превращаются в поля, таблица целиком — в payload."""
        if settings.DRAWING_TITLE_BLOCK_PASS:
            return self._read_title_block_fields(layout, region, fields, payload)
        piece = drawing_vision.crop(layout.image, region.bbox)
        block = self.qwen.read_title_block(piece)
        if block is None:
            logger.warning("Штамп не прочитан")
            return False

        payload["title_block"] = block.get("fields", {})
        if block.get("rows"):
            payload["tables"].append({
                "kind": drawing_vision.KIND_TITLE_BLOCK,
                "bbox": region.bbox,
                "rows": block["rows"],
            })

        for name, value in (block.get("fields") or {}).items():
            text = drawing_text.normalize(str(value))
            if settings.DRAWING_DIMENSION_FIDELITY and "масштаб" in str(name).lower():
                # «7:1» в графе «Масштаб» — это «1:1»: курсивная «1» похожа
                # на «7», а ряд масштабов по ГОСТ 2.302 узкий.
                text = dimension_fidelity.repair_scale(text)
            readable, _ = drawing_text.looks_like_text(text)
            if not readable:
                continue
            category = "material" if _is_material(name, text) else "title_block"
            fields.append(_field(
                category=category,
                text=text,
                bbox=region.bbox,
                confidence=_FROM_REGION,
                label=str(name),
            ))
        return True

    def _read_title_block_fields(
        self,
        layout: drawing_vision.SheetLayout,
        region: drawing_vision.Region,
        fields: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """
        Отдельный проход по штампу (DRAWING_TITLE_BLOCK_PASS).

        Два отличия от обычного чтения области. Вырезка штампа идёт в
        модель в своём, высоком разрешении: на ужатой вырезке путаются
        фамилии и код документа. И колонка дополнительных граф у левого
        поля читается отдельным запросом — в вырезку штампа она не входит
        никогда, а «Инв. № подл.» и «Перв. примен.» лежат именно там.
        """
        bbox = drawing_vision.title_block_crop(layout, region)
        payload["title_block_bbox"] = bbox
        piece = drawing_vision.crop(
            layout.image, bbox,
            upscale=settings.QWEN_TITLE_BLOCK_UPSCALE,
            max_side=settings.QWEN_TITLE_BLOCK_MAX_SIDE,
        )
        parsed = self.qwen.read_title_block_fields(piece)
        side = self._read_side_column(layout, payload)
        if parsed is None and side is None:
            logger.warning("Штамп не прочитан")
            return False
        parsed = _merge_blocks(parsed, side, margin_first=settings.DRAWING_MARGIN_COLUMNS)
        # Графы поля подшивки, прочитанные с боковой колонки, стоят в ней, а
        # не в области штампа (DRAWING_MARGIN_COLUMNS). Клетку каждой графы
        # потом уточняет проход граф подшивки, если он идёт на этом листе.
        side_box = None
        if settings.DRAWING_MARGIN_COLUMNS and side is not None:
            side_box = (payload.get("side_column") or {}).get("bbox")
        if settings.DRAWING_DIMENSION_FIDELITY:
            # «7:1» в графе «Масштаб» — это «1:1»: курсивная «1» похожа на
            # «7», а ряд масштабов по ГОСТ 2.302 узкий. Правка делается до
            # payload, чтобы в structured_payload и в полях стояло одно и
            # то же значение, а запись MinerU с тем же «7:1» не оказалась
            # повтором исправленного прочтения.
            _repair_scale_graph(parsed)

        payload["title_block"] = dict(parsed.get("fields") or {})
        handwritten = dict(parsed.get("handwritten") or {})
        if handwritten:
            payload["title_block_handwritten"] = handwritten
        if parsed.get("rows"):
            payload["tables"].append({
                "kind": drawing_vision.KIND_TITLE_BLOCK,
                "bbox": region.bbox,
                "rows": parsed["rows"],
            })

        for name, value, by_hand in title_block.entries(parsed):
            text = drawing_text.normalize(value)
            readable, _ = drawing_text.looks_like_text(text)
            if not readable:
                continue
            category = "material" if _is_material(name, text) else "title_block"
            fields.append(_field(
                category=category,
                text=title_block.mark(text, by_hand),
                bbox=side_box if side_box and name in _MARGIN_LABELS else region.bbox,
                # Рукописное прочитано с той же вырезки, но проверить его
                # правилами нечем: уверенность ниже печатной графы.
                confidence=_HANDWRITTEN if by_hand else _FROM_REGION,
                label=name,
            ))
        return True

    def _read_side_column(
        self, layout: drawing_vision.SheetLayout, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Колонка дополнительных граф у левого поля, повёрнутая под чтение."""
        if not settings.DRAWING_SIDE_COLUMN_READ:
            return None
        bbox = drawing_vision.side_column(layout)
        if bbox is None:
            return None
        piece = drawing_vision.upright(drawing_vision.crop(
            layout.image, bbox, upscale=settings.QWEN_TITLE_BLOCK_UPSCALE,
            padding=0.0, max_side=settings.QWEN_TITLE_BLOCK_MAX_SIDE,
        ))
        parsed = self.qwen.read_side_column(piece)
        payload["side_column"] = {"bbox": bbox, "read": parsed is not None}
        return parsed

    def _gate_table_regions(
        self, regions: List[drawing_vision.Region], page: int, sheet: Any, whole_page: bool,
    ) -> List[drawing_vision.Region]:
        """
        Таблицы листа, проход по которым не нужен (DRAWING_TABLE_PASS_GATE):
        MinerU страницу разобрал, а таблицы со строками в этой области нет.
        Без слоя распознавания страницы (картинка без MinerU, сбой разбора)
        и на вырезках не пропускается ничего: сравнивать не с чем.
        """
        if not settings.DRAWING_TABLE_PASS_GATE or not whole_page:
            return []
        if self._layer_lines.get(int(page), 0) <= 0:
            return []
        least = settings.DRAWING_TABLE_PASS_GATE_OVERLAP
        tables = [bbox for bbox, rows in self._mineru_tables.get(int(page), []) if rows > 0]
        gated = [
            region for region in regions
            if region.kind == drawing_vision.KIND_TABLE
            and not any(_overlap_share(region.bbox, bbox) >= least for bbox in tables)
        ]
        for region in gated:
            logger.info(
                "Qwen-VL таблица: стр. %s, область %.4f листа, у MinerU таблицы нет — проход пропущен",
                sheet, region.area,
            )
        return gated

    def _mineru_rows(self, region: drawing_vision.Region, page: int) -> int:
        """Строк у таблицы MinerU, которая накрывает область не меньше чем наполовину."""
        best_rows, best_share = 0, 0.0
        for bbox, rows in self._mineru_tables.get(int(page), []):
            share = _overlap_share(region.bbox, bbox)
            if share > best_share:
                best_rows, best_share = rows, share
        return best_rows if best_share >= 0.5 else 0

    def _read_table(
        self, piece, region: drawing_vision.Region, payload: Dict[str, Any], page: int = 1,
    ) -> bool:
        """
        Таблица со листа (спецификация, ведомость) уходит в payload целиком.
        Полей вида «строка спецификации» в контракте нет, и раскладывать её
        по `note` значило бы засорять индекс мусором.
        """
        expected = self._mineru_rows(region, page) if settings.DRAWING_SHEET_GUARDED else 0
        table = self.qwen.read_table(piece, expect_rows=expected) if expected else self.qwen.read_table(piece)
        if table is None or not table.get("rows"):
            return False
        payload["tables"].append({
            "kind": drawing_vision.KIND_TABLE,
            "bbox": region.bbox,
            "title": table.get("title", ""),
            "headers": table.get("headers", []),
            "rows": table["rows"],
        })
        return True

    def _fields_from_annotations(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Надписи, прочитанные моделью на листе целиком -> поля контракта."""
        fields: List[Dict[str, Any]] = []
        for annotation in annotations:
            # Надпись строкой, а не объектом: модель иногда отдаёт голый
            # текст. Без проверки весь лист терялся на AttributeError,
            # который в `enrich_fragments` глотался молча.
            if not isinstance(annotation, dict):
                annotation = {"value": annotation}
            text = drawing_text.normalize(str(annotation.get("value") or ""))
            readable, reason = drawing_text.looks_like_text(text)
            if not readable:
                logger.debug("Надпись отброшена (%s): %r", reason, text)
                continue

            claimed = str(annotation.get("category") or "").strip().lower()
            by_rules, _ = vector_drawing.classify(text)
            category, confidence = _reconcile(claimed, by_rules, text)
            field = _field(
                category=category,
                text=text,
                bbox=_bbox(annotation.get("bbox")),
                confidence=confidence,
            )
            if (settings.DRAWING_CONSENSUS_READS > 1
                    and annotation.get("agreement") in ("agree", "single", "conflict")):
                # Согласие чтений листа (DRAWING_CONSENSUS_READS): по нему
                # дочитывание выбирает спорные поля.
                field["agreement"] = annotation["agreement"]
            fields.append(field)
        return fields

    def _corner_roughness(
        self, layout, fields: List[Dict[str, Any]], payload: Dict[str, Any], started: float,
    ) -> List[Dict[str, Any]]:
        """
        Знак общей шероховатости из правого верхнего угла рамки
        (DRAWING_CORNER_ROUGHNESS): одна вырезка — один короткий вызов.
        Лист уже выправлен и повёрнут штампом в правый нижний угол
        (`layout.turn`), и рамка `layout.frame` найдена заново на повёрнутом
        листе (`drawing_vision.analyse`), поэтому угол берётся у
        `layout.image` как есть. Вызов не начинается, если бюджета листа или
        срока задачи на него не хватает; место на видеокарте он берёт, как
        все вызовы (`QwenClient._post_attempts`), а таймаут чтения не
        повторяет (`QwenClient._ask_short`): резерв бюджета — один VLM_TIMEOUT.
        """
        budget = settings.DRAWING_TIME_BUDGET
        if budget and gpu_gate.budget_now() + settings.VLM_TIMEOUT > started + budget:
            payload["corner_roughness"] = "no_time"
            return []
        left = deadline.remaining()
        if left is not None and left < settings.VLM_TIMEOUT:
            payload["corner_roughness"] = "no_time"
            return []
        box = sense_rules.corner_box(layout.frame)
        try:
            piece = drawing_vision.crop(layout.image, box, padding=0.0,
                                        max_side=settings.DRAWING_REREAD_MAX_SIDE)
            value = sense_rules.corner_roughness(self.qwen.read_corner_roughness(piece))
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            raise
        except Exception as exc:  # noqa: BLE001 — угол не роняет лист
            logger.error("Знак шероховатости в углу листа не прочитан: %s", exc)
            return []
        if value is None or sense_rules.roughness_known(value, fields):
            payload["corner_roughness"] = "none" if value is None else "known"
            return []
        payload["corner_roughness"] = "added"
        # Значение голое, как у прочих полей шероховатости: подпись поля —
        # его категория, а сверка и описание листа сравнивают значение.
        logger.info("%s в углу листа: %s", sense_rules.CORNER_LABEL, value)
        return [_field("roughness", value, box, 0.7)]

    def _margin_columns(
        self, layout, fields: List[Dict[str, Any]], payload: Dict[str, Any], started: float,
    ) -> None:
        """
        Графы поля подшивки и графа 26 (DRAWING_MARGIN_COLUMNS): две вырезки
        по геометрии ГОСТ от внутренней рамки, одна картинка, один вызов.
        Вызов не начинается, если бюджета листа или срока задачи на него не
        хватает. Итог — `structured_payload.margin_columns`.
        """
        from core.providers import margin_columns

        geometry = margin_columns.geometry(layout)
        if geometry is None:
            payload["margin_columns"] = {"status": "no_frame"}
            return
        budget = settings.DRAWING_TIME_BUDGET
        left = deadline.remaining()
        if ((budget and gpu_gate.budget_now() + settings.VLM_TIMEOUT > started + budget)
                or (left is not None and left < settings.VLM_TIMEOUT)):
            payload["margin_columns"] = {"status": "no_time"}
            return
        try:
            strip = drawing_vision.crop(layout.image, geometry["strip"],
                                        upscale=settings.QWEN_TITLE_BLOCK_UPSCALE, padding=0.0,
                                        max_side=settings.QWEN_TITLE_BLOCK_MAX_SIDE)
            box = drawing_vision.crop(layout.image, geometry["top_box"],
                                      upscale=settings.QWEN_TITLE_BLOCK_UPSCALE, padding=0.0,
                                      max_side=settings.QWEN_TITLE_BLOCK_MAX_SIDE)
            parsed = margin_columns.parse(
                self.qwen.read_margin_columns(margin_columns.compose(strip, box)))
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            raise
        except Exception as exc:  # noqa: BLE001 — графы подшивки не роняют лист
            logger.error("Графы поля подшивки не прочитаны: %s", exc)
            payload["margin_columns"] = {"status": "error"}
            return
        reads = []
        for key, (value, by_hand) in parsed["columns"].items():
            text = drawing_text.normalize(value)
            reads.append(_field(
                category="title_block", text=title_block.mark(text, by_hand),
                bbox=geometry["cells"][key],
                confidence=_HANDWRITTEN if by_hand else _FROM_REGION,
                label=margin_columns.label(key),
            ))
        stats: Dict[str, Any] = margin_columns.merge(fields, reads)
        stats["designation"] = margin_columns.check_designation(fields, parsed["designation"])
        stats["status"] = "read"
        payload["margin_columns"] = stats
        logger.info("Графы поля подшивки: %s", stats)

    def _reread_disputed(
        self, image, fields: List[Dict[str, Any]], payload: Dict[str, Any], started: float,
    ) -> List[Dict[str, Any]]:
        """Спорные поля листа — вырезками в родном разрешении (DRAWING_DISPUTED_REREAD)."""
        budget = settings.DRAWING_TIME_BUDGET
        try:
            fields, stats = disputed_reread.apply(
                self.qwen, image, fields, until=started + budget if budget else None,
            )
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            raise
        except Exception as exc:  # noqa: BLE001 — дочитывание не роняет лист
            logger.error("Дочитывание спорных надписей не удалось: %s", exc)
            return fields
        if stats["crops"]:
            payload["disputed_reread"] = stats
        return fields

    def _reread_reconciled(self, fragments: List[Fragment]) -> None:
        """
        Дочитывание спорных полей после сведения D3: спорные — по его согласию
        («conflict», «single»), прочтение вырезки — ещё одно его прочтение.
        Каждое поле дочитывается один раз: в проходе листа вырезки только
        сохранены, а повтор поля листа, оставленный D3 во фрагменте вырезки
        вида (те же прочтения), получает итог поля листа без второго вызова.
        """
        done: Dict[Any, Dict[str, Any]] = {}
        drawings = [f for f in fragments if f.type == "drawing" and f.content.structured_drawing_fields]
        # Сначала листы целиком: их поля — те, чьи повторы есть у вырезок.
        drawings.sort(key=lambda f: not reading_reconcile._is_whole_sheet(f))
        for fragment in drawings:
            page = fragment.position.page if fragment.position else 1
            sheet = self._reread_crops.get((fragment.content.image_ref, page))
            if not sheet or not sheet["boxes"]:
                continue
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                logger.warning("Дочитывание спорных надписей пропущено: до срока задачи %.0f с", left)
                return
            fields = [f.model_dump() for f in fragment.content.structured_drawing_fields]
            keys = [(page, _readings_key(f)) for f in fields]
            allowed = set(sheet["boxes"])
            for n, key in enumerate(keys):
                if key[1] and key in done:
                    fields[n] = {**done[key], "bbox": fields[n]["bbox"], "category": fields[n]["category"],
                                 "link": fields[n].get("link")}
                    allowed.discard(disputed_reread._key(fields[n]["bbox"]))

            def render(sheet=sheet):
                # Тот же выправленный лист, что видел проход листа.
                image = self._open(sheet["image_ref"], sheet["page"])
                return None if image is None else drawing_vision.reproduce(image, sheet["angle"], sheet["turn"])

            # Остаток бюджета листа (DRAWING_TIME_BUDGET): на свои часы, без
            # ожидания мест на видеокарте; проверяется перед открытием листа
            # и перед каждой картинкой вырезок.
            with gpu_gate.budget_clock():
                until = None if sheet["left"] is None else gpu_gate.budget_now() + sheet["left"]
                try:
                    fields, stats = disputed_reread.apply(
                        self.qwen, render, fields, until=until, allowed=allowed,
                        reconcile=lambda item, text: reading_reconcile.witness(item, text, disputed_reread.decide),
                    )
                except (SoftTimeLimitExceeded, deadline.Cancelled):
                    raise
                except Exception as exc:  # noqa: BLE001 — дочитывание не роняет документ
                    logger.error("Дочитывание спорных надписей не удалось: %s", exc)
                    continue
            # Поля D3 дочитывание не снимает (группы — только у несведённых
            # полей), поэтому номера до и после совпадают.
            if len(fields) == len(keys):
                for key, item in zip(keys, fields):
                    if key[1] and key not in done:
                        done[key] = item
            fragment.content.structured_drawing_fields = [DrawingField(**f) for f in fields]
            if stats["crops"]:
                payload = dict(fragment.content.structured_payload or {})
                payload["disputed_reread"] = stats
                fragment.content.structured_payload = payload

    def _what_drawn_pass(self, fragments: List[Fragment]) -> None:
        """
        «Что изображено» (DRAWING_WHAT_DRAWN): один вызов на лист целиком
        детали или сборки. Вызов не начинается, если остатка бюджета листа
        или срока задачи на него не хватает; таймаут чтения не повторяется
        (`QwenClient.read_what_drawn`). Итог — `structured_payload.what_drawn`;
        отказ модели, пустой или непроверенный ответ ключа не дают.
        """
        from core.providers import what_drawn

        done = set()
        pass_started = time.monotonic()
        for fragment in fragments:
            if fragment.type != "drawing" or not fragment.content.structured_drawing_fields:
                continue
            page = fragment.position.page if fragment.position else 1
            key = (fragment.content.image_ref, page)
            sheet = self._what_drawn_sheets.get(key)
            if sheet is None or key in done:
                continue
            if len(done) >= WHAT_DRAWN_SHEETS_MAX:
                logger.info("«Что изображено»: прочитано %d листов документа, остальные пропущены", len(done))
                return
            done.add(key)
            fields = [f.model_dump() for f in fragment.content.structured_drawing_fields]
            prompt, dims = what_drawn.build_prompt(fields)
            if not dims:
                continue
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                logger.info("«Что изображено» пропущено: до срока задачи %.0f с", left)
                return
            if left is None and time.monotonic() - pass_started + settings.VLM_TIMEOUT > 2 * settings.VLM_TIMEOUT:
                # Без срока задачи весь проход — не дольше двух VLM_TIMEOUT.
                logger.info("«Что изображено»: время прохода вышло, остальные листы пропущены")
                return
            started = time.monotonic()
            # Остаток бюджета листа — на свои часы, без ожидания мест на
            # видеокарте, как у дочитывания D4.
            with gpu_gate.budget_clock():
                if sheet["until"] is not None and sheet["until"] - gpu_gate.budget_now() < settings.VLM_TIMEOUT:
                    logger.info("«Что изображено» пропущено для %s: бюджет листа исчерпан", key[0])
                    continue
                try:
                    image = self._open(sheet["image_ref"], sheet["page"])
                    if image is None:
                        continue
                    image = drawing_vision.reproduce(image, sheet["angle"], sheet["turn"])
                    result = what_drawn.parse(self.qwen.read_what_drawn(image, prompt), dims,
                                              what_drawn.sheet_numbers(fields))
                except (SoftTimeLimitExceeded, deadline.Cancelled):
                    raise
                except Exception as exc:  # noqa: BLE001 — описание листа не роняет документ
                    logger.error("«Что изображено» не прочитано: %s", exc)
                    continue
            if result is None:
                continue
            payload = dict(fragment.content.structured_payload or {})
            payload["what_drawn"] = result
            fragment.content.structured_payload = payload
            logger.info("«Что изображено» листа %s: видов %d, величин со смыслом %d из %d за %.1f с",
                        key[0], len(result["views"]), len(result["dimensions"]), len(dims),
                        time.monotonic() - started)

    # ------------------------------------------------------------ хранилище
    def _open(self, image_ref: str, page: int = 1):
        data = self._read(image_ref)
        if not data:
            return None
        if data[:5] == b"%PDF-":
            # Скан чертежа в PDF: лист рендерится в растр, иначе модель
            # получала бы «картинку», которая не открывается.
            return _render_pdf_page(data, page)
        image = raster_drawing.open_image(data)
        if image is None:
            logger.error("Чертёж %s не открылся как растр", image_ref)
        return image

    def _read(self, uri: str) -> bytes:
        try:
            return self.storage.read_bytes(uri)
        except Exception as exc:  # noqa: BLE001
            logger.error("Не удалось прочитать %s: %s", uri, exc)
            return b""


# ===========================================================================
# Поля
# ===========================================================================

# «Что изображено» (DRAWING_WHAT_DRAWN): не больше стольких листов на документ.
WHAT_DRAWN_SHEETS_MAX = 6


def _is_text_document(fragments: List[Fragment], metadata: Dict[str, Any]) -> bool:
    """
    Текстовый документ (договор, ТЗ, ТУ, письмо, протокол), а не чертёж:
    лестница не назвала его чертежом, и текста в нём в среднем не меньше
    `_TEXT_DOC_MIN_CHARS_PER_PAGE` символов на страницу.
    """
    if (metadata.get("classification") or {}).get("genre") == GENRE_DRAWING:
        return False
    chars, pages = 0, set()
    for fragment in fragments:
        if fragment.position is not None:
            pages.add(fragment.position.page)
        if fragment.type in ("text", "table"):
            chars += _visible_chars(fragment.content.text or "")
            table = fragment.content.table_data
            if table is not None:
                # У табличного фрагмента текста нет — всё в ячейках.
                cells = list(table.headers) + [cell for row in table.rows for cell in row]
                chars += sum(_visible_chars(str(cell)) for cell in cells if cell is not None)
    return bool(pages) and chars / len(pages) >= _TEXT_DOC_MIN_CHARS_PER_PAGE


def _tables_by_page(fragments: List[Fragment]) -> Dict[int, List[Tuple[List[float], int]]]:
    """Таблицы MinerU документа по страницам: рамка в долях листа и число строк."""
    tables: Dict[int, List[Tuple[List[float], int]]] = {}
    for fragment in fragments:
        data = fragment.content.table_data if fragment.content is not None else None
        if fragment.type != "table" or data is None or fragment.position is None:
            continue
        bbox = list(fragment.position.bbox or [])
        if len(bbox) != 4 or max(bbox) > 1.5:
            continue          # рамка не в долях листа: сравнить не с чем
        tables.setdefault(int(fragment.position.page), []).append((bbox, len(data.rows or [])))
    return tables


def _layer_lines_by_page(fragments: List[Fragment]) -> Dict[int, int]:
    """Непустых строк текста распознавания (текстовые фрагменты) по страницам."""
    lines: Dict[int, int] = {}
    for fragment in fragments:
        if fragment.type != "text" or fragment.position is None or not fragment.content.text:
            continue
        page = int(fragment.position.page)
        lines[page] = lines.get(page, 0) + sum(
            1 for line in fragment.content.text.splitlines() if line.strip()
        )
    return lines


def _guard_note(reading) -> Optional[Dict[str, Any]]:
    """
    Пометка листа, ответ по которому отбракован и не дочитан и частями
    (DRAWING_SHEET_GUARDED): причина и непрочитанные части. Лист принят или
    дочитан — пометки нет.
    """
    if reading is None or not reading.rejected or reading.complete:
        return None
    return {"rejected": reading.rejected, "tiles": reading.tiles,
            "unread_tiles": [list(box) for box in reading.failed]}


def _overlap_share(region: Sequence[float], other: Sequence[float]) -> float:
    """Доля площади `region`, накрытая `other` (обе рамки в долях листа)."""
    width = min(region[2], other[2]) - max(region[0], other[0])
    height = min(region[3], other[3]) - max(region[1], other[1])
    area = (region[2] - region[0]) * (region[3] - region[1])
    if width <= 0 or height <= 0 or area <= 0:
        return 0.0
    return width * height / area


def _visible_chars(text: str) -> int:
    return len("".join(text.split()))


def _value_view_links() -> bool:
    """
    Связывать ли величины с видами (`DRAWING_VALUE_VIEW_LINKS`).

    Настройку приносит ветка связывания величин с видами; пока её нет,
    поля в `Settings` нет тоже, и прямое обращение подняло бы AttributeError
    на каждом листе. Читается через `getattr` с умолчанием — тем же приёмом,
    что `drawing_genre.enabled()`. Отдельной функцией, а не выражением на
    месте, чтобы проводку можно было проверить тестом, не трогая singleton
    настроек: pydantic отказывается заводить неизвестное поле даже на запись.
    """
    return bool(getattr(settings, "DRAWING_VALUE_VIEW_LINKS", False))


def _covers_page(fragment: Fragment) -> bool:
    """Картинка занимает почти всю страницу (от 80 % площади): это лист, а не вид."""
    bbox = list(fragment.position.bbox or []) if fragment.position else []
    if len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) >= 0.8


def _dimension_links() -> bool:
    """Связывать ли размеры с измеряемыми элементами (`DRAWING_DIMENSION_LINKS`)."""
    return bool(getattr(settings, "DRAWING_DIMENSION_LINKS", False))


def _page_area_share(fragment: Fragment) -> Optional[float]:
    """
    Доля площади листа, которую занимает фрагмент, по его bbox в долях
    листа. None, когда bbox отсутствует или задан не в долях: размер
    фрагмента относительно листа тогда неизвестен.
    """
    bbox = fragment.position.bbox if fragment.position is not None else None
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if max(x0, y0, x1, y1) > 1.0 or min(x0, y0, x1, y1) < 0.0:
        return None
    return abs(x1 - x0) * abs(y1 - y0)


def _is_small_picture(fragment: Fragment) -> bool:
    """Картинка меньше `_TEXT_DOC_MIN_FIGURE_AREA` листа (bbox в долях листа)."""
    share = _page_area_share(fragment)
    return share is not None and share < _TEXT_DOC_MIN_FIGURE_AREA


def _readings_key(field: Dict[str, Any]) -> str:
    """Прочтения поля, сведённого D3 и прочитанного листом, одной строкой; иначе пусто."""
    readings = field.get("readings")
    if not isinstance(readings, list) or not any(r.get("source") == reading_reconcile.SHEET for r in readings):
        return ""
    return repr([(r.get("source"), r.get("value")) for r in readings])


def _render_pdf_page(data: bytes, page: int):
    """Страница PDF как растр в рабочем разрешении чертежей."""
    try:
        import fitz
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    with _PDF_RENDER_LOCK:
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            logger.error("PDF чертежа не открылся: %s", exc)
            return None
        try:
            if not 1 <= page <= document.page_count:
                return None
            pixmap = document[page - 1].get_pixmap(dpi=settings.DRAWING_RENDER_DPI)
            return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        finally:
            document.close()


def _field(
    category: str,
    text: str,
    bbox: List[float],
    confidence: float,
    label: str = "",
) -> Dict[str, Any]:
    """Поле контракта: значение от модели, разметка — правилами ГОСТ."""
    value = f"{label}: {text}" if label else text
    # Марка стали — не посадка: в «12Х18Н10Т» правило посадок видит «H10».
    no_tolerance = settings.DRAWING_MATERIAL_NO_TOLERANCE and category == "material"
    return {
        "category": category,
        "value": value,
        "tolerance": None if no_tolerance else vector_drawing.extract_tolerance(text),
        "nature": vector_drawing.determine_nature(text),
        "unit": vector_drawing.extract_unit(text, category),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "bbox": bbox,
        "source_of_tolerance": "unknown" if no_tolerance else vector_drawing.tolerance_source(text),
        # Надпись прочитана с самого листа. Значений контракта это не
        # расширяет: ровно то же ставит ветка детектора.
        "provenance": "detected_from_annotation",
    }


def _category_strict() -> bool:
    """
    Строгая категория поля (`DRAWING_FIELD_CATEGORY_STRICT`).

    Настройку и обе вспомогательные функции — `decisive_category` и
    `adopt_transcript_signs` — приносит ветка точности категорий
    (`fix/field-category-accuracy-20261114`). Здесь флаг читается через
    `getattr` с умолчанием, как это делает `drawing_genre.enabled()`, и
    ровно за этим: пока той ветки в дереве нет, поля в `Settings` нет тоже,
    а прямое `settings.DRAWING_FIELD_CATEGORY_STRICT` подняло бы
    AttributeError на каждом листе. Широкий `except Exception` в цикле
    обогащения проглотил бы его и превратил бы каждый чертёж в
    `DRAWING_NOT_RECOGNIZED` — то есть ровно в обратное тому, зачем этот
    код написан.

    Поэтому обе правки сидят строго внутри этой проверки: без своей ветки
    они мертвы и порядок слияния безопасен в любую сторону. `extra="ignore"`
    у `Settings` закрывает и случай, когда переменную окружения выставили
    раньше, чем приехал код: неизвестное поле просто не появляется.
    """
    return bool(getattr(settings, "DRAWING_FIELD_CATEGORY_STRICT", False))


def _reconcile(claimed: str, by_rules: str, text: str = "") -> tuple:
    """
    Категория поля: модель против правил. Правила детерминированы и знают
    ГОСТ, модель видит лист — поэтому спор решается не в чью-то пользу, а
    понижением уверенности.
    """
    # Исключение — записи, у которых по ЕСКД другого прочтения нет
    # (`vector_drawing.decisive_category`): на замере модель спорила с такой
    # нотацией 27 раз и не была права ни разу, правила — 26 из 26.
    #
    # Возвращается ответ самой `decisive_category`, а не `by_rules`: в эту
    # ветку и входят потому, что нотация ответила, и отдавать другое значит
    # противоречить причине входа. Разойтись эти два ответа могут только на
    # склеенном «№» с цифрой («2№3»: нотация говорит `position`, старый
    # `classify` — `note`), и сварной шов под «Примечанием» тратит выдачу,
    # не используя `position`, которую модель на этой нотации и заявляет.
    # С включённым флагом `classify` сам спрашивает `decisive_category`
    # первой, так что в собранном дереве эти ответы совпадают и выражение
    # ничего не меняет; расхождение остаётся только там, где флаг выключен,
    # а значит ветка недостижима.
    if text and _category_strict():
        decisive = vector_drawing.decisive_category(text)
        if decisive is not None:
            return decisive, _AGREED if claimed == decisive else _DISAGREED
    if claimed not in CATEGORIES:
        return by_rules, _MODEL_ONLY if by_rules != "note" else 0.4
    if claimed == by_rules:
        return claimed, _AGREED
    if by_rules == "note":
        # Правила такой категории просто не знают: номер позиции, графа
        # штампа. Это не спор.
        return claimed, _MODEL_ONLY
    return claimed, _DISAGREED


def _bbox(raw: Any) -> List[float]:
    """Координаты от модели: проверяются, а не принимаются на веру."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    try:
        coords = [float(c) for c in raw]
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]
    if max(coords) > 1.5:            # модель ответила пикселями — не угадываем
        return [0.0, 0.0, 1.0, 1.0]
    clamp = lambda c: max(0.0, min(1.0, c))  # noqa: E731
    x1, y1, x2, y2 = (clamp(c) for c in coords)
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 1.0, 1.0]
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _deduplicate(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Одна надпись, прочитанная и на листе, и в области, — одно поле.
    Остаётся то, у которого выше уверенность.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for field in fields:
        key = (field["category"], field["value"].strip().lower())
        if key not in best:
            best[key] = field
            order.append(key)
        elif field["confidence"] > best[key]["confidence"]:
            best[key] = field
    return [best[key] for key in order]


# Графы поля подшивки (ГОСТ 2.104, графы 19–25): в основной надписи их нет.
_MARGIN_KEYS = ("inventory", "inventory_dup", "replaced_inventory", "first_use", "reference",
                "signed_date")
_MARGIN_LABELS = frozenset(title_block.label(key) for key in _MARGIN_KEYS)


def _merge_blocks(
    block: Optional[Dict[str, Any]], side: Optional[Dict[str, Any]], margin_first: bool = False,
) -> Dict[str, Any]:
    """
    Графы штампа и графы боковой колонки в один разбор. Штамп главнее:
    одну и ту же графу (обозначение бывает и там и там) он переопределяет,
    а боковая колонка добавляет то, чего в штампе нет.

    `margin_first` (DRAWING_MARGIN_COLUMNS): графы поля подшивки берутся
    только с прочитанной боковой колонки. В вырезке штампа их нет, и то,
    что модель называет там «Подп. и дата» или «Инв. № подл.», — значения
    других граф: на листе крышки ТАДУ 687000.001.001 даты граф «Дата»
    штампа («2.09.75; 14.10.16») заменили дату графы подшивки «09.2025».
    """
    merged: Dict[str, Any] = {"fields": {}, "handwritten": {}, "unknown": {}, "rows": []}
    for part in (side, block):
        if not part:
            continue
        for key in ("fields", "handwritten", "unknown"):
            values = dict(part.get(key) or {})
            if margin_first and side and part is block and key != "unknown":
                for margin_key in _MARGIN_KEYS:
                    values.pop(margin_key, None)
            merged[key].update(values)
        merged["rows"] = part.get("rows") or merged["rows"]
    # Одно и то же значение, названное и печатным, и рукописным, остаётся
    # одним — рукописным. Разные значения сохраняются оба: выбрать между
    # «1085-02» из штампа и «1085» из боковой колонки нечем, а потерять
    # лучшее из двух хуже, чем отдать оба.
    for key, value in list(merged["fields"].items()):
        if merged["handwritten"].get(key, "").strip() == value.strip():
            merged["fields"].pop(key)
    return merged


def _repair_scale_graph(parsed: Dict[str, Any]) -> None:
    """
    Значение графы «Масштаб» в разборе штампа — приводится к ряду
    ГОСТ 2.302 (`dimension_fidelity.repair_scale`). Правится только эта
    графа: отношение в любом другом месте листа масштабом не является.
    """
    for part_name in ("fields", "handwritten"):
        part = parsed.get(part_name) or {}
        value = part.get(title_block.SCALE)
        if isinstance(value, str) and value:
            part[title_block.SCALE] = dimension_fidelity.repair_scale(value)


def _is_material(name: str, text: str) -> bool:
    lowered = f"{name} {text}".lower()
    return "материал" in lowered or vector_drawing.classify(text)[0] == "material"


def _empty(mode: str) -> Dict[str, Any]:
    return {
        "detected_count": 0,
        "parsed_fields": [],
        "avg_confidence": 0.0,
        "completeness": None,
        "structure_score": 0.0,
        "missing_regions": [],
        "warnings": ["DRAWING_NOT_RECOGNIZED"],
        "completeness_mechanisms": {},
        "mode": mode,
        "payload": {},
    }
