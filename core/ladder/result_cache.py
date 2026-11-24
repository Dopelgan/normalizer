"""
Кэш результата лестницы по содержимому файла (LADDER_RESULT_CACHE).

Те же байты не проходят Tesseract и MinerU второй раз: ни точная копия
файла под другим именем, ни переобработка того же документа, ни отложенное
чтение листов. Кэшируется `ParseResult` лестницы (уровни 1–7) целиком, вместе
с `ladder`, `text_layer_stats` и `fallback_reserve`; нормализация, ветка
чертежей и финализация идут после попадания ровно как после разбора.

Ключ — только то, от чего результат лестницы зависит:
- sha256 исходных байтов и тип файла (те же байты как .jpg и как .pdf
  идут разными путями) и язык из метаданных;
- `PROCESSING_PROFILE_VERSION`;
- значения всех настроек, которые читает код лестницы (`KEY_SETTINGS`);
  тест сверяет список со статическим обходом импортов от
  `core.ladder.router` и падает на новой настройке, не внесённой ни в ключ,
  ни в `NOT_IN_KEY`;
- отпечаток исходников того же обхода импортов: любая правка кода
  лестницы — промах, а не чужой результат.

Чего в ключе нет: версии образа MinerU и Tesseract с tessdata. После их
обновления каталог кэша удаляется целиком (его можно удалить в любой
момент).

Привязка к документу. В результат MinerU попадают пути картинок под
`assets/<doc_id>/` и имя файла в сыром ответе. При попадании для другого
документа картинки копируются под его префикс, пути и имя переписываются,
а если в результате остался след чужого документа (doc_id, uri, имя) —
это промах. Так каждый документ сохраняет свой `doc_id`, `s3_fileid`,
путь и свои картинки.

Не кэшируется результат деградированный (`degraded` непуст, в том числе
LAYOUT_UNAVAILABLE и истёкший срок), с упавшими страницами и с любой
неудачной попыткой уровня: отказ сервиса не должен пережить сам отказ.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core import telemetry
from core.config import settings
from core.models.parse_result import ParseResult
from core.processing_contract import PROCESSING_PROFILE_VERSION

logger = logging.getLogger(__name__)

_FORMAT = "ladder-cache-v1"
HIT_STAGE = "parse.ladder_cache_hit"

# Настройки, которые читает код лестницы и которые меняют её результат.
KEY_SETTINGS: Tuple[str, ...] = (
    "ASSETS_PREFIX",
    # OCR (17.11): меняют текст распознавания; OCR_RUS_SECOND_MODEL читается через getattr — тест его не видит
    "OCR_ORDINAL_SUFFIX_FIX", "OCR_RUS_SECOND_MODEL",
    "BLANK_TABLE_OCR_DPI", "BLANK_TABLE_OCR_ENABLED", "BLANK_TABLE_OCR_MAX_PIXELS",
    "BLANK_TABLE_OCR_MAX_REGIONS", "BLANK_TABLE_OCR_MIN_ALPHA_SHARE",
    "BLANK_TABLE_OCR_MIN_CHARS", "BLANK_TABLE_OCR_MIN_CONFIDENCE",
    "DOC_CONVERT_ENABLED", "DOC_CONVERT_TIMEOUT",
    "DRAWING_AUTO_ROTATE", "DRAWING_BLOCK_CLEANUP", "DRAWING_DESKEW_LIMIT", "DRAWING_DIMENSION_FIDELITY",
    "DRAWING_FRAME_FOUR_SIDES", "DRAWING_FRAME_FROM_TITLE_BLOCK", "DRAWING_FRAME_STITCH",
    "DRAWING_INNER_FRAME_EXTENT", "DRAWING_MARGIN_DATE_FIX",
    # Карточки B, D, E: меняют запрос к модели, а значит и разбор.
    "DRAWING_VLM_STABLE", "DRAWING_SHEET_CONTENT_PASS", "DRAWING_SHEET_CONTENT_MAX_SIDE",
    "DRAWING_KEEP_REPEATS",
    "DRAWING_IMAGE_FULL_READ", "DRAWING_MAX_REGIONS", "DRAWING_SHEET_COMPACT_JSON",
    # D4: плитки крупного листа и дочитывание вырезками (через qwen_client)
    "DRAWING_NATIVE_MIN_SCALE", "DRAWING_REREAD_MAX_CROPS", "DRAWING_REREAD_MAX_SIDE",
    "DRAWING_SHEET_NATIVE_TILES",
    "DRAWING_SHEET_GUARDED", "DRAWING_SHEET_HIGH_TOKENS", "DRAWING_SHEET_MAX_ITEMS",
    "DRAWING_SHEET_MAX_TOKENS", "DRAWING_SHEET_SCHEMA", "DRAWING_SHEET_TILES",
    "DRAWING_SHEET_TILE_GRID", "DRAWING_SHEET_TILE_NEAR", "DRAWING_SHEET_TILE_OVERLAP",
    "DRAWING_SHEET_VALUE_MAX_CHARS", "DRAWING_SMALL_IMAGE_SIDE", "DRAWING_SMALL_PRINT_CARE",
    "DRAWING_TRANSCRIBE_MAX_SIDE", "DRAWING_TRANSCRIBE_MAX_TOKENS", "DRAWING_WHAT_DRAWN_MAX_TOKENS",
    "DRAWING_TRANSCRIBE_TILE_DEPTH", "DRAWING_TRANSCRIBE_TILE_GRID",
    "DRAWING_TRANSCRIBE_TILE_OVERLAP",
    "FRAME_TABLE_GUARD_ENABLED",
    "LADDER_CONTENT_KEEP_RATIO", "LADDER_DRAWING_OVER_OCR", "LADDER_ENABLED_LEVELS",
    "LADDER_MIN_REMAINING", "LADDER_PARALLEL_LAYOUT", "LADDER_SCORE_THRESHOLD",
    "LADDER_SKIP_OCR_ON_TEXT_LAYER", "LADDER_STRUCTURE_OVER_LENGTH",
    "MAX_IMAGE_PIXELS",
    "MINERU_BACKEND", "MINERU_DEFAULT_LANG", "MINERU_ENDPOINT", "MINERU_TIMEOUT",
    "MINERU_TIMEOUT_PER_PAGE",
    "OCR_BATCH_PAGES", "OCR_DESKEW_LIMIT", "OCR_DESPECKLE_MIN_SHARE", "OCR_DPI",
    "OCR_FULL_MAX_PAGES", "OCR_KEEP_STRUCTURED_TABLES", "OCR_LAYER_CONTINUATION_PASSES",
    "OCR_LAYER_DROP_LOW_CONFIDENCE", "OCR_LAYER_DROP_REPEATED", "OCR_LAYER_ENABLED",
    "OCR_LAYER_MIN_LINE_CONFIDENCE", "OCR_LAYER_REPEAT_CONTAIN", "OCR_LINE_MAX_THICKNESS",
    "OCR_LINE_MIN_SHARE", "OCR_LINE_REMOVAL_SKIP", "OCR_LINE_STROKE_GUARD",
    "OCR_LINE_STROKE_GUARD_REACH", "OCR_MAX_SIDE", "OCR_MIN_SIDE", "OCR_OSD_MIN_CONFIDENCE",
    "OCR_OTSU_THRESHOLD_STEP", "OCR_PER_LINE_LANGUAGE", "OCR_PREPROCESS_ENABLED",
    "OCR_REMOVE_TABLE_LINES", "OCR_SPLIT_WIDE_GAPS", "OCR_TABLE_LINE_MIN_CONFIDENCE",
    "OCR_TESSDATA_DIR", "OCR_TIME_BUDGET", "OCR_WORD_FIX_ENABLED",
    "PARSE_LAYOUT_UNAVAILABLE_NOTE", "PARSE_TASK_SOFT_TIME_LIMIT",
    "QWEN_ENDPOINT", "QWEN_MAX_TOKENS", "QWEN_MIN_TOKENS_PER_SECOND", "QWEN_MODEL",
    "QWEN_REGION_MAX_SIDE", "QWEN_REGION_UPSCALE", "QWEN_RETRIES", "QWEN_SHEET_MAX_SIDE",
    "QWEN_TEMPERATURE", "QWEN_TIMEOUT", "QWEN_TITLE_BLOCK_MAX_SIDE",
    # Строки распознавания с рамками в блоках результата (23.11)
    "RECOGNITION_LINES",
    "SOFFICE_BIN", "SPREADSHEET_MAX_CELLS", "STAMP_ROW_HEADER_FIX",
    "TABLE_SPLIT_GLUED_NUMBERS", "TABLE_VECTOR_LINES_KEPT", "TABLE_VLM_DPI",
    "TABLE_VLM_ESKD_SPEC", "TABLE_VLM_MAX_TABLES", "TABLE_VLM_RETRY_ENABLED", "TABLE_VLM_SPEC_DPI",
    "TABLE_VLM_SPEC_MAX_SIDE",
    "VLM_ENDPOINT", "VLM_PROMPT", "VLM_TIMEOUT",
)

# Читаются на пути лестницы, но результат не меняют: секреты (в ключ не
# идут вовсе), доступ к хранилищу и брокеру, места кэшей, параллельность и
# очередь к видеокарте, повтор задачи при отказе (результат при отказе не
# кэшируется).
NOT_IN_KEY: Tuple[str, ...] = (
    "ARTIFACTS_MOUNT", "CELERY_BROKER_URL",
    "VLM_SAME_REQUEST_ONCE",  # только повторное использование ответа модели, вывод тот же
    "DRAWING_PASSES_PARALLEL", "DRAWING_TRANSCRIBE_TILE_PARALLEL",
    "GPU_GATE_BUSY_HOURS_UTC", "GPU_GATE_ENABLED", "GPU_GATE_MINERU_SLOTS_DAY",
    "GPU_GATE_MINERU_SLOTS_NIGHT", "GPU_GATE_VLM_SLOTS_DAY", "GPU_GATE_VLM_SLOTS_NIGHT",
    "GPU_GATE_YIELD_ON_WAITING",
    "PARSE_OUTAGE_MAX_WAIT_SECONDS", "PARSE_REQUEUE_ON_OUTAGE",
    "QWEN_API_KEY", "QWEN_REQUEST_PRIORITY",
    "S3_ACCESS_KEY", "S3_ATTACHMENTS_BUCKET", "S3_BUCKET", "S3_ENDPOINT", "S3_READONLY",
    "S3_SECRET_KEY", "SOURCE_PREFIX", "SOURCE_PROBE_EXTENSIONS",
    "STORAGE_LOCAL_MOUNT", "STORAGE_TYPE", "TEMP_DIR", "TESSERACT_PAGE_WORKERS",
    "VLM_CACHE_V1_FALLBACK", "VLM_RESPONSE_CACHE", "VLM_RESPONSE_CACHE_DIR",
    "DRAWING_VLM_REJECTED_LOG",  # только журнал отказов
    "VLM_SHEET_CONCURRENCY",
)

_ROOT_MODULE = "core.ladder.router"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- отпечатки
def ladder_sources(root: Path = _SOURCE_ROOT) -> List[Path]:
    """
    Файлы, из которых собирается лестница: статический обход импортов от
    `core.ladder.router` в пределах пакета `core`. Тот же список даёт
    отпечаток кода и проверку полноты `KEY_SETTINGS` в тесте.
    """
    seen: Dict[str, Path] = {}
    todo = [_ROOT_MODULE]
    while todo:
        module = todo.pop()
        if module in seen or not (module == "core" or module.startswith("core.")):
            continue
        path = _module_file(root, module)
        if path is None:
            continue
        seen[module] = path
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                todo.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    parts = parts[: len(parts) - node.level + 1]
                    base = ".".join(parts + ([base] if base else []))
                todo.append(base)
                todo.extend(f"{base}.{alias.name}" for alias in node.names)
    return sorted(seen.values())


def _module_file(root: Path, module: str) -> Optional[Path]:
    base = root.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


_code_digest: Optional[str] = None
_code_lock = threading.Lock()


def code_digest() -> str:
    """Отпечаток исходников лестницы; считается один раз на процесс."""
    global _code_digest
    with _code_lock:
        if _code_digest is None:
            digest = hashlib.sha256()
            for path in ladder_sources():
                digest.update(path.relative_to(_SOURCE_ROOT).as_posix().encode("utf-8"))
                digest.update(b"\0")
                # Окончания строк не должны делать разными одинаковые исходники.
                digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
                digest.update(b"\0")
            _code_digest = digest.hexdigest()
        return _code_digest


def settings_snapshot() -> Dict[str, Any]:
    return {name: getattr(settings, name, None) for name in KEY_SETTINGS}


def cache_key(file_sha256: str, file_type: str, metadata: Optional[Dict[str, Any]]) -> str:
    material = {
        "format": _FORMAT,
        "file": file_sha256,
        "file_type": (file_type or "").lower(),
        "lang": (metadata or {}).get("lang"),
        "profile": PROCESSING_PROFILE_VERSION,
        "settings": settings_snapshot(),
        "code": code_digest(),
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def content_sha256(storage, uri: str) -> Optional[str]:
    """sha256 байтов файла (в том числе из S3); None — не удалось прочитать."""
    try:
        digest = hashlib.sha256()
        with storage.get_stream(uri) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception as exc:  # noqa: BLE001 — без хеша просто нет кэша
        logger.warning("Кэш лестницы: хеш %s не посчитан: %s", uri, exc)
        return None


# ------------------------------------------------------------------- место
def cache_dir() -> Path:
    name = settings.LADDER_RESULT_CACHE_DIR or str(
        Path(settings.ARTIFACTS_MOUNT) / "ladder-cache"
    )
    return Path(name)


def _path(key: str) -> Path:
    return cache_dir() / key[:2] / f"{key}.json"


def _mineru_name(uri: str, file_type: str) -> str:
    # Имя файла, под которым MinerU получает документ и ключует ответ;
    # то же правило, что `MinerUParser._filename`.
    name = uri.rstrip("/").split("/")[-1] or "document"
    if "." not in name:
        name = f"{name}.{(file_type or '').lower()}"
    stem, ext = name.rsplit(".", 1)
    return f"{stem}.{ext.lower()}"


def _stem(name: str) -> str:
    return name.rsplit(".", 1)[0]


def cacheable(result: ParseResult) -> bool:
    """Только чистый результат: без деградаций, упавших страниц и неудач уровней."""
    if result.degraded or result.failed_pages:
        return False
    attempts = (result.ladder or {}).get("attempts") or []
    if any(isinstance(a, dict) and a.get("error") for a in attempts):
        return False
    reserve = result.fallback_reserve
    if reserve is not None and (reserve.degraded or reserve.failed_pages):
        return False
    return True


# ------------------------------------------------------------ запись/чтение
def put(key: str, result: ParseResult, uri: str, file_type: str,
        metadata: Dict[str, Any]) -> bool:
    """Сохранить результат атомарно (временный файл и переименование)."""
    if not cacheable(result):
        return False
    reserve = result.fallback_reserve
    entry = {
        "format": _FORMAT,
        "key": key,
        "created": time.time(),
        "doc_id": metadata.get("doc_id") or "",
        "uri": uri,
        "asset_prefix": metadata.get("asset_prefix") or "",
        "mineru_name": _mineru_name(uri, file_type),
        "result": result.model_dump(mode="json"),
        "reserve": reserve.model_dump(mode="json") if reserve is not None else None,
    }
    path = _path(key)
    tmp: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{key}.", suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False)
        os.replace(tmp, path)
        tmp = None
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("Кэш лестницы: результат не записан: %s", exc)
        return False
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def get(key: str, storage, uri: str, file_type: str,
        metadata: Dict[str, Any]) -> Optional[ParseResult]:
    """Результат из кэша, привязанный к этому документу, или None (промах)."""
    path = _path(key)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Кэш лестницы: %s не прочитан: %s", path.name, exc)
        return None
    try:
        entry = json.loads(raw)
        if entry.get("format") != _FORMAT or entry.get("key") != key:
            raise ValueError("чужой формат или ключ")
        created = float(entry["created"])
        result_data = entry["result"]
        reserve_data = entry.get("reserve")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("Кэш лестницы: битая запись %s — промах: %s", path.name, exc)
        return None
    if time.time() - created > settings.LADDER_RESULT_CACHE_DAYS * 86400:
        _unlink(path)
        return None
    try:
        result_data, reserve_data = _rebind(
            entry, result_data, reserve_data, storage, uri, file_type, metadata
        )
        result = ParseResult.model_validate(result_data)
        if reserve_data is not None:
            result.fallback_reserve = ParseResult.model_validate(reserve_data)
    except _NotRebindable as exc:
        logger.warning("Кэш лестницы: запись %s не перенесена на %s — промах: %s",
                       path.name, uri, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — битая запись любого вида — промах
        logger.warning("Кэш лестницы: битая запись %s — промах: %s", path.name, exc)
        return None
    return result


class _NotRebindable(Exception):
    pass


def _rebind(entry, result_data, reserve_data, storage, uri, file_type, metadata):
    """Переписать привязку записи к документу: картинки, пути, имя файла."""
    old_prefix = entry.get("asset_prefix") or ""
    new_prefix = metadata.get("asset_prefix") or ""
    copied: Dict[str, str] = {}

    def relocate(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        if isinstance(value, str) and old_prefix and value.startswith(old_prefix):
            if value not in copied:
                copied[value] = _carry_asset(storage, value, new_prefix + value[len(old_prefix):])
            return copied[value]
        return value

    old_name = entry.get("mineru_name") or ""
    new_name = _mineru_name(uri, file_type)
    blobs = []
    for data in (result_data, reserve_data):
        if data is None:
            blobs.append(None)
            continue
        data = relocate(data)
        results = (data.get("raw_output") or {}).get("results")
        if old_name != new_name and isinstance(results, dict):
            # MinerU ключует ответ именем файла без расширения; полное имя
            # тоже принимается.
            renames = {old_name: new_name, _stem(old_name): _stem(new_name)}
            data["raw_output"]["results"] = {
                renames.get(name, name): value for name, value in results.items()
            }
        blobs.append(data)

    # След чужого документа — промах: лучше разобрать заново, чем отдать
    # документу чужие пути или имя.
    text = json.dumps(blobs, ensure_ascii=False)
    traces = []
    if entry.get("doc_id") and entry["doc_id"] != metadata.get("doc_id"):
        traces.append(entry["doc_id"])
    if entry.get("uri") and entry["uri"] != uri:
        traces.append(entry["uri"])
    if old_name and old_name != new_name:
        traces.append(json.dumps(old_name, ensure_ascii=False))
        if _stem(old_name) != _stem(new_name):
            traces.append(json.dumps(_stem(old_name), ensure_ascii=False))
    for trace in traces:
        if trace in text:
            raise _NotRebindable(f"в результате остался след {trace!r}")
    return blobs[0], blobs[1]


def _carry_asset(storage, source: str, target: str) -> str:
    try:
        if source == target:
            if not storage.exists(source):
                raise _NotRebindable(f"нет картинки {source}")
            return target
        with storage.get_stream(source) as stream:
            content = stream.read()
        storage.write_file(target, content)
        return target
    except _NotRebindable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _NotRebindable(f"картинка {source} не скопирована: {exc}") from exc


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# ------------------------------------------------------------------ срок
_PRUNE_EVERY = 86400.0
_last_prune = 0.0
_prune_lock = threading.Lock()


def prune(now: Optional[float] = None) -> int:
    """
    Удалить записи старше LADDER_RESULT_CACHE_DAYS (по времени файла), а
    если оставшиеся больше LADDER_RESULT_CACHE_MAX_GB — самые старые, пока
    объём не уложится в предел.
    """
    now = time.time() if now is None else now
    limit = settings.LADDER_RESULT_CACHE_DAYS * 86400
    removed = 0
    root = cache_dir()
    if not root.is_dir():
        return 0
    kept: List[Tuple[float, int, Path]] = []
    for path in _entries(root):
        try:
            stat = path.stat()
            if now - stat.st_mtime > limit:
                path.unlink()
                removed += 1
            else:
                kept.append((stat.st_mtime, stat.st_size, path))
        except OSError:
            continue
    total = sum(size for _, size, _ in kept)
    cap = settings.LADDER_RESULT_CACHE_MAX_GB * 1024 ** 3
    for _, size, path in sorted(kept, key=lambda item: item[0]):
        if total <= cap:
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        removed += 1
    if removed:
        logger.info("Кэш лестницы: удалено устаревших записей %d", removed)
    return removed


def _entries(root: Path) -> Iterable[Path]:
    for shard in root.iterdir():
        if shard.is_dir():
            yield from shard.glob("*.json")
            yield from shard.glob("*.tmp")


def _maybe_prune() -> None:
    # Отдельной задачи уборки в сервисе нет: раз в сутки на процесс уборку
    # делает тот, кто пишет в кэш.
    global _last_prune
    with _prune_lock:
        now = time.time()
        if now - _last_prune < _PRUNE_EVERY:
            return
        _last_prune = now
    try:
        prune(now)
    except Exception as exc:  # noqa: BLE001 — уборка не должна ронять разбор
        logger.warning("Кэш лестницы: уборка не удалась: %s", exc)


# ------------------------------------------------------------ точка входа
def parse(parser, storage, uri: str, file_type: str, metadata: Dict[str, Any],
          file_sha256: Optional[str]) -> ParseResult:
    """
    Разбор лестницей через кэш. Флаг выключен или хеша нет — ровно прежний
    вызов `parser.parse`. Попадание отмечается в таймингах
    `parse.ladder_cache_hit=1`; уровни 1–7 при этом не запускаются.
    """
    if not settings.LADDER_RESULT_CACHE or not file_sha256:
        return parser.parse(uri, file_type, metadata)
    # Отказ кэша никогда не роняет разбор: любая его ошибка — промах.
    key: Optional[str] = None
    cached: Optional[ParseResult] = None
    try:
        key = cache_key(file_sha256, file_type, metadata)
        cached = get(key, storage, uri, file_type, metadata)
    except Exception as exc:  # noqa: BLE001
        _warn_once(f"Кэш лестницы недоступен, разбор без кэша: {exc}")
        key, cached = None, None
    if cached is not None:
        telemetry.add(HIT_STAGE, 1)
        logger.info("Кэш лестницы: результат %s взят из кэша (%s)", uri, key[:12])
        return cached
    with _OutageWatch() as watch:
        result = parser.parse(uri, file_type, metadata)
    if key is not None and watch.reason:
        # Лестница могла уйти мимо сервиса (полный фоллбэк на Tesseract при
        # лежащем MinerU, пропуск уровня 5, отказ модели), а выбранный
        # результат об этом молчит: проигравший уровень унёс пометку с собой.
        logger.info("Кэш лестницы: %s не кэшируется — %s", uri, watch.reason)
        key = None
    if key is not None:
        try:
            if put(key, result, uri, file_type, metadata):
                _maybe_prune()
        except Exception as exc:  # noqa: BLE001
            _warn_once(f"Кэш лестницы: запись не удалась: {exc}")
    return result


class _OutageWatch(logging.Handler):
    """
    Следит за разбором: был ли отказ сервиса, пусть выбранный результат о нём
    и не знает. Признаки (любой — результат не кэшируется):
    - MinerU помечен недоступным до, во время или после разбора;
    - модель отказала как сервис (`outage.note_vlm(False)`);
    - любая запись уровня ERROR в логгерах `core.*` за время разбора — так
      пишут полный фоллбэк на Tesseract, таймаут MinerU и прочие отказы.
    """

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.reason = ""
        self._logger = logging.getLogger("core")

    def emit(self, record: logging.LogRecord) -> None:
        if not self.reason:
            self.reason = f"ошибка в логе {record.name}"

    def _state(self):
        from core import outage
        from core.providers.mineru_parser import MinerUParserProvider

        state = outage._state.get()
        return (MinerUParserProvider._unavailable_until,
                MinerUParserProvider.is_available(),
                state.vlm_down if state is not None else 0)

    def __enter__(self) -> "_OutageWatch":
        self._before = self._state()
        if not self._before[1]:
            self.reason = "MinerU помечен недоступным"
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc_info) -> None:
        self._logger.removeHandler(self)
        until, available, vlm_down = self._state()
        if self.reason:
            return
        if until != self._before[0] or not available:
            self.reason = "MinerU отказал во время разбора"
        elif vlm_down != self._before[2]:
            self.reason = "модель отказала во время разбора"


_warned = False


def _warn_once(message: str) -> None:
    """Одно предупреждение на процесс: сломанный кэш не засоряет лог."""
    global _warned
    if not _warned:
        _warned = True
        logger.warning(message)
