"""
Классификация документа до разбора и общий контекст для стратегий.

Лестница выбирает самую дешёвую применимую стратегию, а значит должна
что-то знать о документе ещё до того, как он разобран. Знание это дешёвое
и берётся из самого файла: есть ли исходник системы проектирования, есть
ли текстовый слой, чертёж это или таблица, годится ли растр без
восстановления.

Контекст читает файл один раз и раздаёт байты всем стратегиям: иначе
каждый уровень лестницы выкачивал бы его заново.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core import filetypes
from core.providers import raster_drawing
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.text_layer import TextLayer, extract_text_layer

logger = logging.getLogger(__name__)

# КЛ-1: по источнику.
SOURCE_CAD = "cad_source"          # есть исходник системы проектирования
SOURCE_VECTOR_TEXT = "vector_text"  # векторный документ с текстовым слоем
SOURCE_NATIVE = "native"            # формат с извлекаемым содержимым (docx, xlsx, txt)
SOURCE_RASTER = "raster_only"       # только растр

# КЛ-2: по роду документа.
GENRE_DRAWING = "drawing"
GENRE_SPREADSHEET = "spreadsheet"
GENRE_TEXT = "text"
GENRE_MIXED = "mixed"
GENRE_UNKNOWN = "unknown"

# Ниже этой оценки растр считается требующим восстановления (КЛ-3).
RESTORATION_THRESHOLD = 0.45


@dataclass
class Classification:
    """Результат классификации до разбора (разделы КЛ-1, КЛ-2, КЛ-3)."""

    source: str
    genre: str
    raster_quality: Optional[float] = None
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_restoration(self) -> bool:
        """Растр низкого качества: уровень 5 без восстановления даст мало."""
        return (
            self.source == SOURCE_RASTER
            and self.raster_quality is not None
            and self.raster_quality < RESTORATION_THRESHOLD
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "genre": self.genre,
            "raster_quality": self.raster_quality,
            "needs_restoration": self.needs_restoration,
            "signals": self.signals,
        }


class DocumentContext:
    """Один документ и всё, что стратегии о нём знают. Байты читаются раз."""

    def __init__(
        self,
        uri: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        storage: Optional[StorageProvider] = None,
        data: Optional[bytes] = None,
    ):
        self.uri = uri
        self.declared_type = filetypes.normalize(file_type)
        self._file_type = self.declared_type
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.storage = storage or StorageProviderFactory.default()
        self._data = data
        self._layer: Optional[TextLayer] = None
        self._layer_loaded = False
        self._classification: Optional[Classification] = None
        self._type_resolved = False
        self.type_mismatch: Optional[str] = None
        if data is not None:
            self._resolve_type()

    # --------------------------------------------------------------- данные
    @property
    def file_type(self) -> str:
        """
        Фактический тип файла. Уточняется по сигнатуре при первом обращении:
        стратегии спрашивают то `kind`, то сам тип, и ответ должен быть один
        и тот же независимо от порядка вопросов.
        """
        self._resolve_type()
        return self._file_type

    @property
    def kind(self) -> Optional[str]:
        """
        Род содержимого — им лестница отбирает применимые уровни. Считается
        по фактическому типу файла: расширение приходит от отправителя и
        бывает чужим, а от рода зависит, какая стратегия возьмётся за
        документ. Выбрать разбор DOCX для файла, который на самом деле PDF,
        дороже, чем один раз посмотреть на сигнатуру.
        """
        return filetypes.kind_of(self.file_type)

    @property
    def data(self) -> bytes:
        if self._data is None:
            self._data = self.storage.read_bytes(self.uri)
        return self._data

    def _resolve_type(self) -> None:
        """Уточнить тип по сигнатуре. Молчит, если содержимое недоступно."""
        if self._type_resolved:
            return
        try:
            head = self.data
        except Exception as exc:  # noqa: BLE001 — недоступный файл решает вызывающий
            logger.debug("Тип %s не уточнён, файл не прочитан: %s", self.uri, exc)
            return
        self._type_resolved = True
        verdict = filetypes.resolve_declared(self.declared_type, head)
        if verdict.mismatch:
            logger.info("%s: %s", self.uri, verdict.explanation)
            self.type_mismatch = verdict.explanation
            self.metadata.setdefault("declared_type", verdict.declared)
            self.metadata.setdefault("detected_type", verdict.detected)
        self._file_type = verdict.file_type or self.declared_type

    @property
    def text_layer(self) -> Optional[TextLayer]:
        """Текстовый слой PDF, если он есть и покрывает документ."""
        if not self._layer_loaded:
            self._layer_loaded = True
            if self.kind == filetypes.KIND_PDF:
                layer = extract_text_layer(self.data)
                self._layer = layer if (layer and layer.is_usable()) else None
        return self._layer

    # ------------------------------------------------------- классификация
    @property
    def classification(self) -> Classification:
        if self._classification is None:
            self._classification = classify(self)
        return self._classification

    def __repr__(self) -> str:  # pragma: no cover — только для логов
        return f"<DocumentContext {self.uri} type={self.file_type}>"


# ===========================================================================
# Классификация
# ===========================================================================

def classify(context: DocumentContext) -> Classification:
    """Три вопроса до разбора: откуда документ, что это и годен ли растр."""
    kind = context.kind

    if kind == filetypes.KIND_CAD:
        return Classification(source=SOURCE_CAD, genre=GENRE_DRAWING)

    if kind == filetypes.KIND_SPREADSHEET:
        return Classification(source=SOURCE_NATIVE, genre=GENRE_SPREADSHEET)

    if kind in (filetypes.KIND_OFFICE_TEXT, filetypes.KIND_PLAIN_TEXT):
        genre = GENRE_MIXED if kind == filetypes.KIND_OFFICE_TEXT else GENRE_TEXT
        return Classification(source=SOURCE_NATIVE, genre=genre)

    if kind == filetypes.KIND_IMAGE:
        # Картинка декодируется один раз на обе оценки: пригодность растра и
        # род листа. Раньше жанр здесь был всегда `unknown`, и сканированный
        # чертёж для лестницы ничем не отличался от снимка страницы — ветка
        # чертежей на него просто не включалась.
        image = raster_drawing.open_image(context.data)
        quality, signals = assess_raster(context.data, image=image)
        verdict = (
            raster_drawing.analyse_image(image) if image is not None
            else raster_drawing.RasterVerdict(
                raster_drawing.KIND_UNREADABLE, 0.0, "изображение не открылось"
            )
        )
        signals["raster_kind"] = verdict.kind
        signals["raster_kind_confidence"] = round(verdict.confidence, 3)
        signals["raster_kind_reason"] = verdict.reason
        return Classification(
            source=SOURCE_RASTER, genre=_genre_of_raster(verdict),
            raster_quality=quality, signals=signals,
        )

    if kind == filetypes.KIND_PDF:
        return _classify_pdf(context)

    return Classification(source=SOURCE_RASTER, genre=GENRE_UNKNOWN)


# Какой долей чертёжных листов документ становится чертежом целиком. Жанр
# правит и маршрутом разбора, и меткой доступа: `ROUTE_DRAWING` предлагает
# `confidential`. Целочисленное `len // 2` давало «чертёж» уже на одной
# странице из трёх — приложение к договору переводило в закрытые весь
# договор. Половина и больше листов — документ чертёжный; меньше — смешанный,
# и чертёжные листы всё равно разбираются как чертежи, каждый сам по себе.
_DRAWING_GENRE_SHARE = 0.5


def _genre_of_raster(verdict: "raster_drawing.RasterVerdict") -> str:
    """Род растрового листа -> жанр документа для лестницы."""
    if verdict.is_drawing:
        return GENRE_DRAWING
    if verdict.kind == raster_drawing.KIND_TEXT_SCAN:
        return GENRE_TEXT
    return GENRE_UNKNOWN


def _classify_pdf(context: DocumentContext) -> Classification:
    """PDF бывает и векторным, и обёрткой вокруг скана — это разные пути."""
    from core.providers.vector_drawing import detect_drawing_pages

    layer = context.text_layer
    signals: Dict[str, Any] = {}

    verdicts = detect_drawing_pages(context.data)
    drawing_pages = [p for p, v in verdicts.items() if v.is_drawing]
    signals["drawing_pages"] = drawing_pages
    signals["page_count"] = len(verdicts)
    share = len(drawing_pages) / len(verdicts) if verdicts else 0.0
    signals["drawing_share"] = round(share, 3)
    mostly_drawings = share >= _DRAWING_GENRE_SHARE

    if layer is None:
        # Текста в файле нет — значит внутри картинки, и качество растра
        # решает, нужен ли этап восстановления.
        quality, raster_signals = assess_raster(context.data, pdf=True)
        signals.update(raster_signals)
        if mostly_drawings:
            genre = GENRE_DRAWING
        elif drawing_pages:
            genre = GENRE_MIXED
        else:
            genre = GENRE_UNKNOWN
        return Classification(
            source=SOURCE_RASTER, genre=genre,
            raster_quality=quality, signals=signals,
        )

    signals["layer_chars"] = layer.char_count()

    if mostly_drawings:
        genre = GENRE_DRAWING
    elif drawing_pages:
        genre = GENRE_MIXED
    else:
        genre = GENRE_MIXED if _looks_structured(context.data, signals) else GENRE_TEXT

    return Classification(source=SOURCE_VECTOR_TEXT, genre=genre, signals=signals)


def _looks_structured(data: bytes, signals: Dict[str, Any]) -> bool:
    """
    Есть ли на страницах линейная графика — рамки таблиц, схемы. Если есть,
    одного текстового слоя мало: структуру он не описывает, и лестница
    должна подняться до разбора с детекцией областей.
    """
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        try:
            import fitz as pymupdf
        except ImportError:
            return False

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Не удалось оценить структурность PDF: %s", exc)
        return False

    ruled_pages = 0
    try:
        for index in range(min(document.page_count, 12)):
            try:
                drawings = document[index].get_drawings()
            except Exception:  # noqa: BLE001
                continue
            if _count_rules(drawings) >= 6:
                ruled_pages += 1
    finally:
        document.close()

    signals["ruled_pages"] = ruled_pages
    return ruled_pages > 0


def _count_rules(drawings) -> int:
    """Число прямых линий: рамки таблиц дают их пучками."""
    count = 0
    for item in drawings:
        rect = item.get("rect")
        if rect is None:
            continue
        try:
            width, height = abs(float(rect.width)), abs(float(rect.height))
        except Exception:  # noqa: BLE001
            continue
        if min(width, height) <= 2.0 and max(width, height) >= 20.0:
            count += 1
    return count


# ===========================================================================
# Оценка пригодности растра (КЛ-3 и этап В-1)
# ===========================================================================

def assess_raster(data: bytes, pdf: bool = False, image=None) -> tuple:
    """
    Оценка качества растра в [0, 1] и признаки, по которым она получена:
    разрешение, контраст, наклон, шум. Ниже RESTORATION_THRESHOLD документ
    отправляется на восстановление, а не сразу на распознавание.

    `image` — уже открытая картинка: декодировать многомегабайтный TIFF
    дважды только ради второй оценки незачем.
    """
    if image is None:
        image = _load_image(data, pdf=pdf)
    if image is None:
        return None, {"raster": "не удалось открыть"}

    try:
        import numpy
    except ImportError:  # pragma: no cover
        return None, {"raster": "numpy недоступен"}

    grey = numpy.asarray(image.convert("L"), dtype=numpy.float32)
    height, width = grey.shape

    megapixels = (width * height) / 1_000_000
    resolution_score = min(1.0, megapixels / 2.0)

    # Среднеквадратичный контраст: у выцветшей копии он низкий.
    contrast = float(grey.std()) / 128.0
    contrast_score = min(1.0, contrast)

    # Шум: насколько картинка отличается от собственного сглаживания.
    noise = _noise_level(grey, numpy)
    noise_score = max(0.0, 1.0 - noise / 24.0)

    skew = _skew_degrees(grey, numpy)
    skew_score = max(0.0, 1.0 - abs(skew) / 8.0)

    quality = round(
        0.30 * resolution_score + 0.30 * contrast_score
        + 0.20 * noise_score + 0.20 * skew_score,
        3,
    )
    signals = {
        "width": width, "height": height,
        "megapixels": round(megapixels, 2),
        "contrast": round(contrast, 3),
        "noise": round(noise, 2),
        "skew_degrees": round(skew, 2),
    }
    return quality, signals


def _load_image(data: bytes, pdf: bool = False):
    """Картинка или первая страница PDF в виде растра."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None

    if pdf:
        try:
            import pymupdf
        except ImportError:  # pragma: no cover
            try:
                import fitz as pymupdf
            except ImportError:
                return None
        try:
            document = pymupdf.open(stream=data, filetype="pdf")
            try:
                pixmap = document[0].get_pixmap(dpi=150)
                return Image.frombytes(
                    "RGB", (pixmap.width, pixmap.height), pixmap.samples
                )
            finally:
                document.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Не удалось отрендерить первую страницу PDF: %s", exc)
            return None

    import io
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as exc:  # noqa: BLE001
        logger.debug("Не удалось открыть изображение: %s", exc)
        return None


def _noise_level(grey, numpy) -> float:
    """Энергия высоких частот: следы сжатия и зерно съёмки дают её много."""
    if grey.shape[0] < 3 or grey.shape[1] < 3:
        return 0.0
    smoothed = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:]
    ) / 4.0
    return float(numpy.abs(grey[1:-1, 1:-1] - smoothed).mean())


def _skew_degrees(grey, numpy) -> float:
    """
    Наклон страницы. Оценивается по резкости профиля построчных сумм:
    у выровненного текста строки дают чёткие полосы, у наклонённого — размытые.
    """
    sample = grey
    if sample.shape[0] > 900:
        sample = sample[:: max(1, sample.shape[0] // 900)]
    if sample.shape[1] > 900:
        sample = sample[:, :: max(1, sample.shape[1] // 900)]

    binary = (sample < sample.mean()).astype(numpy.float32)
    if binary.sum() == 0:
        return 0.0

    best_angle, best_energy = 0.0, -1.0
    height, width = binary.shape

    for angle in (-6, -4, -2, -1, 0, 1, 2, 4, 6):
        shift = math.tan(math.radians(angle))
        columns = numpy.arange(width, dtype=numpy.float32)
        # Сдвиг строк пропорционально углу — дешёвая замена повороту.
        offsets = numpy.rint(columns * shift).astype(numpy.int32)
        profile = numpy.zeros(height, dtype=numpy.float32)
        for column in range(0, width, max(1, width // 120)):
            shifted = numpy.roll(binary[:, column], int(offsets[column]))
            profile += shifted
        energy = float(numpy.var(profile))
        if energy > best_energy:
            best_angle, best_energy = float(angle), energy

    return best_angle
