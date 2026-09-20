"""
OCR-фоллбэк на Tesseract.

Используется, когда MinerU недоступен или не вернул текстового слоя.
Путь к файлу всегда получается через StorageProvider, поэтому относительные
URI резолвятся относительно STORAGE_LOCAL_MOUNT, а не рабочего каталога.

Уверенность здесь настоящая, а не константа. Раньше любой распознанный блок
получал 0.7 — и выцветший скан, на котором распознались одни обрывки, и
чистая страница. На этом числе стоят и проверка читаемости на приёме, и
выбор уровня лестницы, так что константа означала, что оба решения
принимались вслепую. Tesseract отдаёт уверенность по каждому слову
(`image_to_data`), и блок наследует её среднее, взвешенное по длине слов.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from core import filetypes
from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.workspace import temp_dir
from core.providers.text_layer import TextLayer, TextLine, group_lines, merge_line_bboxes

logger = logging.getLogger(__name__)

# Расширения берутся из единого реестра форматов, а не дублируются здесь:
# второй набор успел разойтись с ним на .gif и .webp.
IMAGE_EXTENSIONS = {"." + ext for ext in filetypes.IMAGE_EXTENSIONS}

# Режим сегментации Tesseract: страница связного текста.
PSM_BLOCK = 6

# Уверенность, которая ставится, когда Tesseract не отдал разбивку по словам
# и измерить нечего. Это не оценка качества, а признак «не измерено».
UNMEASURED_CONFIDENCE = 0.5


class OcrUnavailable(RuntimeError):
    """
    OCR не удалось запустить: нет poppler, файл не забрался из хранилища,
    pdf2image упал. Это не «в документе нет текста», и путать их нельзя:
    на приёме второе означает карантин с формулировкой «документ нечитаем».
    """


@dataclass
class OcrLine:
    """Строка распознанного текста с координатами в долях изображения."""

    text: str
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    confidence: float = 0.0


class TesseractFallbackProvider:
    def __init__(self, lang: str = "rus+eng", storage: Optional[StorageProvider] = None):
        self.lang = lang
        self.storage = storage or StorageProviderFactory.default()
        # (сколько распознали, сколько всего) последнего полного прохода;
        # None — усечения не было.
        self.truncated_at: Optional[tuple] = None

    # ------------------------------------------------------------- публичное
    def parse_pages(
        self, uri: str, pages: List[int], file_type: Optional[str] = None,
        data: Optional[bytes] = None,
    ) -> List[ParsedBlock]:
        """
        Распознаёт только указанные страницы PDF (нумерация с 1).

        `file_type` — фактический тип файла, если он уже определён по
        содержимому. Имя файла приходит от отправителя и бывает чужим:
        картинка под именем `.pdf` уезжала в pdf2image, тот падал, и
        читаемость объявлялась непроверенной на файле, который читается.

        `data` — уже прочитанные байты. Приём держит их в руках к моменту
        проверки читаемости, и без них каждый растровый документ выкачивался
        из S3 второй раз.
        """
        if not pages:
            return []

        local_path, cleanup = self._materialize(uri, data)
        if not local_path:
            raise OcrUnavailable(f"Файл {uri} не удалось получить локально")

        try:
            if self._treat_as_image(local_path, file_type):
                logger.info("Постраничный OCR неприменим к изображению, обрабатываем целиком")
                return self._ocr_single_image(local_path, page=pages[0])

            blocks: List[ParsedBlock] = []
            first, last = min(pages), max(pages)
            try:
                images = convert_from_path(
                    local_path, first_page=first, last_page=last, dpi=settings.OCR_DPI
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("pdf2image не смог обработать страницы %s: %s", pages, exc)
                raise OcrUnavailable(str(exc)) from exc

            wanted = set(pages)
            for idx, img in enumerate(images):
                page_num = first + idx
                if page_num not in wanted:
                    continue
                blocks.extend(self._blocks_from_image(img, page_num))
            return blocks
        finally:
            cleanup()

    def parse_all_pages(self, uri: str) -> List[ParsedBlock]:
        """
        Распознаёт документ целиком (или всё изображение).

        Страницы рендерятся пачками и не больше, чем `OCR_FULL_MAX_PAGES`.
        Прежний код звал `convert_from_path` без границ: на техническом
        задании в несколько сотен листов это разворачивало в память весь
        документ разом. Усечение видно снаружи — `truncated_at` последнего
        разбора, его читает вызывающий и пишет в degraded.
        """
        self.truncated_at = None
        local_path, cleanup = self._materialize(uri)
        if not local_path:
            logger.error("Не удалось получить локальный путь для %s", uri)
            return []

        try:
            if self._is_image(local_path):
                return self._ocr_single_image(local_path, page=1)

            total = self._page_count(local_path)
            limit = max(1, settings.OCR_FULL_MAX_PAGES)
            last_page = min(total, limit) if total else limit
            if total and total > limit:
                logger.warning(
                    "OCR %s усечён: %d страниц из %d", uri, limit, total
                )
                self.truncated_at = (limit, total)

            blocks: List[ParsedBlock] = []
            batch = max(1, settings.OCR_BATCH_PAGES)
            for first in range(1, last_page + 1, batch):
                last = min(first + batch - 1, last_page)
                try:
                    images = convert_from_path(
                        local_path, first_page=first, last_page=last,
                        dpi=settings.OCR_DPI,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "pdf2image не смог обработать страницы %d-%d файла %s: %s",
                        first, last, local_path, exc,
                    )
                    break
                if not images:
                    break
                for index, img in enumerate(images):
                    blocks.extend(self._blocks_from_image(img, first + index))
                    img.close()
            return blocks
        finally:
            cleanup()

    @staticmethod
    def _page_count(local_path: str) -> int:
        """Сколько страниц в PDF. 0 — узнать не вышло, работаем по лимиту."""
        try:
            from pdf2image import pdfinfo_from_path

            return int(pdfinfo_from_path(local_path).get("Pages") or 0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Число страниц %s не определено: %s", local_path, exc)
            return 0

    # ------------------------------------------------------------ внутреннее
    def text_layer(self, uri: str) -> Optional[TextLayer]:
        """
        Построчное распознавание всего документа как текстовый слой.

        Нужно затем же, зачем слой векторного PDF: MinerU остаётся
        источником структуры, а текст берётся отсюда. Для растра это
        единственный способ прочитать русскую прозу, которую модель таблиц
        MinerU отдаёт побуквенной LaTeX-разметкой.
        """
        local_path, cleanup = self._materialize(uri)
        if not local_path:
            raise OcrUnavailable(f"Файл {uri} не удалось получить локально")

        try:
            images = self._images_of(local_path)
            pages: Dict[int, List[TextLine]] = {}
            for page_num, image in images:
                lines = [
                    TextLine(
                        text=line.text, bbox=list(line.bbox), page=page_num,
                        confidence=line.confidence,
                    )
                    for line in self.ocr_lines(image)
                    if line.text.strip()
                ]
                if lines:
                    pages[page_num] = lines
            if not pages:
                return None
            return TextLayer(pages=pages, page_count=max(pages))
        finally:
            cleanup()

    def _images_of(self, local_path: str):
        """
        Страницы документа по одной: (номер, картинка).

        Рендер идёт пачками и не дальше `OCR_FULL_MAX_PAGES`. Прежняя версия
        разворачивала весь PDF в список картинок разом — на многостраничном
        сканированном приложении это съедало память воркера целиком, ещё до
        первой строки распознавания.
        """
        if self._is_image(local_path):
            try:
                image = Image.open(local_path)
                image.load()
            except Exception as exc:  # noqa: BLE001
                raise OcrUnavailable(f"Изображение не открылось: {exc}") from exc
            return [(1, image)]
        return self._pdf_pages(local_path)

    def _pdf_pages(self, local_path: str):
        """Генератор страниц PDF: рендерим пачками, лишнего в памяти не держим."""
        total = self._page_count(local_path)
        limit = max(1, settings.OCR_FULL_MAX_PAGES)
        last_page = min(total, limit) if total else limit
        if total and total > limit:
            logger.warning(
                "Слой OCR %s усечён: %d страниц из %d", local_path, limit, total
            )
            self.truncated_at = (limit, total)
        batch = max(1, settings.OCR_BATCH_PAGES)
        for first in range(1, last_page + 1, batch):
            last = min(first + batch - 1, last_page)
            try:
                images = convert_from_path(
                    local_path, first_page=first, last_page=last, dpi=settings.OCR_DPI
                )
            except Exception as exc:  # noqa: BLE001
                if first == 1:
                    raise OcrUnavailable(str(exc)) from exc
                logger.error("Страницы %d-%d не отрендерились: %s", first, last, exc)
                return
            if not images:
                return
            for index, image in enumerate(images):
                yield first + index, image

    def _ocr_single_image(self, path: str, page: int) -> List[ParsedBlock]:
        try:
            with Image.open(path) as image:
                return self._blocks_from_image(image, page)
        except Exception as exc:  # noqa: BLE001
            logger.error("OCR изображения %s не удался: %s", path, exc)
            return []

    @staticmethod
    def _fit(image: "Image.Image") -> "Image.Image":
        """
        Уменьшить страницу до рабочего размера распознавания.

        Скан A3 приходит листом в 6600 пикселей по длинной стороне: OCR на
        нём идёт минутами, а строки читаются не лучше — их высота и так
        кратно выше порога распознавания. Координаты остаются в долях, так
        что масштабирование ничего не ломает ниже по конвейеру.

        Граница есть и снизу. Tesseract работает по высоте строки, и на
        мелком растре она ниже порога: лист 80x50 не давал ни одного слова,
        тот же лист крупнее читается уверенностью около 0.65. Пустой
        результат на приёме означает карантин «документ нечитаем», поэтому
        мелкое увеличивается до рабочего размера.
        """
        limit = max(512, settings.OCR_MAX_SIDE)
        floor = min(limit, max(0, settings.OCR_MIN_SIDE))
        side = max(image.width, image.height)
        if not side:
            return image
        if side > limit:
            scale = limit / float(side)
        elif side < floor:
            scale = floor / float(side)
        else:
            return image
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        try:
            return image.resize(size, Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001 — не вышло, читаем как есть
            logger.debug("Страница не приведена к рабочему размеру: %s", exc)
            return image

    def _blocks_from_image(self, image: "Image.Image", page: int) -> List[ParsedBlock]:
        """
        Блоки страницы по группам строк, а не один блок на весь лист.

        Раньше сюда уезжала вся страница одним блоком с рамкой во весь лист.
        Следствий три: нарезке нечего было делить, подсветка источника в RAG
        указывала на страницу целиком, а провал уверенности на отдельных
        строках растворялся в среднем по абзацам. Координаты и уверенность
        по строкам у Tesseract есть — их незачем было выбрасывать.
        """
        lines = self.ocr_lines(image)
        if lines:
            blocks: List[ParsedBlock] = []
            as_text_lines = [
                TextLine(text=l.text, bbox=list(l.bbox), page=page, confidence=l.confidence)
                for l in lines if l.text.strip()
            ]
            for group in group_lines(as_text_lines):
                text = "\n".join(l.text.strip() for l in group).strip()
                if not text:
                    continue
                blocks.append(ParsedBlock(
                    type="text",
                    text=text,
                    page=page,
                    bbox=merge_line_bboxes(group),
                    confidence=_weighted_lines(group),
                    is_fallback=True,
                ))
            if blocks:
                return blocks

        # Разбивки по словам нет — остаётся простое распознавание, и тогда
        # уверенность честно помечается как неизмеренная.
        text = self._ocr_image(image)
        if not text or not text.strip():
            return []
        return [ParsedBlock(
            type="text",
            text=text.strip(),
            page=page,
            bbox=[0.0, 0.0, 1.0, 1.0],
            confidence=UNMEASURED_CONFIDENCE,
            is_fallback=True,
        )]

    def ocr_lines(self, image: "Image.Image", psm: int = PSM_BLOCK) -> List[OcrLine]:
        """
        Строки с координатами и уверенностью распознавания. Координаты в
        долях изображения — как и везде в конвейере.
        """
        image = self._fit(image)
        try:
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=f"--psm {psm}",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка OCR (разбивка по словам): %s", exc)
            return []

        width, height = image.size
        if not width or not height:
            return []

        grouped: Dict[tuple, Dict[str, Any]] = {}
        for index, raw in enumerate(data.get("text") or []):
            word = (raw or "").strip()
            if not word:
                continue
            try:
                confidence = float(data["conf"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                confidence = -1.0
            if confidence < 0:
                continue
            # Урезанный или нестандартный вывод image_to_data не должен
            # ронять весь разбор: слово без геометрии просто пропускается.
            try:
                key = (
                    data["block_num"][index],
                    data["par_num"][index],
                    data["line_num"][index],
                )
                left, top = data["left"][index], data["top"][index]
                right = left + data["width"][index]
                bottom = top + data["height"][index]
            except (KeyError, IndexError, TypeError):
                continue
            entry = grouped.setdefault(
                key, {"words": [], "weight": 0.0, "sum": 0.0,
                      "box": [left, top, right, bottom]}
            )
            entry["words"].append(word)
            weight = max(1, len(word))
            entry["weight"] += weight
            entry["sum"] += (confidence / 100.0) * weight
            box = entry["box"]
            entry["box"] = [
                min(box[0], left), min(box[1], top),
                max(box[2], right), max(box[3], bottom),
            ]

        lines: List[OcrLine] = []
        for key in sorted(grouped):
            entry = grouped[key]
            left, top, right, bottom = entry["box"]
            lines.append(OcrLine(
                text=" ".join(entry["words"]),
                bbox=[
                    max(0.0, min(1.0, left / width)),
                    max(0.0, min(1.0, top / height)),
                    max(0.0, min(1.0, right / width)),
                    max(0.0, min(1.0, bottom / height)),
                ],
                confidence=round(entry["sum"] / entry["weight"], 3) if entry["weight"] else 0.0,
            ))
        return lines

    def _ocr_image(self, image: "Image.Image") -> str:
        try:
            return pytesseract.image_to_string(image, lang=self.lang, config="--psm 6")
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка OCR: %s", exc)
            return ""

    def _materialize(self, uri: str, data: Optional[bytes] = None):
        """
        Возвращает (локальный путь, функция очистки).
        Для S3 файл выкачивается во временный, для локального — резолвится
        через StorageProvider (с проверкой path traversal). Переданные байты
        избавляют от повторной выкачки.
        """
        noop = lambda: None  # noqa: E731

        if uri.startswith("s3://"):
            try:
                ext = os.path.splitext(uri.split("/")[-1])[1] or ".pdf"
                content = data if data is not None else self.storage.read_bytes(uri)
                # dir= задаётся явно: даже если TMPDIR кто-то перебил,
                # выкачанный из S3 исходник остаётся в томе сервиса.
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=ext, dir=str(temp_dir())
                )
                tmp.write(content)
                tmp.close()
                return tmp.name, lambda: self._safe_unlink(tmp.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Не удалось выкачать файл из S3 %s: %s", uri, exc)
                return None, noop

        try:
            path = self.storage.get_accessible_uri(uri)
        except Exception as exc:  # noqa: BLE001
            logger.error("Не удалось разрешить путь %s: %s", uri, exc)
            return None, noop

        if not os.path.exists(path):
            logger.error("Файл не найден: %s", path)
            return None, noop
        return path, noop

    @staticmethod
    def _safe_unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    @staticmethod
    def _is_image(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS

    @staticmethod
    def _treat_as_image(path: str, file_type: Optional[str] = None) -> bool:
        """Читать ли файл одной картинкой. Явный тип старше расширения."""
        if file_type:
            return filetypes.kind_of(file_type) == filetypes.KIND_IMAGE
        return TesseractFallbackProvider._is_image(path)


def _weighted_lines(lines: Sequence[Any]) -> float:
    """
    Средняя уверенность строк, взвешенная по длине текста. Годится и для
    `OcrLine`, и для `TextLine`: обеим нужны только `text` и `confidence`.
    """
    weights = [max(1, len(line.text)) for line in lines]
    total = sum(weights)
    if not total:
        return 0.0
    return round(
        sum(line.confidence * weight for line, weight in zip(lines, weights)) / total, 3
    )
