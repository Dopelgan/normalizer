"""
Клиент внешней мультимодальной модели Qwen3-VL (vLLM, OpenAI-совместимый /v1).

Почему модель внешняя. В бюджет видеокарты рабочего хоста укладывается либо
парсер документов, либо модель зрения — вместе они не живут. Поэтому разбор
чертежа вынесен на отдельный хост с GPU: конвейер ходит туда по HTTP и не
знает ни про веса, ни про CUDA. Если адрес не задан (`QWEN_ENDPOINT` пуст),
ветка чертежей честно ничего не возвращает — это видно в логе и в полноте
фрагмента, а не маскируется сплошным распознаванием листа.

Что здесь важно.

* **Ответ только JSON.** Модель просят вернуть объект заданной формы, а
  разбор терпит обёртку в ```json и болтовню вокруг: берётся первый
  сбалансированный объект. Непригодный ответ — это `None`, а не выдумка.
* **Температура 0 и потолок токенов.** Чертёж читается как документ, а не
  сочиняется; разброс ответов тут вреден.
* **Повтор только на транспорте.** Таймаут и 5xx повторяются с паузой,
  ответ «модель сказала ерунду» не повторяется: это не сетевой сбой.
* **Одна сессия на процесс.** Клиент создаётся на каждый документ, а пул
  соединений общий: иначе на каждый лист уходило бы новое рукопожатие, а
  сокеты копились бы — закрывать их было некому.
* **Картинка ужимается перед отправкой.** Лист в 300 dpi — это десятки
  мегабайт base64; длинная сторона режется до `QWEN_SHEET_MAX_SIDE`, а
  вырезанная область, наоборот, может быть увеличена — мелкий текст штампа
  модель читает заметно лучше на увеличенном фрагменте.
"""

from __future__ import annotations

import base64
import contextvars
import io
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from core import deadline
from core.config import settings
from core.providers import title_block

logger = logging.getLogger(__name__)

# Одна HTTP-сессия на процесс. Клиент чертежей создаётся на каждый документ,
# и своя сессия у каждого означала бы новое соединение на каждый лист плюс
# сокеты, которые некому закрыть: у клиента нет ни владельца, ни конца
# жизни. Пул соединений живёт столько же, сколько воркер.
_session_lock = threading.Lock()
_shared: Optional[requests.Session] = None


def shared_session() -> requests.Session:
    """Общая сессия процесса; создаётся при первом обращении."""
    global _shared
    with _session_lock:
        if _shared is None:
            _shared = requests.Session()
        return _shared


def close_shared_session() -> None:
    """Закрыть пул. Нужна при остановке воркера и в тестах."""
    global _shared
    with _session_lock:
        if _shared is not None:
            _shared.close()
            _shared = None

# Категории полей чертежа, которые модель имеет право назвать. Всё прочее
# сводится к правилам разбора: выдуманная категория в индекс не попадает.
CATEGORIES = (
    "size", "tolerance", "roughness", "thread", "radius",
    "material", "title_block", "callout", "position", "note",
)

_SYSTEM = (
    "Ты читаешь машиностроительные чертежи, оформленные по ЕСКД (ГОСТ). "
    "Ты извлекаешь то, что написано на листе, и никогда не досочиняешь "
    "недостающее. Если надпись не читается — ты её пропускаешь. "
    "Ответ — только JSON, без пояснений и без markdown."
)

_SHEET_PROMPT = """Прочитай чертёж и верни JSON:

{{"sheet_type": "detail|assembly|scheme|unknown",
  "annotations": [
    {{"category": "<одна из: {categories}>",
      "value": "надпись как на листе",
      "bbox": [x1, y1, x2, y2]}}
  ]}}

Правила:
- bbox — доли стороны листа от 0 до 1, порядок: лево, верх, право, низ.
- Знак диаметра пиши как ⌀, допуск как ±, шероховатость как Ra/Rz.
- Размер, посадку и допуск пиши одной надписью, как на листе: «⌀44H7», «20±0,1».
- position — номер позиции у выноски на сборочном чертеже.
- title_block — надписи основной надписи (штампа).
- Служебные графы бланка («Инв. № подл.», «Взам. инв. №», «Подп. и дата») пропускай.
- Не повторяй одну надпись дважды и не выдумывай надписей, которых нет.{hint}
"""

_TITLE_BLOCK_PROMPT = """На картинке — основная надпись (штамп) чертежа, вырезанная с листа.
Верни JSON:

{"fields": {"<название графы>": "<значение>"},
 "rows": [["<ячейка>", "..."]]}

Правила:
- fields — только заполненные графы: наименование, обозначение, материал,
  масштаб, масса, лист, листов, литера, организация, разработал, проверил.
- rows — та же таблица построчно, как она нарисована.
- Пустые графы пропускай, ничего не додумывай.
"""

# Отдельный проход по штампу (DRAWING_TITLE_BLOCK_PASS). Отличия от
# `_TITLE_BLOCK_PROMPT` — ровно те, на которых терялись данные: перечислены
# все графы ГОСТ 2.104, включая подписи, даты и боковые, рукописное
# помечается отдельно, а обозначение требуется со всеми точками. Старый
# промпт прямо велел пропускать служебные графы бланка — именно там лежат
# «Инв. № подл.» и «Перв. примен.».
_TITLE_BLOCK_FIELDS_PROMPT = """На картинке — основная надпись (штамп) чертежа по ГОСТ 2.104
с прилегающими графами, вырезанная с листа. Верни JSON:

{"fields": {"<название графы>": "<значение как на листе>"},
 "handwritten": ["<название графы, заполненной от руки>"],
 "rows": [["<ячейка>", "..."]]}

Правила:
- fields — все заполненные графы. Названия граф пиши ровно так:
  «Обозначение», «Наименование», «Материал», «Масса», «Масштаб», «Лист»,
  «Листов», «Литера», «Организация», «Разраб.», «Пров.», «Т.контр.»,
  «Н.контр.», «Утв.», «Формат», «Изм.», «Инв. № подл.», «Инв. № дубл.»,
  «Взам. инв. №», «Перв. примен.», «Справ. №», «Подп. и дата».
- Графы подписантов («Разраб.», «Пров.», «Т.контр.», «Н.контр.», «Утв.») —
  это фамилия в графе, без даты. Даты собери в «Подп. и дата» через «; ».
- «Изм.» — строки графы изменений целиком: «2 Нов. ТАДУ.409-23».
- Боковые графы читай обязательно: «Инв. № подл.» («инв. № 1085-02» ->
  «1085-02»), «Взам. инв. №», «Инв. № дубл.», «Перв. примен.», «Справ. №».
  Они бывают повёрнуты на 90°. Пропускать их нельзя.
- handwritten — названия тех граф, значение которых дописано от руки
  (даты, инвентарные номера, номера извещений). Рукописную закорючку без
  читаемого текста не возвращай, но рукописные цифры и буквы возвращай.
- Обозначение переписывай со всеми точками и пробелами:
  «ТАДУ 421451.001 СБ», а не «ТАДУ 421451001 СБ»; код документа пиши
  кириллицей: «СБ», «ТЗ», «ГЧ», а не «C6», «73».
- rows — та же таблица построчно, как она нарисована.
- Пустые графы пропускай, ничего не додумывай.
"""

_SIDE_COLUMN_PROMPT = """На картинке — колонка дополнительных граф у левого поля чертежа
(ГОСТ 2.104): «Перв. примен.», «Справ. №», «Инв. № подл.», «Подп. и дата»,
«Взам. инв. №», «Инв. № дубл.». Надписи в ней повёрнуты. Верни JSON:

{"fields": {"<название графы>": "<значение как на листе>"},
 "handwritten": ["<название графы, заполненной от руки>"]}

Правила:
- Возвращай только заполненные графы и только их значения: «Инв. № подл.»
  со значением «1085-02», а не с самим названием графы.
- Инвентарные номера и даты здесь почти всегда дописаны от руки — такие
  графы перечисли в handwritten.
- Обозначение в «Перв. примен.» переписывай со всеми точками.
- Ничего не додумывай; пустую графу пропускай.
"""

_TABLE_PROMPT = """На картинке — таблица, вырезанная с листа чертежа
(спецификация, таблица составных частей или ведомость). Верни JSON:

{"title": "<заголовок таблицы или пустая строка>",
 "headers": ["<шапка по колонкам>"],
 "rows": [["<ячейка>", "..."]]}

Правила:
- Число колонок в каждой строке равно числу колонок шапки.
- Пустая ячейка — пустая строка.
- Читай только то, что написано; порядок строк сохраняй.
"""


_TRANSCRIBE_PROMPT = """Перепиши весь текст, который есть на листе. Верни JSON:

{"lines": ["<строка текста как на листе>", "..."]}

Правила:
- Читай всё: основную надпись (штамп) со всеми графами и фамилиями,
  обозначение и наименование, лист и листов, масштаб, массу, литеру,
  строки изменений (изм., лист, № докум., дата), боковые графы
  («Инв. № подл.», «Подп. и дата», «Перв. примен.» и их значения),
  обозначение в верхнем углу (оно бывает перевёрнуто на 180°),
  технические требования и характеристики, примечания, заголовки видов,
  разрезов и рисунков, надписи у выносок, размеры, таблицы построчно,
  надписи на изображённых табличках и бирках, «Копировал», «Формат».
- Одна строка листа — один элемент lines, в порядке чтения: сверху вниз,
  слева направо. Строку таблицы пиши одной строкой, ячейки через « | ».
- Пиши как на листе: буквы, цифры, знаки ⌀ ± ° и индексы допусков,
  знак допуска (+ или −) ставь тот, что на листе.
- Обозначения переписывай со всеми точками и пробелами: «ТАДУ 421451.001 СБ»,
  а не «ТАДУ 421451001 СБ».
- Латиницу пиши латиницей, кириллицу кириллицей: коды KKS («30KUA70AX001»),
  марки и английские надписи не переводи в кириллицу.
- Рукописную подпись без читаемого текста пропускай. Ничего не додумывай.
"""


# Знак отклонения и справочная звёздочка (`DRAWING_DIMENSION_FIDELITY`).
# В транскрипции листа «30±10*» приходило как «30+10*», а на вырезке вида
# знак был верный: модель подменяет «±» на «+» сама, и правится это запросом.
_SIGN_RULE = """- Знак отклонения переписывай ровно тот, что стоит на листе: «±» — это «±»,
  а не «+»; «−» — это «−», а не «+». Симметричное отклонение пиши одним
  знаком «±»: «30±10», «⌀4,5±1».
- Звёздочку справочного размера не теряй: «30±10*», «(Y+12)*».
"""

# Мелкий шрифт штампа и выносок (`DRAWING_DIMENSION_FIDELITY`). На проде
# «Ербасов» прочитано как «Ервасов», «Бажко» — как «Бахко», «Н1-ИНп» — как
# «H1-ИНн»: ошибка в одной букве, которую не выправит ни словарь (фамилии
# в нём нет), ни правило (прочтений у буквы два). Поэтому вырезка штампа
# увеличивается, а в запрос добавляется правило про эти буквы.
_CAREFUL_LETTERS_RULE = """- Фамилии в графах штампа читай по буквам и не подбирай похожее слово.
  В шрифте по ГОСТ 2.304 путаются «б» и «в», «ж» и «х», «н» и «п»,
  «и» и «н»: «Ербасов», а не «Ервасов»; «Бажко», а не «Бахко».
- Обозначение сварного шва пиши кириллицей и целиком: «Н1-ИНп», «У4-ИНп»,
  «Т1-ИНп» — не «H1-ИНн» и не «T1-ИНн».
"""

_SMALL_PRINT_RULE = """- Запятую в десятичной дроби не теряй: «M20x1,5», «Rz 3,2», «⌀8,5» — а не
  «M20x15», «Rz 32», «⌀85». Предельные отклонения, написанные мелко справа
  вверху и внизу от размера, пиши полностью, со знаком и запятой: «12+0,8»,
  «40−0,25», «⌀16(+0,018/−0,007)».
"""

@dataclass
class SheetReading:
    """
    Прочитанный лист: надписи и род чертежа. `cut` — ответ модели оборвался
    на потолке токенов, и список надписей неполон. `tiles` — сколько частей
    листа прочитано сверх самого листа (DRAWING_SHEET_TILES), `failed` —
    части листа (доли стороны), надписи которых прочитать не удалось.
    """

    sheet_type: str = "unknown"
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    cut: bool = False
    tiles: int = 0
    failed: List[List[float]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """
        Список надписей полон: либо ответ не обрывался, либо оборванный лист
        дочитан частями и все части прочитаны. `cut` при этом остаётся
        правдой — она говорит, что ответ по листу целиком не поместился,
        и это видно в логе и в payload.
        """
        if self.failed:
            return False
        return not self.cut or bool(self.tiles)


@dataclass
class Transcript:
    """
    Весь текст листа. `failed` — части листа (доли стороны: лево, верх,
    право, низ) и причина, по которой их текст не прочитан целиком.
    """

    lines: List[str]
    tiles: int = 0
    failed: List[Tuple[List[float], str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.failed


# Причины, по которым часть листа не прочитана целиком.
TILE_CUT = "transcript_cut"            # ответ оборвался на потолке токенов
TILE_NOT_READ = "transcript_not_read"  # модель не ответила
TILE_NO_TIME = "time_budget"           # до срока задачи вызов не уложится


class QwenVisionClient:
    """HTTP-клиент к Qwen3-VL. Все методы возвращают `None` при неудаче."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        session: Optional[requests.Session] = None,
    ):
        self.endpoint = (endpoint if endpoint is not None else settings.QWEN_ENDPOINT) or ""
        self.endpoint = self.endpoint.rstrip("/")
        self.model = model or settings.QWEN_MODEL
        self.api_key = api_key if api_key is not None else settings.QWEN_API_KEY
        self.timeout = timeout or settings.QWEN_TIMEOUT
        self.retries = max(0, int(settings.QWEN_RETRIES))
        # Своя сессия — только если её передали снаружи (тесты, стенд).
        self.session = session or shared_session()

    # --------------------------------------------------------------- наружу
    @property
    def available(self) -> bool:
        """Адрес задан. Доступность самого хоста проверяется вызовом."""
        return bool(self.endpoint)

    def read_sheet(
        self, image, hint: str = "", until: Optional[float] = None,
    ) -> Optional[SheetReading]:
        """
        Лист целиком: надписи с их местами и род чертежа.

        Плотный лист не помещается в потолок ответа: JSON обрывается посреди
        списка надписей. С DRAWING_SHEET_TILES такой лист читается заново
        частями с перекрытием, а списки надписей сливаются без повторов.
        `until` — момент time.monotonic(), после которого части листа уже не
        читаются (бюджет листа DRAWING_TIME_BUDGET).
        """
        prompt = self._sheet_prompt(hint)
        if settings.DRAWING_SMALL_PRINT_CARE:
            image = enlarge_small(image, settings.DRAWING_SMALL_IMAGE_SIDE)
        whole = self._read_sheet_piece(image, prompt)
        if whole is None or not whole.cut or not settings.DRAWING_SHEET_TILES:
            return whole
        return self._read_sheet_tiles(image, prompt, whole, until)

    def _sheet_prompt(self, hint: str) -> str:
        """Запрос прохода «лист целиком». Один и тот же для листа и частей."""
        prompt = _SHEET_PROMPT.format(
            categories=", ".join(CATEGORIES),
            hint=f"\n- Подсказка по разметке листа: {hint}" if hint else "",
        )
        if settings.DRAWING_DIMENSION_FIDELITY:
            prompt += _SIGN_RULE + _CAREFUL_LETTERS_RULE
        if settings.DRAWING_SMALL_PRINT_CARE:
            prompt += _SMALL_PRINT_RULE
        return prompt

    @staticmethod
    def _sheet_max_tokens() -> Optional[int]:
        """
        Потолок ответа прохода «лист целиком». Без флага — общий
        QWEN_MAX_TOKENS, как и было.
        """
        if settings.DRAWING_SHEET_HIGH_TOKENS:
            return settings.DRAWING_SHEET_MAX_TOKENS
        return None

    def _read_sheet_piece(self, image, prompt: str) -> Optional[SheetReading]:
        """
        Лист или его часть одним вызовом. `cut` — ответ оборвался: модель
        упёрлась в потолок токенов или JSON не разобрался и надписи пришлось
        спасать из оборванного текста. `None` — ответа нет вовсе.
        """
        if not self.available:
            logger.error("QWEN_ENDPOINT не задан — разбор чертежа невозможен")
            return None
        answer = self._ask_text(image, prompt, settings.QWEN_SHEET_MAX_SIDE,
                                self._sheet_max_tokens())
        if answer is None:
            return None
        content, finish = answer
        data = extract_json(content)
        cut = finish == "length"
        if data is None or not isinstance(data.get("annotations"), list):
            # Ответ оборван потолком токенов: целого объекта нет, и разбор
            # находил первый вложенный — одну надпись вместо всего листа.
            items = salvage_list(content, "annotations")
            if items:
                logger.warning("Ответ модели оборван: из списка %s спасено %d элементов",
                               "annotations", len(items))
                cut = True
                salvaged: Dict[str, Any] = {"annotations": items}
                kind = re.search(r'"sheet_type"\s*:\s*"([^"]*)"', content)
                if kind:
                    salvaged["sheet_type"] = kind.group(1)
                data = salvaged
        if data is None:
            logger.error("Модель вернула не-JSON (%d символов): %.200s", len(content), content)
            return None
        annotations = data.get("annotations")
        if not isinstance(annotations, list):
            logger.warning("Модель вернула JSON без списка annotations")
            annotations = []
        return SheetReading(
            sheet_type=str(data.get("sheet_type") or "unknown"),
            annotations=[_with_fixed_grade(a) for a in annotations if isinstance(a, dict)],
            raw=data,
            cut=cut,
        )

    def _read_sheet_tiles(
        self, image, prompt: str, whole: SheetReading, until: Optional[float],
    ) -> SheetReading:
        """
        Лист, ответ по которому оборвался, частями с перекрытием. Сетка та
        же, что у прохода «весь текст листа» (`tile_boxes`): часть листа
        меньше, надписей в ней меньше, и ответ по ней в потолок помещается.

        Деления вглубь нет: часть, которая тоже оборвалась, и часть, на
        которую не ответили или не осталось времени, записываются в `failed`
        — лист честно недочитан. Так цена ограничена четырьмя вызовами, а не
        растёт вчетверо на каждом уровне.
        """
        boxes = tile_boxes([0.0, 0.0, 1.0, 1.0], settings.DRAWING_SHEET_TILE_GRID,
                           settings.DRAWING_SHEET_TILE_OVERLAP)
        logger.warning(
            "Надписи листа не поместились в ответ (%d до обрыва): лист "
            "дочитывается %d частями", len(whole.annotations), len(boxes),
        )
        limit = threading.Semaphore(max(1, settings.DRAWING_TRANSCRIBE_TILE_PARALLEL))

        def read(part):
            # Срок проверяется после ожидания очереди: часть, дождавшаяся
            # места, не должна начинать вызов, на который времени уже нет.
            with limit:
                left = deadline.remaining()
                if left is not None and left < settings.VLM_TIMEOUT:
                    return None
                if until is not None and time.monotonic() > until:
                    return None
                return self._read_sheet_piece(crop_box(image, part), prompt)

        # Срок задачи хранится в contextvars: потоку нужна копия контекста.
        with ThreadPoolExecutor(max_workers=len(boxes)) as pool:
            futures = [pool.submit(contextvars.copy_context().run, read, part) for part in boxes]
            answers = [future.result() for future in futures]

        parts: List[List[Dict[str, Any]]] = []
        failed: List[List[float]] = []
        read_count = 0
        for box, part in zip(boxes, answers):
            if part is None:
                failed.append([round(value, 4) for value in box])
                continue
            read_count += 1
            parts.append([place_in_sheet(a, box) for a in part.annotations])
            if part.cut:
                failed.append([round(value, 4) for value in box])
        # Части идут первыми — они покрывают весь лист; надписи оборванного
        # ответа по листу добавляются, если такой надписи нет ни в одной
        # части (надпись на стыке, которую обе части прочитали кусками).
        merged = merge_annotations(parts + [whole.annotations])
        logger.info(
            "Надписи листа дочитаны частями: частей %d, не прочитано %d, надписей %d",
            read_count, len(failed), len(merged),
        )
        return SheetReading(
            sheet_type=whole.sheet_type,
            annotations=merged,
            raw=whole.raw,
            cut=whole.cut,
            tiles=read_count,
            failed=failed,
        )

    def read_title_block(self, image) -> Optional[Dict[str, Any]]:
        """Вырезанный штамп -> графы и та же таблица построчно."""
        prompt = _TITLE_BLOCK_PROMPT
        if settings.DRAWING_DIMENSION_FIDELITY:
            # Фамилии в штампе — мелкий рукописный шрифт: вырезка
            # увеличивается, как и вырезка вида с мелкими отклонениями.
            prompt += _CAREFUL_LETTERS_RULE
            image = enlarge_small(image, settings.DRAWING_SMALL_IMAGE_SIDE)
        data = self._ask(image, prompt, settings.QWEN_REGION_MAX_SIDE)
        if data is None:
            return None
        fields = data.get("fields")
        named = (
            {str(key): str(value) for key, value in fields.items() if value not in (None, "")}
            if isinstance(fields, dict) else {}
        )
        return {"fields": named, "rows": _string_rows(data.get("rows"))}

    def read_title_block_fields(self, image) -> Optional[Dict[str, Any]]:
        """
        Вырезка штампа в высоком разрешении -> графы с подписями,
        рукописные отдельно, та же таблица построчно
        (DRAWING_TITLE_BLOCK_PASS). Разбор — `title_block.parse`.

        Своя длинная сторона (`QWEN_TITLE_BLOCK_MAX_SIDE`) — не настройка
        «на всякий случай»: вырезка штампа A4 в 200 dpi после увеличения
        даёт около 2900 px, а `QWEN_REGION_MAX_SIDE` = 1600 ужимал её
        обратно, и увеличение не доезжало до модели вовсе. На ужатой
        вырезке и путаются фамилии и код документа. Тот же потолок обязан
        стоять и в `drawing_vision.crop`, иначе вырезку ужмут до отправки.
        """
        data = self._ask(image, _careful(_TITLE_BLOCK_FIELDS_PROMPT),
                         settings.QWEN_TITLE_BLOCK_MAX_SIDE)
        if data is None:
            return None
        return title_block.parse(data)

    def read_side_column(self, image) -> Optional[Dict[str, Any]]:
        """Колонка дополнительных граф у левого поля -> графы и рукописные."""
        data = self._ask(image, _careful(_SIDE_COLUMN_PROMPT),
                         settings.QWEN_TITLE_BLOCK_MAX_SIDE)
        if data is None:
            return None
        return title_block.parse(data)

    def read_table(self, image) -> Optional[Dict[str, Any]]:
        """Вырезанная таблица -> заголовок, шапка и строки."""
        data = self._ask(image, _TABLE_PROMPT, settings.QWEN_REGION_MAX_SIDE)
        if data is None:
            return None
        headers = data.get("headers")
        return {
            "title": str(data.get("title") or ""),
            "headers": [str(h) for h in headers] if isinstance(headers, list) else [],
            "rows": _string_rows(data.get("rows")),
        }

    def transcribe(self, image) -> Optional[List[str]]:
        """Весь текст листа строками: штамп, графы, примечания, таблицы."""
        data = self._ask(
            image, _transcribe_prompt(), settings.DRAWING_TRANSCRIBE_MAX_SIDE,
            max_tokens=settings.DRAWING_TRANSCRIBE_MAX_TOKENS,
        )
        if data is None:
            return None
        lines = data.get("lines")
        if not isinstance(lines, list):
            logger.warning("Модель вернула JSON без списка lines")
            return None
        return _clean_lines(lines)

    def transcribe_sheet(self, image, until: Optional[float] = None) -> Optional[Transcript]:
        """
        Весь текст листа с дочитыванием частями (DRAWING_TRANSCRIBE_TILES).

        Плотная схема не помещается в потолок ответа: JSON обрывается. Строки,
        которые модель успела вернуть, не выбрасываются, а лист читается
        заново частями с перекрытием; оборванная часть делится ещё раз.
        `None` — модель не ответила на лист вовсе. `until` — момент
        time.monotonic(), после которого части листа уже не читаются
        (бюджет листа DRAWING_TIME_BUDGET).
        """
        lines, cut = self._transcribe_piece(image)
        if lines is None:
            return None
        if not cut:
            return Transcript(lines=lines)
        logger.warning(
            "Текст листа не поместился в ответ (%d строк до обрыва): лист "
            "дочитывается частями", len(lines),
        )
        limit = threading.Semaphore(max(1, settings.DRAWING_TRANSCRIBE_TILE_PARALLEL))
        failed: List[Tuple[List[float], str]] = []
        parts, counted = self._read_tiles(image, [0.0, 0.0, 1.0, 1.0], 1, limit, failed, until)
        # Части идут первыми — они покрывают весь лист; строки оборванного
        # ответа добавляются, если их нет ни в одной части (строка на стыке,
        # которую обе части прочитали лишь кусками).
        merged = merge_lines(parts + [lines])
        logger.info(
            "Лист дочитан частями: частей %d, не прочитано %d, строк %d",
            counted, len(failed), len(merged),
        )
        return Transcript(lines=merged, tiles=counted, failed=failed)

    def _read_tiles(self, image, box, depth, limit, failed, until=None):
        """
        Часть листа `box` сеткой GRID x GRID. Возвращает списки строк частей
        в порядке чтения и число прочитанных частей. Оборванная часть
        делится дальше, пока не исчерпана глубина; на последнем уровне её
        строки сохраняются, а сама она записывается в `failed`.
        """
        boxes = tile_boxes(box, settings.DRAWING_TRANSCRIBE_TILE_GRID,
                           settings.DRAWING_TRANSCRIBE_TILE_OVERLAP)

        def read(part):
            # Срок проверяется после ожидания очереди: часть, дождавшаяся
            # места, не должна начинать вызов, на который времени уже нет.
            with limit:
                left = deadline.remaining()
                if left is not None and left < settings.VLM_TIMEOUT:
                    return None, TILE_NO_TIME
                if until is not None and time.monotonic() > until:
                    return None, TILE_NO_TIME
                lines, cut = self._transcribe_piece(crop_box(image, part))
            if lines is None:
                return None, TILE_NOT_READ
            return lines, (TILE_CUT if cut else None)

        # Срок задачи хранится в contextvars: потоку нужна копия контекста.
        with ThreadPoolExecutor(max_workers=len(boxes)) as pool:
            futures = [pool.submit(contextvars.copy_context().run, read, part) for part in boxes]
            answers = [future.result() for future in futures]

        result: List[List[str]] = []
        counted = 0
        for part, (lines, problem) in zip(boxes, answers):
            counted += 1
            if problem == TILE_CUT and depth < settings.DRAWING_TRANSCRIBE_TILE_DEPTH:
                inner, inner_count = self._read_tiles(image, part, depth + 1, limit, failed, until)
                result.extend(inner)
                result.append(lines)
                counted += inner_count
                continue
            if problem is not None:
                failed.append(([round(value, 4) for value in part], problem))
            if lines:
                result.append(lines)
        return result, counted

    def _transcribe_piece(self, image) -> Tuple[Optional[List[str]], bool]:
        """
        Строки листа или его части и признак обрыва. Обрыв — ответ упёрся в
        потолок токенов или не разобрался как JSON; тогда возвращаются
        строки, которые модель успела закончить. `(None, False)` — ответа
        нет или он целый, но не того вида (без списка lines): делением листа
        такой ответ не исправить.
        """
        answer = self._ask_text(
            image, _transcribe_prompt(), settings.DRAWING_TRANSCRIBE_MAX_SIDE,
            max_tokens=settings.DRAWING_TRANSCRIBE_MAX_TOKENS,
        )
        if answer is None:
            return None, False
        content, finish = answer
        data = extract_json(content)
        if data is not None and finish != "length":
            if not isinstance(data.get("lines"), list):
                logger.warning("Модель вернула JSON без списка lines")
                return None, False
            lines, cut = _clean_lines(data["lines"]), False
        else:
            lines, cut = salvage_lines(content), True
        # Целый ответ с длинным рядом номеров бывает и настоящим (ряд клемм
        # «X1:1» … «X1:48»): счётом он считается, только если ряд невероятно
        # длинный для листа; у оборванного ответа порог обычный.
        start = counting_run(lines, _COUNT_RUN if cut else _COUNT_RUN_WHOLE)
        if start is not None:
            # Модель сбилась на счёт («QF1», «QF2», … «QF1000»): хвост счёта —
            # выдумка, а не текст листа. Ответ считается оборванным на нём,
            # и часть листа дочитывается мельче.
            logger.warning("Модель сбилась на счёт с «%s»: %d строк отброшено",
                           lines[start], len(lines) - start - _COUNT_RUN)
            return lines[:start + _COUNT_RUN], True
        return lines, cut

    def health(self) -> bool:
        """Отвечает ли хост модели. Нужна для приёмки и для понятного лога."""
        if not self.available:
            return False
        try:
            response = self.session.get(f"{self._base()}/models", timeout=min(30, self.timeout))
            return response.status_code < 500
        except requests.RequestException as exc:
            logger.error("Qwen недоступен: %s", exc)
            return False

    # ----------------------------------------------------------- внутреннее
    def _ask(
        self, image, prompt: str, max_side: int, max_tokens: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Вызов, от которого ждут целый JSON-объект. Спасение оборванного
        списка живёт в `_read_sheet_piece`: оно нужно только проходу «лист
        целиком», и только там обрыв — это состояние, а не просто неудача.
        """
        if not self.available:
            logger.error("QWEN_ENDPOINT не задан — разбор чертежа невозможен")
            return None
        answer = self._ask_text(image, prompt, max_side, max_tokens)
        if answer is None:
            return None
        content = answer[0]
        data = extract_json(content)
        if data is None:
            logger.error("Модель вернула не-JSON (%d символов): %.200s", len(content), content)
        return data

    def _ask_text(
        self, image, prompt: str, max_side: int, max_tokens: Optional[int] = None,
    ) -> Optional[Tuple[str, Optional[str]]]:
        """Текст ответа и finish_reason («length» — упёрся в потолок токенов)."""
        if not self.available:
            logger.error("QWEN_ENDPOINT не задан — разбор чертежа невозможен")
            return None
        encoded = encode_image(image, max_side)
        if not encoded:
            return None

        body = {
            "model": self.model,
            "temperature": settings.QWEN_TEMPERATURE,
            "max_tokens": max_tokens or settings.QWEN_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": encoded}},
                    {"type": "text", "text": prompt},
                ]},
            ],
        }
        payload = self._post(body)
        if payload is None:
            return None
        content = _content_of(payload)
        if content is None:
            return None
        return content, _finish_of(payload)

    def _post(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ответ модели текстом. Повторяется только то, что имеет смысл
        повторять: обрыв связи, таймаут и 5xx (vLLM отвечает так, пока
        грузит веса). Отказ 4xx — это неверный запрос или чужая модель,
        и повтор его не исправит.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self._base()}/chat/completions"
        for attempt in range(self.retries + 1):
            last = attempt >= self.retries
            try:
                response = self.session.post(
                    url, json=body, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                self._log_attempt(exc, attempt, last)
                if last:
                    return None
                time.sleep(min(5.0, 1.0 * (attempt + 1)))
                continue

            if response.status_code >= 500:
                self._log_attempt(
                    f"{response.status_code} {response.text[:200]}", attempt, last
                )
                if last:
                    return None
                time.sleep(min(5.0, 1.0 * (attempt + 1)))
                continue

            if response.status_code >= 400:
                logger.error(
                    "Qwen отклонил запрос: %d %s", response.status_code, response.text[:200]
                )
                return None

            try:
                payload = response.json()
            except ValueError as exc:
                logger.error("Qwen вернул не-JSON на уровне протокола: %s", exc)
                return None
            if not isinstance(payload, dict):
                logger.error("Qwen вернул ответ не того вида: %.200s", payload)
                return None
            return payload
        return None

    def _log_attempt(self, reason: Any, attempt: int, last: bool) -> None:
        logger.log(
            logging.ERROR if last else logging.WARNING,
            "Запрос к Qwen не удался (попытка %d из %d): %s",
            attempt + 1, self.retries + 1, reason,
        )

    def _base(self) -> str:
        """Адрес до /v1: в настройке допустимы обе записи."""
        if self.endpoint.endswith("/v1"):
            return self.endpoint
        return f"{self.endpoint}/v1"


# ===========================================================================
# Помощники
# ===========================================================================

# Марка стали или сварочной проволоки с молибденом на конце: «…Н11М3». Чертёжную
# тройку после «М» модель через раз пишет буквой «З» («Св-04Х19Н11МЗ») — по
# такой записи марку не найти.
_GRADE_M3 = re.compile(r"(\d{2}[ХXхx]\d{1,2}[НHнh]\d{1,2}[А-ЯA-Z0-9]*[МM])З(?![А-Яа-яA-Za-z0-9])")


def _with_fixed_grade(annotation: Dict[str, Any]) -> Dict[str, Any]:
    value = annotation.get("value")
    if isinstance(value, str) and settings.DRAWING_SMALL_PRINT_CARE:
        return {**annotation, "value": fix_grade(value)}
    return annotation


def fix_grade(text: str) -> str:
    """«04Х19Н11МЗ» -> «04Х19Н11М3» (DRAWING_SMALL_PRINT_CARE); прочий текст не меняется."""
    return _GRADE_M3.sub(r"\g<1>3", text) if settings.DRAWING_SMALL_PRINT_CARE else text


def _careful(prompt: str) -> str:
    """
    Запрос по штампу с правилом про путающиеся буквы
    (`DRAWING_DIMENSION_FIDELITY`). Нужен и отдельному проходу по штампу
    (`DRAWING_TITLE_BLOCK_PASS`), и колонке у левого поля: фамилии и
    обозначение шва путаются в обоих запросах одинаково.
    """
    if not settings.DRAWING_DIMENSION_FIDELITY:
        return prompt
    return prompt + _CAREFUL_LETTERS_RULE


def _transcribe_prompt() -> str:
    """Запрос «перепиши весь текст листа»; с DRAWING_SMALL_PRINT_CARE — с правилом про дроби."""
    return (
        _TRANSCRIBE_PROMPT
        + (_SIGN_RULE + _CAREFUL_LETTERS_RULE
           if settings.DRAWING_DIMENSION_FIDELITY else "")
        + (_SMALL_PRINT_RULE if settings.DRAWING_SMALL_PRINT_CARE else "")
    )


def enlarge_small(image, small_side: int):
    """Небольшая картинка увеличивается вдвое: мелкие индексы и запятые читаются увереннее."""
    try:
        if not small_side or max(image.size) >= small_side:
            return image
        from PIL import Image
        return image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    except Exception as exc:  # noqa: BLE001 — битая картинка не должна ронять разбор
        logger.error("Не удалось увеличить картинку для модели: %s", exc)
        return image


def encode_image(image, max_side: int) -> str:
    """PIL-картинка -> data-URL с PNG. Пустая строка, если не вышло."""
    try:
        prepared = image.convert("RGB")
        if max_side and max(prepared.size) > max_side:
            prepared = prepared.copy()
            prepared.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        prepared.save(buffer, format="PNG", optimize=True)
    except Exception as exc:  # noqa: BLE001 — битая картинка не должна ронять разбор
        logger.error("Не удалось подготовить картинку для модели: %s", exc)
        return ""
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Первый сбалансированный объект из ответа. Модель любит обернуть JSON в
    ```json и добавить фразу до или после — на разбор это влиять не должно.
    """
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except ValueError:
        pass

    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start:index + 1])
                    except ValueError:
                        break
                    return data if isinstance(data, dict) else None
        start = text.find("{", start + 1)
    return None


def salvage_list(content: str, key: str) -> List[Any]:
    """
    Целые элементы списка `key` из оборванного ответа. Модель упёрлась в
    потолок токенов посреди списка: всё, что она успела дописать целиком,
    прочитано верно и пропадать не должно.
    """
    marker = content.find(f'"{key}"')
    if marker == -1:
        return []
    index = content.find("[", marker)
    if index == -1:
        return []
    decoder = json.JSONDecoder()
    items: List[Any] = []
    index += 1
    while index < len(content):
        while index < len(content) and (content[index].isspace() or content[index] == ","):
            index += 1
        if index >= len(content) or content[index] == "]":
            break
        try:
            item, index = decoder.raw_decode(content, index)
        except ValueError:
            break
        items.append(item)
    return items


def _content_of(payload: Dict[str, Any]) -> Optional[str]:
    choices = payload.get("choices") or []
    if not choices:
        logger.error("Ответ модели без choices: %.200s", payload)
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):      # некоторые сборки vLLM отдают частями
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not content:
        logger.error("Ответ модели пуст")
        return None
    return str(content)


def _finish_of(payload: Dict[str, Any]) -> Optional[str]:
    choices = payload.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    reason = first.get("finish_reason")
    return str(reason) if reason else None


def _clean_lines(lines: Sequence[Any]) -> List[str]:
    return [fix_grade(str(line).strip()) for line in lines
            if isinstance(line, (str, int, float)) and str(line).strip()]


def salvage_lines(content: str) -> List[str]:
    """
    Законченные элементы `lines` из оборванного ответа
    `{"lines": ["…", "…", "незаконч`. Строка, на которой ответ оборвался,
    отбрасывается: её дочитает часть листа.
    """
    return _collapse_runs(_clean_lines(salvage_list(content or "", "lines")))


# Подряд столько строк «префикс + число» с шагом 1 — модель сбилась на
# счёт. На схеме ряд клемм бывает длинным, но в одной части листа столько
# подписей подряд не стоит; настоящие строки дочитает часть меньше.
_COUNT_RUN = 30
_COUNT_RUN_WHOLE = 200
_NUMBERED = re.compile(r"^(.*?)(\d+)$")


def counting_run(lines: Sequence[str], limit: int = _COUNT_RUN) -> Optional[int]:
    """Начало счёта длиной больше `limit` строк или None."""
    start, length, previous = 0, 0, None
    for index, line in enumerate(lines):
        match = _NUMBERED.match(line)
        current = (match.group(1), int(match.group(2))) if match else None
        if (current and previous and current[0] == previous[0]
                and current[1] == previous[1] + 1):
            length += 1
        else:
            start, length = index, 1 if current else 0
        if length > limit:
            return start
        previous = current
    return None


# Ответ часто упирается в потолок оттого, что модель зациклилась на одной
# строке: подряд больше стольких одинаковых строк — повтор, а не текст листа.
_MAX_RUN = 3


def _collapse_runs(lines: List[str]) -> List[str]:
    result: List[str] = []
    run = 0
    for line in lines:
        run = run + 1 if result and result[-1] == line else 1
        if run <= _MAX_RUN:
            result.append(line)
    return result


def tile_boxes(box: Sequence[float], grid: int, overlap: float) -> List[List[float]]:
    """
    Сетка grid x grid внутри `box` (доли листа) в порядке чтения: сверху
    вниз, слева направо. Соседние части перекрываются на `overlap` доли
    стороны всего листа, чтобы строка на стыке целиком попала хотя бы в одну.
    """
    grid = max(2, int(grid))
    x1, y1, x2, y2 = (float(value) for value in box)
    step_x, step_y = (x2 - x1) / grid, (y2 - y1) / grid
    half = max(0.0, float(overlap)) / 2
    boxes = []
    for row in range(grid):
        for column in range(grid):
            boxes.append([
                max(x1, x1 + column * step_x - half),
                max(y1, y1 + row * step_y - half),
                min(x2, x1 + (column + 1) * step_x + half),
                min(y2, y1 + (row + 1) * step_y + half),
            ])
    return boxes


def crop_box(image, box: Sequence[float]):
    """Часть картинки по долям стороны, в исходном разрешении."""
    width, height = image.size
    left, top = int(box[0] * width), int(box[1] * height)
    right = max(left + 1, int(round(box[2] * width)))
    bottom = max(top + 1, int(round(box[3] * height)))
    return image.crop((left, top, right, bottom))


def _line_key(line: str) -> str:
    return " ".join(line.split()).casefold()


def place_in_sheet(annotation: Dict[str, Any], box: Sequence[float]) -> Dict[str, Any]:
    """
    bbox надписи из долей части листа в доли всего листа. Модель отвечает в
    долях присланной картинки, а прислана ей была часть: без переноса все
    надписи частей легли бы на лист как попало, и место надписи врало бы.

    Ответ в пикселях (координаты заметно больше 1) не переносится: такой
    bbox всё равно отбрасывается при разборе, и домножать его не на что.
    """
    bbox = annotation.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return annotation
    try:
        coords = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return annotation
    if max(coords) > 1.5:
        return annotation
    left, top, right, bottom = (float(value) for value in box)
    width, height = right - left, bottom - top
    x1, y1, x2, y2 = coords
    return {**annotation, "bbox": [
        round(left + x1 * width, 4), round(top + y1 * height, 4),
        round(left + x2 * width, 4), round(top + y2 * height, 4),
    ]}


def _centre(bbox: Any) -> Tuple[float, float]:
    """Середина bbox в долях листа. Нет координат — середина листа."""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return 0.5, 0.5
        if max(x1, y1, x2, y2) <= 1.5:
            return (x1 + x2) / 2, (y1 + y2) / 2
    return 0.5, 0.5


def merge_annotations(
    parts: Sequence[Sequence[Dict[str, Any]]], near: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Надписи частей листа в один список без повторов из перекрытия.

    Повтор — это та же надпись, прочитанная двумя перекрывшимися частями:
    тот же текст и то же место. Поэтому ключ — текст плюс место, а не один
    текст: «⌀8» стоит у каждого отверстия, и восемь таких надписей на листе
    — восемь размеров, а не один. Надпись считается той же, если её
    середина ближе `near` доли стороны листа к уже добавленной: координаты
    от модели приблизительны, и точного совпадения у двух частей не бывает.
    """
    if near is None:
        near = settings.DRAWING_SHEET_TILE_NEAR
    near = max(0.0, float(near))
    emitted: Dict[str, List[Tuple[float, float]]] = {}
    merged: List[Dict[str, Any]] = []
    for part in parts:
        for annotation in part:
            if not isinstance(annotation, dict):
                continue
            text = _line_key(str(annotation.get("value") or ""))
            spot = _centre(annotation.get("bbox"))
            seen = emitted.setdefault(text, [])
            if any(abs(spot[0] - x) <= near and abs(spot[1] - y) <= near for x, y in seen):
                continue
            seen.append(spot)
            merged.append(annotation)
    return merged


def merge_lines(parts: Sequence[Sequence[str]]) -> List[str]:
    """
    Слияние строк частей листа без повторов из перекрытия. Строка части
    добавляется, если в этой части она встречается чаще, чем уже добавлено:
    повтор внутри одной части (две клеммы «XT1» на схеме) сохраняется, а
    та же строка из соседней части, видевшей перекрытие, — нет. Цена: одна
    и та же короткая надпись в разных частях вне перекрытия («PE», «N»)
    остаётся одной строкой — её текст в листе есть, теряется лишь повтор.
    """
    emitted: Dict[str, int] = {}
    merged: List[str] = []
    for part in parts:
        seen: Dict[str, int] = {}
        for line in part:
            key = _line_key(line)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > emitted.get(key, 0):
                emitted[key] = seen[key]
                merged.append(line)
    return merged


def _string_rows(rows: Any) -> List[List[str]]:
    if not isinstance(rows, list):
        return []
    result: List[List[str]] = []
    for row in rows:
        if isinstance(row, list):
            result.append([("" if cell is None else str(cell)) for cell in row])
        elif isinstance(row, dict):
            result.append([("" if value is None else str(value)) for value in row.values()])
    return result
