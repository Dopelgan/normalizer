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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from core import deadline
from core.config import settings

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


@dataclass
class SheetReading:
    """Прочитанный лист: надписи и род чертежа."""

    sheet_type: str = "unknown"
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


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

    def read_sheet(self, image, hint: str = "") -> Optional[SheetReading]:
        """Лист целиком: надписи с их местами и род чертежа."""
        prompt = _SHEET_PROMPT.format(
            categories=", ".join(CATEGORIES),
            hint=f"\n- Подсказка по разметке листа: {hint}" if hint else "",
        )
        data = self._ask(image, prompt, settings.QWEN_SHEET_MAX_SIDE)
        if data is None:
            return None
        annotations = data.get("annotations")
        if not isinstance(annotations, list):
            logger.warning("Модель вернула JSON без списка annotations")
            annotations = []
        return SheetReading(
            sheet_type=str(data.get("sheet_type") or "unknown"),
            annotations=[a for a in annotations if isinstance(a, dict)],
            raw=data,
        )

    def read_title_block(self, image) -> Optional[Dict[str, Any]]:
        """Вырезанный штамп -> графы и та же таблица построчно."""
        data = self._ask(image, _TITLE_BLOCK_PROMPT, settings.QWEN_REGION_MAX_SIDE)
        if data is None:
            return None
        fields = data.get("fields")
        named = (
            {str(key): str(value) for key, value in fields.items() if value not in (None, "")}
            if isinstance(fields, dict) else {}
        )
        return {"fields": named, "rows": _string_rows(data.get("rows"))}

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
            image, _TRANSCRIBE_PROMPT, settings.DRAWING_TRANSCRIBE_MAX_SIDE,
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
            image, _TRANSCRIBE_PROMPT, settings.DRAWING_TRANSCRIBE_MAX_SIDE,
            max_tokens=settings.DRAWING_TRANSCRIBE_MAX_TOKENS,
        )
        if answer is None:
            return None, False
        content, finish = answer
        data = extract_json(content)
        if data is not None and finish != "length":
            if isinstance(data.get("lines"), list):
                return _clean_lines(data["lines"]), False
            logger.warning("Модель вернула JSON без списка lines")
            return None, False
        return salvage_lines(content), True

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
    return [str(line).strip() for line in lines
            if isinstance(line, (str, int, float)) and str(line).strip()]


def salvage_lines(content: str) -> List[str]:
    """
    Законченные элементы `lines` из оборванного ответа
    `{"lines": ["…", "…", "незаконч`. Строка, на которой ответ оборвался,
    отбрасывается: её дочитает часть листа.
    """
    if not content:
        return []
    key = content.find('"lines"')
    start = content.find("[", key) if key != -1 else -1
    if start == -1:
        return []
    decoder = json.JSONDecoder()
    position, found = start + 1, []
    while True:
        while position < len(content) and content[position] in " \t\r\n,":
            position += 1
        if position >= len(content) or content[position] == "]":
            break
        try:
            value, position = decoder.raw_decode(content, position)
        except ValueError:
            break
        found.append(value)
    return _collapse_runs(_clean_lines(found))


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
