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
import io
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

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


@dataclass
class SheetReading:
    """Прочитанный лист: надписи и род чертежа."""

    sheet_type: str = "unknown"
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


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
    def _ask(self, image, prompt: str, max_side: int) -> Optional[Dict[str, Any]]:
        if not self.available:
            logger.error("QWEN_ENDPOINT не задан — разбор чертежа невозможен")
            return None
        encoded = encode_image(image, max_side)
        if not encoded:
            return None

        body = {
            "model": self.model,
            "temperature": settings.QWEN_TEMPERATURE,
            "max_tokens": settings.QWEN_MAX_TOKENS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": encoded}},
                    {"type": "text", "text": prompt},
                ]},
            ],
        }
        content = self._post(body)
        if content is None:
            return None

        data = extract_json(content)
        if data is None:
            logger.error("Модель вернула не-JSON (%d символов): %.200s", len(content), content)
        return data

    def _post(self, body: Dict[str, Any]) -> Optional[str]:
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
                return _content_of(response.json())
            except ValueError as exc:
                logger.error("Qwen вернул не-JSON на уровне протокола: %s", exc)
                return None
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
