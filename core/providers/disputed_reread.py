"""
Третье свидетельство там, где оно нужно (карточка D4): дочитывание спорных
чисел вырезками в родном разрешении и сведение нескольких чтений листа.

Измеренные ошибки прохода «лист целиком» на сверенном эталоне 22 листов —
это ошибки одного символа в числе, которое прочитано в остальном верно:
знак отклонения («⌀4,5+0,2» прочитано как «⌀4,5−0,2»), потерянная или
лишняя запятая («1,5» как «15», «1349» как «134,9»), похожие знаки
(«Э4» как «34», «п.2» как «n.2») и запись резьбы («M18x1,5» как «M18x15»).
Лист ужимается до QWEN_SHEET_MAX_SIDE, и мелкий знак отклонения или запятая
на нём не различимы. Вырезка той же надписи из рендера листа в родном
разрешении показывает их модели крупно.

Цена ограничена. Дочитываются только поля чисел (размер, допуск, резьба,
радиус, шероховатость) и графы штампа «обозначение», «масса», «масштаб» —
и из них только спорные:
- если проход сведения (D3) уже пометил поля согласием (`agreement`),
  берутся «conflict» и «single» — его список главнее;
- иначе — своё простое правило: два прочтения одного числа рядом на листе
  с разной записью, и записи, в которых живут измеренные ошибки (знак
  отклонения, десятичная запятая, резьба, «п»/«n», «Э»/«З»).
До 12 вырезок идут одной картинкой с номерами, ответ — короткий список
строк. На эталоне это в среднем один вызов на лист.

Новое прочтение меняет значение поля, только если отличается от него
лишь там, где ошибаются: знак, запятая, звёздочка, похожие буквы. Всё
остальное (вырезка не той надписи, модель не прочитала, другое число)
оставляет поле как было: вырезка — свидетельство, а не новый источник
надписей. Неудачный вызов не меняет ничего.

С D3 (DRAWING_READINGS_RECONCILED) дочитывание идёт после сведения по
документу: в проходе листа сохраняются только рамки полей (`boxes`), спорные
поля выбирает согласие D3, а прочтение вырезки становится ещё одним
прочтением поля (`reading_reconcile.witness`). Своё сведение (группы двух
записей рядом) для полей D3 не повторяется.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.config import settings
from core.providers import drawing_text, gpu_gate, vector_drawing

logger = logging.getLogger(__name__)

NUMERIC = frozenset({"size", "tolerance", "thread", "radius", "roughness"})
# Графы штампа, которые ищут числом или кодом: обозначение, масса, масштаб.
_STAMP_FIELD = re.compile(r"^\s*(обознач|масса|масштаб)", re.IGNORECASE)
# Согласие прочтений по D3: что дочитывать.
DISPUTED_AGREEMENT = frozenset({"conflict", "single"})

CELLS = 12            # вырезок в одной картинке
_ROW = 112            # высота строки картинки, px
_LABEL = 84           # ширина поля номера, px
_MAX_WIDTH = 1100     # потолок ширины вырезки в строке, px
_MARGIN = 0.012       # запас вокруг рамки надписи, доля стороны листа
_NEAR = 0.05          # два прочтения ближе этого — одна надпись

# Признаки записей, в которых живут измеренные ошибки одного символа.
_SIGN = re.compile(r"\d\s*[+\-−±*]\s*\d|[+\-−±]\s*\d+[,.]\d")
_COMMA = re.compile(r"\d[,.]\d")
_THREAD = re.compile(r"^\s*[MМ]\s?\d|\d\s*[xх×]\s*\d")
_SIGNS = re.compile(r"[+\-−–±]")
_LETTER = re.compile(r"[nпNЭэЗз]\.?\s?\d")

# Знаки, в которых ошибается чтение мелкого шрифта. Две записи, равные после
# их снятия, — одна надпись, прочитанная с ошибкой одного знака.
_MINOR = str.maketrans({
    "+": None, "-": None, "−": None, "–": None, "±": None, "*": None,
    ",": None, ".": None, " ": None, "'": None, "’": None,
    "Э": "3", "э": "3", "З": "3", "з": "3", "n": "п", "N": "п", "П": "п",
    "х": "x", "Х": "x", "×": "x", "М": "M",
})


# ------------------------------------------------------------------ выбор
def _centre(bbox: Any) -> Optional[Tuple[float, float]]:
    """Середина рамки надписи; None — рамки нет или это область, а не надпись."""
    box = _place(bbox)
    if box is None:
        return None
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _place(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    if x2 - x1 > 0.5 or y2 - y1 > 0.5:
        return None  # рамка по умолчанию (весь лист) или крупная область
    return [x1, y1, x2, y2]


def eligible(field: Dict[str, Any]) -> bool:
    """Поле, которое дочитывается вырезкой: число или графа штампа с кодом."""
    category = field.get("category")
    if category in NUMERIC:
        return True
    return category == "title_block" and bool(_STAMP_FIELD.match(str(field.get("value") or "")))


def risky(value: str) -> bool:
    """Запись, в которой живут измеренные ошибки одного символа."""
    return bool(_SIGN.search(value) or _COMMA.search(value)
                or _THREAD.search(value) or _LETTER.search(value))


def _skeleton(value: str) -> str:
    return re.sub(r"[^0-9]", "", str(value).translate(_MINOR))


def _same(a: str, b: str) -> bool:
    """Одна запись: без пробелов и регистра, минусы одного вида."""
    clean = lambda s: re.sub(r"\s+", "", s.replace("−", "-").replace("–", "-")).casefold()  # noqa: E731
    return clean(a) == clean(b)


def reconciled(field: Dict[str, Any]) -> bool:
    """Поле сведено D3: есть его прочтения и согласие между ними."""
    return isinstance(field.get("readings"), list) and field.get("agreement") is not None


def targets(fields: Sequence[Dict[str, Any]]) -> List[Tuple[List[int], List[float]]]:
    """
    Что дочитывать: группы полей (номера в `fields`) и рамка вырезки.

    Группа из нескольких полей — одно число, прочитанное двумя записями
    рядом на листе: вырезка одна, и её прочтение выбирает запись. Порядок —
    сначала такие расхождения, потом одиночные спорные; срез по
    DRAWING_REREAD_MAX_CROPS.
    """
    conflicts: List[Tuple[List[int], List[float]]] = []
    singles: List[Tuple[List[int], List[float]]] = []
    taken: set = set()
    candidates = [n for n, f in enumerate(fields) if eligible(f) and _place(f.get("bbox"))]
    for n in candidates:
        if n in taken:
            continue
        if reconciled(fields[n]):
            # Поле уже сведено D3 со всеми прочтениями страницы: второй раз
            # его в группу не собирать, решает согласие D3.
            disputed = fields[n].get("agreement") in DISPUTED_AGREEMENT
            if disputed:
                (conflicts if fields[n]["agreement"] == "conflict" else singles).append(
                    ([n], _place(fields[n]["bbox"])))
            continue
        key = _skeleton(fields[n]["value"])
        here = _centre(fields[n]["bbox"])
        group = [n]
        for m in candidates:
            if (m <= n or m in taken or fields[m]["category"] != fields[n]["category"]
                    or reconciled(fields[m])):
                continue
            there = _centre(fields[m]["bbox"])
            if (key and _skeleton(fields[m]["value"]) == key
                    and not _same(fields[m]["value"], fields[n]["value"])
                    and math.dist(here, there) <= _NEAR):
                group.append(m)
        if len(group) > 1:
            taken.update(group)
            conflicts.append((group, _union([_place(fields[m]["bbox"]) for m in group])))
            continue
        # Согласие прочтений (D3 или сведение N чтений) — по полю: где оно
        # есть, решает оно; где нет — своё правило по записи.
        agreement = fields[n].get("agreement")
        disputed = (agreement in DISPUTED_AGREEMENT if agreement is not None
                    else risky(str(fields[n]["value"])))
        if disputed:
            (conflicts if agreement == "conflict" else singles).append(([n], _place(fields[n]["bbox"])))
    limit = max(0, int(settings.DRAWING_REREAD_MAX_CROPS))
    return (conflicts + singles)[:limit]


def _union(boxes: Sequence[List[float]]) -> List[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ---------------------------------------------------------------- картинка
def crop(image, bbox: Sequence[float], margin: float = _MARGIN):
    """
    Вырезка надписи в родном разрешении рендера с запасом. Высокая узкая
    вырезка — надпись у вертикального размера: она поворачивается по часовой
    стрелке, и текст, написанный снизу вверх, ложится строкой.
    """
    width, height = image.size
    x1 = max(0.0, bbox[0] - margin)
    y1 = max(0.0, bbox[1] - margin)
    x2 = min(1.0, bbox[2] + margin)
    y2 = min(1.0, bbox[3] + margin)
    box = (int(x1 * width), int(y1 * height),
           max(int(x1 * width) + 1, int(round(x2 * width))),
           max(int(y1 * height) + 1, int(round(y2 * height))))
    piece = image.crop(box).convert("RGB")
    if piece.height > piece.width * 1.5:
        piece = piece.rotate(-90, expand=True)
    return piece


def grid(pieces: Sequence[Any]):
    """
    До CELLS вырезок одной картинкой: строка на вырезку, слева номер в рамке.
    Вырезка масштабируется под высоту строки (мелкая — увеличивается), чтобы
    модель видела знак отклонения и запятую крупно.
    """
    from PIL import Image, ImageDraw

    rows = []
    for piece in pieces:
        scale = (_ROW - 8) / max(1, piece.height)
        size = (max(1, int(piece.width * scale)), _ROW - 8)
        if size[0] > _MAX_WIDTH:
            size = (_MAX_WIDTH, max(1, int(piece.height * _MAX_WIDTH / piece.width)))
        rows.append(piece.resize(size, Image.LANCZOS))
    width = _LABEL + max((r.width for r in rows), default=1) + 16
    sheet = Image.new("RGB", (width, _ROW * len(rows) or 1), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        top = index * _ROW
        draw.rectangle([2, top + 2, _LABEL - 6, top + _ROW - 3], outline="black", width=3)
        sheet.paste(_number(index + 1), (14, top + (_ROW - 48) // 2))
        sheet.paste(row, (_LABEL + 8, top + (_ROW - row.height) // 2))
        draw.line([0, top + _ROW - 1, width, top + _ROW - 1], fill=(160, 160, 160), width=1)
    return sheet


def _number(n: int):
    """Номер строки крупными цифрами: встроенный шрифт PIL, увеличенный вчетверо."""
    from PIL import Image, ImageDraw, ImageFont

    text = str(n)
    small = Image.new("L", (8 * len(text) + 2, 12), 255)
    ImageDraw.Draw(small).text((1, 0), text, fill=0, font=ImageFont.load_default())
    return small.resize((small.width * 4, small.height * 4), Image.NEAREST).convert("RGB")


def parse_values(data: Any, count: int) -> Dict[int, str]:
    """
    Ответ модели -> {номер вырезки: прочтение}. Принимаются
    {"values": [{"n": 1, "text": "…"}]}, {"values": ["…", …]} по порядку и
    {"1": "…"}. Номер вне 1..count и пустое прочтение отбрасываются.
    """
    found: Dict[int, str] = {}
    values = data.get("values") if isinstance(data, dict) else None
    if isinstance(values, list):
        for position, item in enumerate(values, start=1):
            if isinstance(item, dict):
                number, text = item.get("n"), item.get("text")
            else:
                number, text = position, item
            try:
                number = int(number)
            except (TypeError, ValueError):
                continue
            if isinstance(text, (str, int, float)) and str(text).strip() and 1 <= number <= count:
                found.setdefault(number, str(text).strip())
    elif isinstance(data, dict):
        for key, text in data.items():
            if str(key).isdigit() and isinstance(text, str) and text.strip() and 1 <= int(key) <= count:
                found.setdefault(int(key), text.strip())
    return found


# ---------------------------------------------------------------- решение
def decide(current: str, reread: str) -> Optional[str]:
    """
    Новое значение поля по прочтению вырезки или None — оставить как есть.

    Прочтение принимается, только если отличается от поля одними знаками,
    в которых ошибается чтение мелкого шрифта (знак отклонения, запятая,
    звёздочка, «Э»/«З»/«3», «п»/«n»): тогда это та же надпись, прочитанная
    крупно. Иное расхождение — вырезка показала не ту надпись или модель
    не дочитала; поле остаётся.
    """
    if not reread or _same(current, reread):
        return None
    if current.translate(_MINOR).casefold() != reread.translate(_MINOR).casefold():
        return None
    # Знак отклонения можно заменить или дописать, но не потерять: «40−0,25»,
    # прочитанное вырезкой как «400,25», — недочитанная вырезка, а не поправка.
    if len(_SIGNS.findall(reread)) < len(_SIGNS.findall(current)):
        return None
    return reread


def _with_value(field: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Поле с новым значением: разметка допуска и единицы — заново, как у `_field`."""
    label, sep, _ = str(field.get("value") or "").partition(": ")
    value = f"{label}{sep}{text}" if field.get("category") == "title_block" and sep else text
    updated = dict(field)
    updated["value"] = value
    if field.get("tolerance") is not None or field.get("category") in NUMERIC:
        updated["tolerance"] = vector_drawing.extract_tolerance(text)
        updated["source_of_tolerance"] = vector_drawing.tolerance_source(text)
    updated["nature"] = vector_drawing.determine_nature(text)
    updated["unit"] = vector_drawing.extract_unit(text, field.get("category") or "")
    return updated


def _read_part(value: str, category: Optional[str]) -> str:
    """Значение графы штампа без подписи: вырезка показывает только его."""
    if category == "title_block":
        return value.partition(": ")[2] or value
    return value


def _key(box: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(float(v), 6) for v in box)


def boxes(fields: Sequence[Dict[str, Any]]) -> set:
    """
    Рамки полей листа, которые может понадобиться дочитать. Нужны, когда
    дочитывание идёт после сведения D3: какие поля спорные, решит только
    D3, а вырезаются потом только выбранные — из заново открытого листа.
    """
    found: set = set()
    for f in fields:
        box = _place(f.get("bbox")) if eligible(f) else None
        if box is not None:
            found.add(_key(box))
    return found


def apply(
    client: Any, image, fields: List[Dict[str, Any]], until: Optional[float] = None,
    rule: Callable[[str, str], Optional[str]] = decide,
    allowed: Optional[set] = None,
    reconcile: Optional[Callable[[Dict[str, Any], str], Tuple[Dict[str, Any], str]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Дочитывает спорные поля листа. Возвращает новый список полей и учёт:
    сколько вырезок и вызовов, сколько подтверждено, исправлено, снято
    повторов и не прочитано. `until` — момент по `gpu_gate.budget_now()`,
    после которого вызов не начинается (бюджет листа). `rule` — правило
    значения.

    После сведения D3 картинки листа в памяти нет: `image` — функция без
    аргументов, которая открывает лист заново (зовётся, только если есть что
    дочитывать; None — лист не открылся), `allowed` — рамки, сохранённые в
    проходе листа (`boxes`). Поле, сведённое D3 (есть `readings`), решает `reconcile`:
    прочтение вырезки — ещё одно свидетельство в его прочтениях. Он
    возвращает поле и исход: «agreed», «corrected» или «kept».
    """
    chosen = targets(fields)
    if allowed is not None:
        chosen = [item for item in chosen if _key(item[1]) in allowed]
    stats: Dict[str, Any] = {"crops": len(chosen), "calls": 0, "failed_calls": 0,
                             "agreed": 0, "corrected": 0, "dropped": 0, "unread": 0,
                             "skipped_time": 0, "changes": []}
    if not chosen:
        return fields, stats
    if callable(image):
        if until is not None and gpu_gate.budget_now() + settings.VLM_TIMEOUT > until:
            stats["skipped_time"] = len(chosen)
            return fields, stats
        image = image()
        if image is None:
            stats["failed_calls"] = 1
            stats["unread"] = len(chosen)
            return fields, stats
    answers: Dict[int, str] = {}
    for start in range(0, len(chosen), CELLS):
        batch = chosen[start:start + CELLS]
        if until is not None and gpu_gate.budget_now() + settings.VLM_TIMEOUT > until:
            stats["skipped_time"] += len(batch)
            continue
        try:
            picture = grid([crop(image, box) for _, box in batch])
        except Exception as exc:  # noqa: BLE001 — битая вырезка не роняет лист
            logger.error("Вырезки спорных надписей не собраны: %s", exc)
            stats["failed_calls"] += 1
            continue
        stats["calls"] += 1
        read = client.read_values(picture, len(batch))
        if read is None:
            stats["failed_calls"] += 1
            continue
        for number, text in read.items():
            answers[start + number - 1] = drawing_text.normalize(text)

    drop: set = set()
    result = list(fields)
    for index, (members, _) in enumerate(chosen):
        text = answers.get(index)
        if text is None:
            stats["unread"] += 1
            continue
        if reconcile is not None and len(members) == 1 and reconciled(result[members[0]]):
            n = members[0]
            before = str(result[n]["value"])
            result[n], outcome = reconcile(result[n], text)
            if outcome in ("agreed", "corrected"):
                stats[outcome] += 1
            if outcome == "corrected":
                stats["changes"].append([before, str(result[n]["value"])])
            continue
        for n in members:
            readings = result[n].get("readings")
            if isinstance(readings, list):
                result[n] = {**result[n], "readings": readings + [{"source": "crop_reread", "value": text}]}
        values = [_read_part(str(result[n]["value"]), result[n].get("category")) for n in members]
        same = [n for n, v in zip(members, values) if _same(v, text)]
        if same:
            stats["agreed"] += 1
            keep = same[0]
        else:
            keep = None
            for n, v in zip(members, values):
                new = rule(v, text)
                if new is not None:
                    stats["corrected"] += 1
                    stats["changes"].append([str(result[n]["value"]), new])
                    result[n] = _with_value(result[n], new)
                    keep = n
                    break
            if keep is None:
                continue
        others = [n for n in members if n != keep]
        drop.update(others)
        stats["dropped"] += len(others)
    if drop:
        result = [f for n, f in enumerate(result) if n not in drop]
    logger.info(
        "Спорные надписи дочитаны вырезками: вырезок %d, вызовов %d, подтверждено %d, "
        "исправлено %d, снято повторов %d, без ответа %d",
        stats["crops"], stats["calls"], stats["agreed"], stats["corrected"],
        stats["dropped"], stats["unread"],
    )
    return result, stats


# ------------------------------------------------------- сведение N чтений
def consensus(runs: Sequence[Sequence[Dict[str, Any]]], near: float = _NEAR) -> List[Dict[str, Any]]:
    """
    Надписи N чтений листа (DRAWING_CONSENSUS_READS) одним списком.

    Надпись одного чтения сопоставляется с надписями других по
    нормализованному значению и месту (середина ближе `near`). Надпись,
    найденная хотя бы в двух чтениях, идёт с `agreement` = «agree», найденная
    в одном — с «single»; так её потом дочитает вырезка. Порядок — порядок
    первого появления.
    """
    groups: List[Dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        for annotation in run:
            if not isinstance(annotation, dict):
                continue
            value = str(annotation.get("value") or "")
            key = re.sub(r"\s+", "", drawing_text.normalize(value)).casefold()
            spot = _centre(annotation.get("bbox"))
            for group in groups:
                if group["key"] != key or run_index in group["runs"]:
                    continue
                if spot is None or group["spot"] is None or math.dist(spot, group["spot"]) <= near:
                    group["runs"].add(run_index)
                    break
            else:
                groups.append({"key": key, "spot": spot, "runs": {run_index}, "item": annotation})
    return [{**g["item"], "agreement": "agree" if len(g["runs"]) >= 2 else "single"} for g in groups]


# ---------------------------------------------------------- плитки листа
def native_scale(image) -> float:
    """Доля пикселей рендера, которая доходит до модели в проходе листа (по стороне)."""
    side = max(image.size) if image is not None else 0
    limit = settings.QWEN_SHEET_MAX_SIDE
    return 1.0 if not side or not limit or side <= limit else limit / side


def needs_native_tiles(image) -> bool:
    """
    Лист читается плитками сразу (DRAWING_SHEET_NATIVE_TILES): ужатие ниже
    DRAWING_NATIVE_MIN_SCALE. Без порога не включается — его дают D1 и
    эталон листов, догадки по умолчанию нет.
    """
    threshold = settings.DRAWING_NATIVE_MIN_SCALE
    if not settings.DRAWING_SHEET_NATIVE_TILES or threshold is None:
        return False
    return native_scale(image) < float(threshold)
