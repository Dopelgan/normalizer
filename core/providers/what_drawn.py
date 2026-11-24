"""
«Что изображено» на листе чертежа (DRAWING_WHAT_DRAWN).

Один вызов модели на лист детали или сборки: картинка листа целиком и
пронумерованные уже прочитанные величины с их рамками. Модель отвечает
простым русским языком: что изображено, что показывает каждый вид и что
измеряет каждая величина. Ответ проверяется правилами без модели:

- строка с неизвестным номером величины отбрасывается;
- смысл должен сходиться со знаком величины: «⌀» — диаметр, отверстие или
  окружность; «R» — радиус или скругление; «S» — толщина; величина без
  знака — не диаметр и не радиус;
- одна и та же величина (по написанию) с двумя разными смыслами смысла не
  получает;
- в смысле величины нет чисел, которых нет среди прочитанных величин;
- фраза о том, как части складываются в целое, остаётся, только если её
  арифметика сходится на прочитанных величинах.

Связь величины с элементом из `DRAWING_DIMENSION_LINKS` (подпись
`link.target_label`) идёт в запрос подсказкой и предпочитается ответу
модели: она построена по выноске на самом листе.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERSION = 1

# Категории полей, которые объясняются как величины.
DIM_CATEGORIES = frozenset({"size", "radius", "roughness", "thread", "tolerance"})
# Сколько прочих надписей (позиции, буквы видов, примечания) идёт в запрос.
CONTEXT_MAX = 30
NOTES_MAX = 12

SCHEMA = """{"object": "<1-2 предложения: что изображено, форма, примерный размер по величинам>",
 "how_parts_fit": "<1 предложение: как главные части и размеры связаны между собой, например «общая высота 505 = корпус 305 + патрубки сверху и снизу»; только то, что видно на листе>",
 "views": [{"name": "главный вид|вид сверху|разрез А-А|выносной элемент Б|...", "shows": "что видно на этом виде"}],
 "features": [{"id": "f1", "kind": "hole|bend|fillet|flange|pipe|weld|edge|surface|other", "where": "простыми словами", "count": 1}],
 "dimensions": [{"value_id": "v1", "measures": "что измеряет, простыми словами", "feature_id": "f1 или null", "confidence": "high|low"}]}"""

RULES = """Правила:
- Пиши простым русским языком для новичка, не инженера. Термин стандарта — только с пояснением в 3 слова.
- Используй ТОЛЬКО перечисленные величины, никаких новых чисел не придумывай.
- По каждой величине из списка размеров дай ровно одну строку в "dimensions" с её value_id.
- Если не видно, что измеряет величина, пиши "measures": "не видно" и "confidence": "low".
- "confidence": "high" — только если на листе ясно видна размерная линия или выноска от надписи к элементу.
- Звёздочка (*) у размера — размер для справки; ⌀ — диаметр; R — радиус скругления; Ra — шероховатость (гладкость поверхности).
- Название изделия возьми из основной надписи (рамка внизу справа) и назови его в "object".
- Рамка величины [x0,y0,x1,y1] — доли ширины и высоты листа от левого верхнего угла. Найди надпись именно в этом месте и проследи её размерную линию (тонкая, со стрелками на концах) до выносных линий: величина — расстояние между выносными линиями или элемент, на который указывает стрелка выноски.
- Размерная линия параллельна измеряемому отрезку: надпись вдоль вертикальной линии — размер по вертикали, вдоль горизонтальной — по горизонтали.
- Сначала определи вид, к которому относится надпись (по её месту), и называй элемент с указанием вида: «на главном виде», «на разрезе А-А», «на выносном элементе Б».
Как читать знаки (ГОСТ 2.307):
- ⌀ — всегда диаметр круглого: отверстия, трубы, вала, окружности. Никогда не радиус и не длина. Если у одной круглой детали несколько ⌀: самый большой — наружный, меньший внутри — внутреннее отверстие; ⌀ штрихпунктирной окружности, на которой стоят центры отверстий, — окружность расположения отверстий.
- Круглая плоская деталь (диск, кольцо, фланец), показанная сбоку или в разрезе, выглядит узким прямоугольником: ⌀ на ней — диаметры, а узкий размер поперёк — толщина.
- R — радиус дуги: скругления угла, гиба. Никогда не отверстие.
- «N отв. ⌀d» — N одинаковых отверстий диаметром d; N — количество, а не размер.
- Число без знака — линейный размер: длина, ширина, высота, толщина или расстояние между осями/краями. Число с ° — угол (фаска, наклон).
- * — справочный размер (для сведения); «X», «Y» в размере — параметр, который задаётся заказом.
- S перед числом у листовой детали — толщина листа.
- Небольшие размеры на увеличенном выносном элементе у сварного шва или кромки — размеры шва или разделки кромки (зазор, катет, притупление), а не пазы.
- Знак «круг со стрелкой» над видом — развёртка: деталь до гибки, плоская заготовка; её размеры — размеры заготовки, а не готового изделия.
- Ra — шероховатость (гладкость поверхности); Ra в правом верхнем углу листа — для всех поверхностей, где не указано иное.
- "confidence": "high" — только если ты видишь оба конца размерной линии или остриё выноски и можешь назвать, между чем они.
- Таблицу обозначений на листе не описывай построчно, только упомяни в "object".
Ответ — только JSON по схеме:"""

_TITLE_PREFIX = "Наименование:"
_NUMBER = re.compile(r"\d+(?:[,.]\d+)?")


def _box(raw: Any) -> List[float]:
    try:
        return [round(float(c), 3) for c in raw][:4]
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]


def _label(item: Dict[str, Any]) -> str:
    link = item.get("link")
    if isinstance(link, dict):
        return str(link.get("target_label") or "").strip()
    return ""


def build_prompt(fields: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Текст запроса и пронумерованные величины листа:
    `[{"id": "v1", "value": ..., "bbox": ..., "link": <подпись связи или "">}]`.
    Величин нет — текст пустой, вызова не будет.
    """
    dims: List[Dict[str, Any]] = []
    context: List[Dict[str, Any]] = []
    title, notes, seen = "", [], set()
    for item in fields:
        value = str(item.get("value") or "").strip()
        if not value:
            continue
        category = item.get("category")
        if category in DIM_CATEGORIES:
            dims.append({"id": f"v{len(dims) + 1}", "value": value, "bbox": _box(item.get("bbox")),
                         "link": _label(item)})
            continue
        if category == "title_block" and value.startswith(_TITLE_PREFIX) and not title:
            title = value[len(_TITLE_PREFIX):].strip()
        if category == "note" and re.match(r"^\s*\d+[.)]?\s", value) and len(notes) < NOTES_MAX:
            notes.append(value[:160])
        if value not in seen and len(context) < CONTEXT_MAX:
            seen.add(value)
            context.append({"value": value[:80], "bbox": _box(item.get("bbox"))})
    if not dims:
        return "", []
    lines = ["Это лист машиностроительного чертежа. Объясни новичку, что на нём изображено."]
    if title:
        lines.append(f"Основная надпись: {title}")
    if notes:
        lines.append("Примечания листа: " + " | ".join(notes))
    lines.append("Размеры, уже прочитанные с листа (id, значение, рамка надписи в долях листа):")
    for d in dims:
        hint = f"  выноска ведёт к: {d['link']}" if d["link"] else ""
        lines.append(f"{d['id']}: {d['value']}  рамка {d['bbox']}{hint}")
    if context:
        lines.append("Для справки (не размеры: номера позиций, буквы видов, надписи):")
        lines.append("; ".join(f"{c['value']} {c['bbox']}" for c in context))
    lines += [RULES, SCHEMA]
    return "\n".join(lines), dims


# ---------------------------------------------------------------- проверка

def sign_of(value: str) -> str:
    """Знак величины: dia, rad, thick, Ra, angle, count, note, lin."""
    v = value.strip()
    if v.startswith("Ra") or v.startswith("Rz"):
        return "Ra"
    if v.startswith(("⌀", "Ø", "ø")):
        return "dia"
    if re.match(r"^S?R\d", v):
        return "rad"
    if re.match(r"^S\d", v):
        return "thick"
    if "°" in v:
        return "angle"
    if "отв" in v:
        return "count"
    if re.match(r"^\d+\.\s", v):
        return "note"
    return "lin"


def contradicts(value: str, meaning: str) -> bool:
    """Смысл противоречит знаку величины (или его нет вовсе)."""
    m = (meaning or "").lower()
    if not m.strip() or "не видно" in m:
        return True
    kind = sign_of(value)
    if kind == "dia":
        return not any(w in m for w in ("диаметр", "отверст", "окружност", "круг"))
    if kind == "rad":
        return not any(w in m for w in ("радиус", "скругл", "дуг"))
    if kind == "Ra":
        return "шерохов" not in m and "гладк" not in m
    if kind == "thick":
        return "толщин" not in m
    if kind == "angle":
        return "угол" not in m and "наклон" not in m
    if kind == "lin":
        return "диаметр" in m or "радиус" in m or "шерохов" in m
    return False


def _norm_meaning(meaning: str) -> str:
    m = re.sub(r"\s+на\s+(главном виде|виде\s+\S+|разрезе\s+\S+|выносном элементе\s+\S+).*$", "",
               (meaning or "").lower())
    return re.sub(r"\s+", " ", m).strip(" .")


def _known_numbers(dims: Sequence[Dict[str, Any]]) -> set:
    return {n.replace(".", ",") for d in dims for n in _NUMBER.findall(d["value"])}


def _invents_numbers(text: str, known: set) -> bool:
    return any(n.replace(".", ",") not in known for n in _NUMBER.findall(text or ""))


def parts_fit(sentence: Any, dims: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Фраза о связи частей — только без чужих чисел и со сходящейся суммой."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    sentence = sentence.strip()
    values = " ".join(d["value"] for d in dims)
    for token in re.findall(r"\(?[XYХ]\s*\+\s*\d+\)?", sentence):
        if token.replace(" ", "") not in values.replace(" ", ""):
            return None
    nums = _NUMBER.findall(sentence)
    if _invents_numbers(sentence, _known_numbers(dims)):
        return None
    if re.search(r"складыва|из которых|состоит из|=|\+ ", sentence):
        main = [float(n.replace(",", ".")) for n in nums
                if not re.search(r"±\s*" + re.escape(n), sentence)]
        if len(main) < 3 or abs(max(main) - (sum(main) - max(main))) > 0.5:
            return None
    return sentence


def _text(raw: Any) -> str:
    return re.sub(r"\s+", " ", raw).strip() if isinstance(raw, str) else ""


def parse(data: Any, dims: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Ответ модели -> `what_drawn` листа или None (ответа нет, он пустой).
    `dims` — величины из `build_prompt`.
    """
    if not isinstance(data, dict):
        return None
    obj = _text(data.get("object"))
    views = []
    for view in data.get("views") or []:
        if isinstance(view, dict) and _text(view.get("name")) and _text(view.get("shows")):
            views.append({"name": _text(view.get("name")), "shows": _text(view.get("shows"))})
    features = {}
    for feat in data.get("features") or []:
        if isinstance(feat, dict) and isinstance(feat.get("id"), str) and _text(feat.get("where")):
            features[feat["id"]] = _text(feat.get("where"))
    by_id = {d["id"]: d for d in dims}
    model: Dict[str, Dict[str, Any]] = {}
    for row in data.get("dimensions") or []:
        # Неизвестный номер — величина, которой на листе не прочитано.
        if isinstance(row, dict) and row.get("value_id") in by_id and row["value_id"] not in model:
            model[row["value_id"]] = row
    known = _known_numbers(dims)
    picked: List[Tuple[Dict[str, Any], str, Optional[str]]] = []
    meanings: Dict[str, set] = {}
    for d in dims:
        row = model.get(d["id"])
        if d["link"]:
            meaning = d["link"]
        elif row is not None and row.get("confidence") == "high":
            meaning = _text(row.get("measures"))
        else:
            continue
        if contradicts(d["value"], meaning) or _invents_numbers(meaning, known):
            continue
        feature = features.get(row.get("feature_id")) if row is not None else None
        meanings.setdefault(d["value"], set()).add(_norm_meaning(meaning))
        picked.append((d, meaning, feature))
    # Одна величина — один смысл: разные смыслы у одинаковых значений — ни одного.
    clash = {value for value, found in meanings.items() if len(found) > 1}
    rows, seen = [], set()
    for d, meaning, feature in picked:
        key = (d["value"], meaning, feature)
        if d["value"] in clash or key in seen:
            continue
        seen.add(key)
        rows.append({"value": d["value"], "measures": meaning, "feature": feature})
    if not obj and not views and not rows:
        return None
    return {
        "version": VERSION,
        "object": obj,
        "views": views,
        "dimensions": rows,
        "parts_fit": parts_fit(data.get("how_parts_fit"), dims),
    }
