"""
Запись чертёжного листа: собрать прочитанное так, как чертёж читают.

Зачем это нужно. До этого модуля текст записи чертежа — строки, которые
модель выдала, переписывая лист (`drawing_processor.build_fragment`,
`content.text`). Строки идут в порядке выдачи, и в этом порядке запись
нечитаема: числа стоят без того, что они измеряют, номера позиций
перемешаны с размерами, названия видов — рядом с посторонними цифрами, а
подписи боковых граф рамки («Инв. № подл.», «Подп. и дата») приходят без
значений. Настоящая запись листа ТАДУ 302318.207.00 начиналась так:

    ТАДУ 302318.207.00 СБ / Справ. N / Перв. применение / A / 6 / N°1 /
    N°1 / 5 / R65 / N°1 / 8 / 2 / B / A / 7 / … / 629 / 03.2020

Всё прочитано, и ответить по этому нельзя ни на что. При этом подписанные
величины у разбора **уже есть** и лежат в том же фрагменте: графы штампа —
в `structured_payload["title_block"]` (там и `mass: "26,02"`, которого в
тексте записи нет вовсе), рукописные графы — в `title_block_handwritten`,
размеры с их родом — в `structured_drawing_fields`, таблицы листа — в
`structured_payload["tables"]`. В текст записи они не попадали никогда.

Чем это является и чем не является. Это правка **читаемости**, а не возврат
потерянных данных, и путать их нельзя. Адаптер RAG
(`super_rag/parser/adapter.py`, `_fragment_search_text`) у фрагмента, у
которого `content.text` пуст, уже отдавал в индекс поля и весь
`structured_payload` каноническим JSON под меткой `[DRAWING]` — то есть
масса в поиске была, в виде JSON. Таких фрагментов большинство: 4687 из
5110 на проде. Для них сборка JSON не дублирует, а **заменяет** его прозой
(адаптер возвращает текст, едва он появился), и индексируемый текст
становится короче примерно в двенадцать раз. У остальных 423 фрагментов
текст листа есть, адаптер отдаёт только его, и JSON не отдавался никогда —
вот там графы штампа в поиск действительно не попадали, и именно такие
листы читал заказчик, когда забраковал запись.

Что делает модуль. Собирает из того же фрагмента запись по частям, у каждой
части — название, у каждого значения — то, чем оно является:

1. основная надпись — графы ГОСТ 2.104 со значениями;
2. записи от руки (боковые графы, даты подписей, графа изменений);
3. виды, разрезы и сечения, названные на листе;
4. размеры, сгруппированные по роду величины (диаметры, радиусы, резьба,
   шероховатость, углы, допуски, линейные);
5. номера позиций;
6. обозначения и выноски;
7. технические требования и примечания;
8. таблицы листа — каждое значение под именем своего столбца;
9. надписи листа в порядке чтения — то, чем запись была до сборки.

Чего модуль не делает — и это главное ограничение. Он **не привязывает
размер к виду**: в разборе нет областей видов, а названия видов приходят
строками без координат. Привязка была бы догадкой, поэтому размеры
перечислены под честным заголовком «к какому виду относится каждый размер,
запись чертежа не указывает». Непривязанный список под ясным заголовком
полезнее нынешнего перемешивания и не врёт.

Числа и обозначения переносятся посимвольно: не переписываются, не
склеиваются, не округляются, единицы не подставляются (в штампе «Масса —
26,02», без «кг», которого в данных нет).

Потерь нет по построению: часть 9 — это весь прежний текст записи, из
которого убраны только подписи боковых граф рамки без значения (они
перечислены в `FURNITURE`, их число уходит в
`structured_payload["record_dropped_furniture"]`).

Связи. Вызывается из `core.contract_finalize.finalize_fragments` — после
ветки чертежей, после сверки «чертёж против OCR» и после извлечения
фактов, то есть позже всех, кто читает `content.text` внутри конвейера.
Поэтому сборка не меняет ни правку абзацев OCR
(`core/drawing_transcript.py`), ни объём для сверки
(`core/ladder/drawing_route.volume`), ни факты, ни `fragment_id`.

Флаг `DRAWING_RECORD_COMPOSED`, по умолчанию выключен: при выключенном
флаге текст записи не меняется ни на байт.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.models.contract import Fragment
from core.providers.title_block import (
    HANDWRITTEN_MARK, LABELS, canonical, label,
)

# Ключ в payload: сколько подписей граф без значения убрано из текста листа.
DROPPED_KEY = "record_dropped_furniture"
# Ключ в payload: запись собрана по частям (для отладки и замеров).
COMPOSED_KEY = "record_composed"

# Оговорка о непривязанных размерах. Одна на запись: областей видов в
# разборе нет, и сказать это надо честно, но один раз.
NO_VIEW_LINK = ("К какому виду или разрезу относится каждый размер, "
                "запись чертежа не указывает.")

# Графы основной надписи в том порядке, в котором их спрашивают.
# Подписи берутся из `title_block.LABELS`: имя графы здесь и в разборе
# штампа обязано быть одним и тем же.
_STAMP_ORDER: Sequence[str] = (
    "designation", "name", "material", "mass", "scale", "letter",
    "sheet", "sheets", "format", "organization",
)
# Подписавшие лист: отдельной фразой, глаголом, а не перечнем граф.
_SIGNERS: Sequence[tuple] = (
    ("developed", "разработал"),
    ("checked", "проверил"),
    ("tech_control", "технический контроль"),
    ("norm_control", "нормоконтроль"),
    ("approved", "утвердил"),
)
# Графы, которые заполняются от руки: идут отдельной частью с оговоркой.
_HAND_ORDER: Sequence[str] = (
    "changes", "inventory", "inventory_dup", "replaced_inventory",
    "first_use", "reference", "signed_date",
)

# Подписи боковых граф рамки, приходящие строками текста листа. Значения у
# них нет: графа подшивки на листе пуста, а то, что в ней стоит, разбор
# приносит графами (`title_block_handwritten`). Только эти подписи и только
# целой строкой без цифр убираются из текста листа — всё остальное
# остаётся, в том числе «Масса» и «Масштаб»: рядом с ними в строках может
# стоять значение, и подпись — единственное, что о нём говорит.
FURNITURE: frozenset = frozenset({
    "инв № подл", "инв № подл.", "инв номер подлинника",
    "инв № дубл", "инв № дубл.",
    "взам инв №", "взам инв номер",
    "перв примен", "перв применение", "первичное применение",
    "справ №", "справ n", "справочный №",
    "подп и дата", "подпись и дата",
})

# Род величины -> как он называется в записи. Порядок — порядок частей.
_KIND_TITLES: Sequence[tuple] = (
    ("diameter", "Диаметры"),
    ("radius", "Радиусы"),
    ("thread", "Резьба"),
    ("roughness", "Шероховатость"),
    ("angle", "Углы"),
    ("tolerance", "Допуски и посадки"),
    ("linear", "Линейные размеры"),
)

# Знак диаметра во всех начертаниях. Образец тот же, что в
# `core/providers/drawing_text.py`: векторный путь складывает в поля то, что
# прочитал, не приводя знак к одному виду, и «Ø57» иначе уходило бы в
# линейные размеры.
_DIAMETER_GLYPHS = re.compile(r"[Øøᴓ⌀ΦϕφФ]")
_ANGLE_SIGN = re.compile(r"°")
# Надпись сама говорит, что она за величина: предельное отклонение, знак
# умножения размеров или названная единица.
_MEASURED_SIGN = re.compile(r"[±]|\d\s*[xх×]\s*\d|[+\-]\s*\d|\d\s*(мм|mm|м\b|°)")
# Название вида, разреза или сечения по ГОСТ 2.305 — целой строкой.
# Либо парная буква («А-А», «Б—Б»), либо буква с масштабом в скобках
# («Б(2:1)», «Г(2)»), либо строка со словом «вид»/«разрез»/«сечение».
# Одиночная буква («A») видом не считается: на листе она и обозначение
# места разреза, и номер исполнения — назвать её видом было бы догадкой.
_VIEW_PAIR = re.compile(
    r"^([А-ЯЁA-Z])\s*[-–—]\s*\1\s*(\([^()]*\))?$")
# Масштаб по ГОСТ 2.302 — всегда отношение, с двоеточием: «Б(2:1)», «A(1:4)».
# Без двоеточия буква с числом видом не считается: «Б(4)» это номер
# исполнения или позиции, и на настоящих листах таких «видов» набиралось по
# восемнадцать штук. Это та же неоднозначность, из-за которой видом не
# считается и одиночная буква.
_VIEW_SCALED = re.compile(
    r"^[А-ЯЁA-Z]\s*\(\s*\d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?\s*\)$")
_VIEW_WORD = re.compile(
    r"^(вид|разрез|сечение|развёртка|развертка)\b", re.IGNORECASE)
# Строка-название вида не бывает длинной: «Вид на изделие со снятой…» —
# это примечание, а не заголовок вида.
_VIEW_MAX_CHARS = 40

_FOLD = re.compile(r"[\s.,:;№\-_]+")


def _fold(text: str) -> str:
    """Подпись графы без того, чем её украсили: точек, пробелов, «№»."""
    return _FOLD.sub(" ", str(text).strip().lower()).replace("ё", "е").strip()


_FURNITURE_FOLDED = frozenset(_fold(name) for name in FURNITURE)
# Подписи граф ГОСТ 2.104 в свёрнутой записи: по ним узнаётся сетка штампа.
_LABELS_FOLDED = (
    frozenset(_fold(label) for label in LABELS.values())
    | _FURNITURE_FOLDED
    | frozenset(_fold(name) for name in (
        "N докум.", "№ докум.", "Докум.", "Подп.", "Дата", "Проверил",
        "Лист", "Листов", "Лит.", "Изм.", "Масса", "Масштаб",
    ))
)
# `kind` таблицы, которой разбор помечает сетку основной надписи.
_TITLE_BLOCK_KIND = "title_block"
# Чем разделяются два прочтения одной графы.
_GRAPH_JOIN = "; "
# Меньше стольких подписей граф — сеткой штампа таблица не считается:
# на двух-трёх ячейках доля ничего не значит.
_STAMP_GRID_MIN_LABELS = 3
_DIGIT = re.compile(r"\d")


def enabled() -> bool:
    from core.config import settings

    return bool(getattr(settings, "DRAWING_RECORD_COMPOSED", False))


def apply(fragments: List[Fragment]) -> List[Fragment]:
    """
    Запись каждого чертёжного фрагмента собирается по частям.

    При выключенном флаге не делается ничего: текст уходит байт в байт
    таким, каким был.
    """
    if not enabled():
        return fragments
    for fragment in fragments:
        if fragment.type != "drawing":
            continue
        compose_fragment(fragment)
    return fragments


def compose_fragment(fragment: Fragment) -> Optional[str]:
    """
    Собирает текст записи фрагмента на месте и возвращает его.

    `None` — собирать нечего (ни строк листа, ни полей, ни таблиц): такой
    фрагмент остаётся как был.
    """
    content = fragment.content
    payload = dict(content.structured_payload or {})
    if payload.get(COMPOSED_KEY):
        # Запись этого фрагмента уже собрана. Второй проход вложил бы
        # сборку внутрь её же части «Надписи листа в порядке чтения», и
        # заголовок перестал бы соответствовать тому, что под ним стоит.
        return content.text
    fields = [
        {"category": f.category, "value": f.value, "tolerance": f.tolerance,
         "unit": f.unit}
        for f in (content.structured_drawing_fields or [])
    ]
    lines = [line for line in (content.text or "").split("\n")]
    kept, dropped = _without_furniture(lines)

    parts: List[str] = []
    parts.extend(_stamp(payload, fields))
    parts.extend(_handwritten(payload, fields))
    # Оговорка про непривязанные размеры говорится один раз — в той из двух
    # частей, которая встретится первой.
    views = _views(kept, hedge=True)
    parts.extend(views)
    parts.extend(_dimensions(fields, hedge=not views))
    parts.extend(_positions(fields))
    parts.extend(_callouts(fields))
    parts.extend(_notes(fields))
    parts.extend(_tables(payload))
    parts.extend(_sheet_lines(kept))

    if not parts:
        return None
    text = "\n\n".join(parts)
    content.text = text
    payload[COMPOSED_KEY] = True
    if dropped:
        payload[DROPPED_KEY] = dropped
    content.structured_payload = payload
    return text


# ----------------------------------------------------------------- части

def _corrected(fields: Sequence[Dict[str, Any]]) -> Dict[tuple, str]:
    """
    Исправленные значения граф штампа, взятые из полей фрагмента.

    Зачем. Разбор штампа кладёт в `structured_payload["title_block"]` ответ
    модели как есть, а `drawing_text.normalize` применяет только к тем
    значениям, которые уходят в `structured_drawing_fields` (там графа
    записана как «Подпись: значение»). Если собирать запись по payload,
    заказчик увидит менее исправленную копию — с гомоглифами в обозначении
    и посадке, — тогда как исправленная лежит рядом. Поэтому значение
    берётся из поля, а payload остаётся запасным источником для граф,
    которых в полях нет (поле могло не пройти `looks_like_text`).

    Ключ — (каноническое имя графы, рукописная ли): одна и та же графа
    бывает прочитана и печатной в штампе, и рукописной в боковой колонке, и
    это два разных значения.
    """
    out: Dict[tuple, str] = {}
    for field in fields:
        if field.get("category") not in ("title_block", "material"):
            continue
        value = _clean(field.get("value"))
        name, _, text = value.partition(":")
        text = _clean(text)
        if not text or not _clean(name):
            continue
        by_hand = text.endswith(f"({HANDWRITTEN_MARK})")
        if by_hand:
            text = _clean(text[: -len(f"({HANDWRITTEN_MARK})")])
        key = canonical(name) or _fold(name)
        if text and (key, by_hand) not in out:
            out[(key, by_hand)] = text
    return out


def _graphs(block: Any, corrected: Dict[tuple, str],
            by_hand: bool) -> Dict[str, str]:
    """
    Графы штампа с приведёнными именами.

    Разбор отдаёт имя графы то внутренним ключом (`mass`), то так, как её
    назвала модель («масса», «Вес»). Имя приводится к каноническому через
    `title_block.canonical` — тем же справочником, которым его приводит сам
    разбор штампа, чтобы «масса» и `mass` не стали двумя разными графами.
    Непонятое имя остаётся как есть: потерять графу хуже, чем назвать её
    словом модели.
    """
    if not isinstance(block, dict):
        return {}
    out: Dict[str, str] = {}
    for raw_key, raw_value in block.items():
        value = _clean(raw_value)
        if not value:
            continue
        key = canonical(str(raw_key)) or str(raw_key)
        # Исправленная копия того же значения, если она есть.
        value = corrected.get((key, by_hand), value)
        if key in out:
            # Две записи одной графы: вторая дописывается, а не затирает
            # первую. Сравнение точное: значение «2,7» не считается уже
            # сказанным только потому, что входит в «12,75».
            if value not in out[key].split(_GRAPH_JOIN):
                out[key] = f"{out[key]}{_GRAPH_JOIN}{value}"
            continue
        out[key] = value
    return out


def _stamp(payload: Dict[str, Any],
           fields: Sequence[Dict[str, Any]]) -> List[str]:
    """Основная надпись: графа со своим значением, в порядке ГОСТ 2.104."""
    block = _graphs(payload.get("title_block"), _corrected(fields), by_hand=False)
    if not block:
        return []
    said = [
        f"{label(key)} — {block[key]}."
        for key in _STAMP_ORDER if key in block
    ]
    signed = [
        f"{verb} {block[key]}"
        for key, verb in _SIGNERS if key in block
    ]
    # Графы, которых нет в списке выше: разбор штампа вправе принести
    # графу, о которой здесь не сказано, и терять её нельзя.
    known = set(_STAMP_ORDER) | {key for key, _ in _SIGNERS} | set(_HAND_ORDER)
    rest = [
        f"{label(key)} — {block[key]}."
        for key in sorted(block) if key not in known
    ]
    if not (said or signed or rest):
        return []
    body = ["Основная надпись листа."]
    if said:
        body.append(" ".join(said))
    if rest:
        body.append(" ".join(rest))
    if signed:
        body.append("Лист подписали: " + ", ".join(signed) + ".")
    return ["\n".join(body)]


def _handwritten(payload: Dict[str, Any],
                 fields: Sequence[Dict[str, Any]]) -> List[str]:
    """Графы, заполненные от руки: с оговоркой о надёжности чтения."""
    hand = _graphs(payload.get("title_block_handwritten"), _corrected(fields),
                   by_hand=True)
    if not hand:
        return []
    said = [f"{label(key)} — {hand[key]}." for key in _HAND_ORDER if key in hand]
    said += [
        f"{label(key)} — {hand[key]}."
        for key in sorted(hand) if key not in _HAND_ORDER
    ]
    if not said:
        return []
    return ["Записи от руки (прочитаны менее надёжно, чем печатные).\n"
            + " ".join(said)]


def _views(lines: Sequence[str], hedge: bool) -> List[str]:
    """Виды, разрезы и сечения, названные на листе, — их названиями."""
    found: List[str] = []
    for line in lines:
        name = line.strip()
        if not name or len(name) > _VIEW_MAX_CHARS or name in found:
            continue
        if (_VIEW_PAIR.match(name) or _VIEW_SCALED.match(name)
                or _VIEW_WORD.match(name)):
            found.append(name)
    if not found:
        return []
    said = "Виды, разрезы и сечения, названные на листе: " + "; ".join(found) + "."
    return [said + ("\n" + NO_VIEW_LINK if hedge else "")]


def _dimensions(fields: Sequence[Dict[str, Any]], hedge: bool) -> List[str]:
    """Размеры по роду величины: число всегда при том, чем оно является."""
    buckets: Dict[str, List[str]] = {kind: [] for kind, _ in _KIND_TITLES}
    for field in fields:
        kind = _kind_of(field)
        if kind is None:
            continue
        value = _with_tolerance(field)
        if value and value not in buckets[kind]:
            buckets[kind].append(value)
    said = [
        f"{title} — " + "; ".join(buckets[kind]) + "."
        for kind, title in _KIND_TITLES if buckets[kind]
    ]
    if not said:
        return []
    head = "Размеры и обозначения величин на листе, по роду величины."
    if hedge:
        head += " " + NO_VIEW_LINK
    return [head + "\n" + "\n".join(said)]


def _positions(fields: Sequence[Dict[str, Any]]) -> List[str]:
    """Номера позиций: что они означают — в спецификации, не на листе."""
    numbers = _values(fields, "position")
    if not numbers:
        return []
    return ["Номера позиций на листе: " + ", ".join(numbers)
            + ".\nЧто обозначает каждый номер, сказано в спецификации "
              "изделия, а не на этом листе."]


def _callouts(fields: Sequence[Dict[str, Any]]) -> List[str]:
    """Выноски и обозначения (швы, маркировка) — отдельной частью."""
    marks = _values(fields, "callout")
    if not marks:
        return []
    return ["Обозначения и выноски на листе: " + "; ".join(marks) + "."]


def _notes(fields: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Прочий текст листа — как он написан.

    Заголовок нарочно не называет это техническими требованиями, хотя чаще
    всего они тут и лежат. Категория `note` у разбора — это «не
    классифицировано» с уверенностью 0,4 (`vector_drawing.classify`), и в
    неё попадает что угодно: пункты требований, строки спецификации,
    английский перечень изделий, а на листах, которые чертежом не являются
    вовсе, — текст документа СМК целиком. Назвать такое «техническими
    требованиями листа» значило бы утверждать то, чего данные не говорят.
    Нумерованные пункты и так видны тем, что они нумерованные.
    """
    notes = _values(fields, "note")
    if not notes:
        return []
    return ["Текст и примечания листа.\n" + "\n".join(notes)]


def _tables(payload: Dict[str, Any]) -> List[str]:
    """Таблицы листа: значение под именем своего столбца."""
    tables = payload.get("tables")
    if not isinstance(tables, list):
        return []
    out: List[str] = []
    for table in tables:
        if not isinstance(table, dict):
            continue
        rows = [row for row in (table.get("rows") or []) if _row_said(row)]
        if not rows:
            continue
        if _is_stamp_grid(table, rows, payload):
            # Сетка самой основной надписи: её ячейки — подписи граф, а
            # значения этих граф запись уже называет выше, разобранными.
            # Повторять сетку значило бы вернуть в запись то самое
            # перемешивание подписей и чисел, из-за которого она нечитаема.
            continue
        headers = [_clean(cell) for cell in (table.get("headers") or [])]
        title = _clean(table.get("title"))
        if title:
            head = f"Таблица на листе «{title}»."
        elif table.get("kind") == "title_block":
            head = "Таблица в основной надписи листа."
        else:
            head = "Таблица на листе."
        body = [head]
        if headers and any(headers):
            body.append("Столбцы: " + "; ".join(h for h in headers if h) + ".")
        for row in rows:
            body.append(_row_text(row, headers))
        out.append("\n".join(body))
    return out


def _sheet_lines(lines: Sequence[str]) -> List[str]:
    """
    Надписи листа в порядке чтения — прежний текст записи целиком.

    Эта часть и делает сборку беспотерьной: что бы ни собралось выше,
    каждая строка листа по-прежнему в записи есть, и в том же порядке.
    Порядок внутри части не меняется: рядом стоящие «⌀8,5» и «2 отв.»
    остаются рядом, и читающий видит их связь, которую разбор не называет.
    """
    body = [line for line in lines if line.strip()]
    if not body:
        return []
    return ["Надписи листа в порядке чтения.\n" + "\n".join(body)]


# --------------------------------------------------------------- частности

def _without_furniture(lines: Sequence[str]) -> tuple:
    """Строки листа без подписей боковых граф рамки; и сколько убрано."""
    kept: List[str] = []
    dropped = 0
    for line in lines:
        stripped = line.strip()
        if (stripped and not _DIGIT.search(stripped)
                and _fold(stripped) in _FURNITURE_FOLDED):
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped


def _kind_of(field: Dict[str, Any]) -> Optional[str]:
    """
    Род величины поля: по его категории, а при `size` — по самому знаку.

    `None` — поле в перечень родов не попадает. Так уходит голое число:
    категорию `size` разбор ставит уже за то, что надпись похожа на число,
    и в эту категорию попадают номера выносок и зон («3», «1», «7») и
    буквенные обозначения без значения («L», «h», «Dp»). Назвать их
    линейными размерами значило бы сказать неправду о том, что они
    измеряют, — а запись затем и собирается, чтобы значение не стояло без
    того, чем оно является. Такие надписи остаются в части «Надписи листа
    в порядке чтения», где ничего не утверждается.
    """
    category = field.get("category")
    if category == "radius":
        return "radius"
    if category == "thread":
        return "thread"
    if category == "roughness":
        return "roughness"
    if category == "tolerance":
        return "tolerance"
    if category != "size":
        return None
    value = str(field.get("value") or "")
    if _DIAMETER_GLYPHS.search(value):
        return "diameter"
    if _ANGLE_SIGN.search(value):
        return "angle"
    return "linear" if _says_what_it_measures(field, value) else None


def _says_what_it_measures(field: Dict[str, Any], value: str) -> bool:
    """
    Говорит ли надпись сама, что она за величина.

    Достаточно любого из: предельное отклонение в самой надписи или
    отдельной графой, знак умножения размеров («⌀32x2,5», «18x2,5»),
    названная в самой надписи единица. Голая цифра — не говорит.

    Поле `unit` признаком не считается, хотя и выглядит им: разбор
    выводит его из категории, а не читает с листа
    (`vector_drawing.extract_unit` отдаёт «мм» любому числу категории
    `size`). Поэтому единица принимается только тогда, когда она есть в
    самой надписи: «0,5 кг» — названа, «3» с выведенными «мм» — нет.
    """
    if not any(character.isdigit() for character in value):
        return False
    if _clean(field.get("tolerance")):
        return True
    unit = _clean(field.get("unit"))
    if unit and unit in value:
        return True
    return bool(_MEASURED_SIGN.search(value))


def _with_tolerance(field: Dict[str, Any]) -> str:
    """
    Значение поля, а если допуск записан отдельной графой и в само
    значение не вошёл — он назван словом, а не приклеен к числу.
    """
    value = _clean(field.get("value"))
    tolerance = _clean(field.get("tolerance"))
    if not (value and tolerance):
        return value
    # Отклонение уже внутри значения — второй раз не называется. Но голая
    # цифра совпадает с чем угодно: допуск «2» входит в «⌀200», и отклонение
    # пропало бы молча. Поглощённым считается только отклонение со знаком.
    if tolerance in value and not tolerance.isdigit():
        return value
    return f"{value} (допуск {tolerance})"


def _values(fields: Sequence[Dict[str, Any]], category: str) -> List[str]:
    """Значения полей одной категории: по порядку, без точных повторов."""
    out: List[str] = []
    for field in fields:
        if field.get("category") != category:
            continue
        value = _with_tolerance(field)
        if value and value not in out:
            out.append(value)
    return out


def _is_stamp_grid(
    table: Dict[str, Any], rows: Sequence[Any], payload: Dict[str, Any],
) -> bool:
    """
    Это сетка основной надписи, а не таблица со сведениями.

    Признак — из самой таблицы: не меньше трети её заполненных ячеек это
    подписи граф ГОСТ 2.104 («Изм.», «Лист», «N докум.», «Разраб.») или
    подписи боковых граф. Треть, а не половина: в сетке штампа подписи и
    значения стоят попеременно, и половины подписи не набирают. Таблица
    «Комплектность» с тем же `kind`, но с наименованиями и количествами,
    этой проверки не проходит и остаётся.

    Проверка применяется только когда графы штампа разобраны: если их нет,
    сетка — единственное, что о штампе известно, и она нужна.
    """
    if table.get("kind") != _TITLE_BLOCK_KIND:
        return False
    if not (payload.get("title_block") or payload.get("title_block_handwritten")):
        return False
    cells = [_fold(cell) for row in rows for cell in row if _clean(cell)]
    if not cells:
        return False
    labels = sum(1 for cell in cells if cell in _LABELS_FOLDED)
    return labels >= _STAMP_GRID_MIN_LABELS and labels * 3 >= len(cells)


def _row_said(row: Any) -> bool:
    return bool(isinstance(row, (list, tuple)) and any(_clean(c) for c in row))


def _row_text(row: Iterable[Any], headers: Sequence[str]) -> str:
    """Строка таблицы: каждое значение при имени своего столбца."""
    cells = list(row)
    said: List[str] = []
    for index, cell in enumerate(cells):
        value = _clean(cell)
        if not value:
            continue
        header = headers[index] if index < len(headers) else ""
        said.append(f"{header} — {value}" if header else value)
    return "; ".join(said) + "."


def _clean(value: Any) -> str:
    """Значение как строка, без обрамляющих пробелов. Ничего не переписывает."""
    if value is None:
        return ""
    return str(value).strip()
