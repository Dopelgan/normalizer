"""
Род значения на чертеже: что это за шероховатость и к какому изображению
относится размер.

Зачем. Поле чертежа несёт категорию («roughness», «size») и само значение,
но не то, что чертёжник читает из оформления листа: «Ra 6,3» в правом
верхнем углу — шероховатость всех поверхностей, не обозначенных на видах
(ГОСТ 2.309), а «35°» под заголовком «Б (1:2)» — угол на выносном элементе
Б. Без этого потребитель вынужден угадывать, и угадывает хуже, чем молчит:
у «Крышки» структурным полем приходила только местная «Ra 1,6», а общая
«Ra 6,3» — голой строкой текста листа.

Правило модуля одно: неверная подпись хуже отсутствующей. Подпись ставится
только там, где её несёт оформление листа и где это подтверждено двумя
независимыми прочтениями. Во всех остальных случаях поле остаётся без
подписи, а причина отказа пишется в `structured_payload` — отсутствие
подписи должно быть видно, а не выглядеть как «не искали».

Что здесь есть.

* `general_roughness` (`DRAWING_CORNER_ROUGHNESS`) — знак шероховатости в
  правом верхнем углу поля чертежа.
* `chamfer` (`DRAWING_VALUE_KIND`) — фаска в записи «2×45°» (ГОСТ 2.307).
* `view_membership` (`DRAWING_VALUE_KIND`) — размер на изображении с
  заголовком («Б (1:2)», «В–В», «Вид А»).

Чего здесь нет и почему. «Высота» и «длина» по направлению размерной линии
не ставятся: вертикальный размер на главном виде — высота, а на виде сверху
тот же вертикальный размер — ширина; лист этого не называет. «Наибольший
размер — общая длина» — тоже догадка: на сборочном листе наибольшим бывает
справочный или монтажный размер.

Модуль чистый: модель зовёт `core.drawing_processor`, сюда приходят уже
готовые прочтения. Связи: `core.providers.qwen_client`
(`read_corner_roughness`, `read_view_values`), `core.providers.drawing_vision`
(`roughness_corner`), контракт — `DrawingField.kind`, `kind_label`, `view`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.providers import drawing_text

# Род значения — закрытый список: потребитель сравнивает строки.
KIND_ROUGHNESS_UNSPECIFIED = "roughness_unspecified_surfaces"
KIND_CHAMFER = "chamfer"

# Подпись рода по-русски — для потребителя: её можно показать перед значением.
KIND_LABELS = {
    KIND_ROUGHNESS_UNSPECIFIED: "Шероховатость неуказанных поверхностей",
    KIND_CHAMFER: "Фаска",
}

# Причины отказа ставить подпись (пишутся в structured_payload).
REFUSED_NO_FRAME = "no_frame"                    # рамка не найдена: где угол, неизвестно
REFUSED_NOT_READ = "corner_not_read"             # модель не ответила на вырезку угла
REFUSED_NO_SIGN = "no_sign_in_corner"            # в углу знака нет
REFUSED_NO_VALUE = "value_unreadable"            # знак есть, значение не разобрано
REFUSED_DISAGREE = "readings_disagree"           # два прочтения угла дали разные значения
REFUSED_NOT_CONFIRMED = "not_confirmed"          # второго прочтения того же значения нет
REFUSED_VIEW_IN_CORNER = "view_in_corner"        # в углу стоят размеры: там изображение
REFUSED_UNKNOWN_TITLE = "unknown_title"          # заголовка нет среди прочитанных на листе
REFUSED_UNKNOWN_VALUE = "unknown_value"          # такого размера среди полей нет
REFUSED_AMBIGUOUS = "ambiguous_value"            # одинаковых размеров на листе несколько
REFUSED_NO_POSITION = "no_position"              # места надписи или заголовка нет
REFUSED_POSITION = "position_disagrees"          # место говорит о другом изображении

_NUMBER = r"\d+(?:[.,]\d+)?"
# Знак шероховатости без текста вокруг: «Ra 6,3», «√ Ra 6,3 (√)», «Rz 40()».
# Скобки после значения — знак в скобках по ГОСТ 2.309: распознавание сам
# значок теряет, а скобки оставляет («Ra 6,3()»).
_SIGN = r"[√✓∇vV\\/]"
_ROUGHNESS_SIGN = re.compile(
    r"^\s*" + _SIGN + r"{0,2}\s*(R[az])\s?(" + _NUMBER + r")\s*(\(\s*" + _SIGN + r"{0,2}\s*\))?\s*$"
)
# Фаска под 45°: катет, знак умножения, 45°. Другие углы так не пишут
# (ГОСТ 2.307: фаска не под 45° задаётся линейным и угловым размерами).
_CHAMFER = re.compile(
    r"^\s*(?:(?:\d+\s*)?фаск[аи]\w*\s*)?(" + _NUMBER + r")\s?[xх×]\s?45\s?°"
    r"(?:\s*,?\s*\d+\s*фаск[аи]\w*)?\s*$",
    re.IGNORECASE,
)
# Заголовок изображения по ГОСТ 2.305 и 2.316: выносной элемент или вид с
# масштабом («Б (1:2)»), разрез или сечение («В–В», «А-А (2:1)»), «Вид А».
# Одна буква заголовком не считается: в прочтении она неотличима от буквы
# у стрелки взгляда и у кружка выносного элемента.
_LETTER = r"[А-ЯЁA-Z]\d?"
_SCALE = r"\(\s*\d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?\s*\)"
_VIEW_TITLE = re.compile(
    r"^\s*(?:(" + _LETTER + r")\s*" + _SCALE
    + r"|(" + _LETTER + r")\s*[-–—]\s*\2(?:\s*" + _SCALE + r")?"
    + r"|Вид\s+(" + _LETTER + r"))(?:\s+\d+\s+мест\w*)?\s*$"
)
# Одна буква без цифры: «А» у вида по стрелке. «R1» — радиус, а не буква.
_BARE_LETTER = re.compile(r"^[А-ЯЁA-Z]$")
# Размеры и резьбы: надпись такой категории стоит на изображении.
_LINEAR_CATEGORIES = ("size", "radius", "thread", "tolerance")
_PLACES = re.compile(r"\s+\d+\s+мест\w*$")
# Категории полей, у которых бывает «свой вид».
_DIMENSION_CATEGORIES = ("size", "radius", "thread", "tolerance", "roughness")

_UNKNOWN_BBOX = [0.0, 0.0, 1.0, 1.0]
# Насколько место надписи от модели может выходить за угол: её координаты
# приблизительны, а сам угол мал.
_CORNER_SLACK = 0.03
# Как далеко от своего заголовка может стоять размер, доля стороны листа.
# Правило «ближайший заголовок не ниже надписи» само по себе относит к
# выносному элементу и размеры главного вида, стоящие ниже и в стороне: на
# листе «Расширителя» так набиралось 8 чужих размеров из 12. Дальше 0,2
# стороны не лежал ни один свой размер выносного элемента и ни один чужой
# не лежал ближе. Цена — крупное изображение: его дальние размеры остаются
# без подписи.
_TITLE_REACH = 0.2


# ===========================================================================
# Шероховатость неуказанных поверхностей
# ===========================================================================

def roughness_sign(text: Any) -> Optional[Tuple[str, bool]]:
    """
    Надпись — один знак шероховатости: значение («Ra 6,3») и признак знака в
    скобках после него. `None` — это не знак, а текст, в котором
    шероховатость упомянута («шерох. не хуже Ra 1,6»).
    """
    match = _ROUGHNESS_SIGN.match(drawing_text.normalize(str(text or "")))
    if not match:
        return None
    value = f"{match.group(1)[0].upper()}{match.group(1)[1].lower()} {match.group(2).replace('.', ',')}"
    return value, bool(match.group(3))


def general_roughness(
    fields: Sequence[Dict[str, Any]],
    transcript: Optional[Sequence[str]],
    corner: Optional[Sequence[float]],
    reading: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Решение по знаку в правом верхнем углу поля чертежа.

    Подпись «шероховатость неуказанных поверхностей» требует двух вещей.
    Место: знак прочитан в самом углу — на вырезке угла, которую нашла
    геометрия (`reading`), или полем листа, место которого модель указала в
    углу. Подтверждение: то же значение отдельным знаком есть в другом,
    независимом прочтении листа — строкой переписанного текста, полем листа
    или вырезкой угла. Одного места мало: в углу может оказаться вид со
    своим знаком; одного значения мало: «Ra 1,6» строкой текста — это и
    местный знак тоже. Отсюда два запрета. Если в углу стоит хоть один
    размер, там изображение, и знак в нём может быть местным — подписи нет.
    Если то же значение стоит полем в другом месте листа, строка текста
    может быть его прочтением и подтверждением не считается.

    Знак в скобках после значения записывается, когда прочитан, но не
    требуется: модель его теряет («Ra 6,3» у листа, где стоит «Ra 6,3 (√)»).
    На род это не влияет: без скобок знак в углу относится ко всем
    поверхностям, со скобками — ко всем, кроме обозначенных на видах, и
    «поверхности, не обозначенные на изображениях» верно в обоих случаях.

    Возвращает `{"value", "bracket_sign", "evidence"}` либо `{"refused"}`.
    """
    if not corner:
        return {"refused": REFUSED_NO_FRAME}

    in_corner: List[Tuple[str, bool]] = []
    confirming: List[str] = []
    local: List[str] = []
    for field in fields:
        bbox = field.get("bbox")
        sign = roughness_sign(field.get("value"))
        if sign is None:
            if (field.get("category") in _LINEAR_CATEGORIES and _known(bbox)
                    and _inside(bbox, corner)):
                return {"refused": REFUSED_VIEW_IN_CORNER}
            continue
        if not _known(bbox):
            confirming.append(sign[0])       # место неизвестно: только значение
        elif _inside(bbox, corner, _CORNER_SLACK):
            in_corner.append(sign)
        else:
            local.append(sign[0])            # местный знак: ничего не подтверждает

    bracket: Optional[bool] = None
    from_crop: Optional[str] = None
    if reading is not None:
        if not reading.get("sign"):
            # Вырезка угла пуста, а поле листа «стоит в углу»: одно из двух
            # прочтений ошибается, и выбрать между ними нечем.
            return {"refused": REFUSED_DISAGREE if in_corner else REFUSED_NO_SIGN}
        sign = roughness_sign(reading.get("value"))
        if sign is None:
            return {"refused": REFUSED_NO_VALUE}
        from_crop = sign[0]
        bracket = bool(reading.get("bracket_sign")) or sign[1]
    elif not in_corner:
        # Места знака нет ни у одного прочтения: вырезку угла не спросили
        # (нет времени) или модель на неё не ответила.
        return {"refused": REFUSED_NOT_READ}

    values = {value for value, _ in in_corner} | ({from_crop} if from_crop else set())
    if len(values) > 1:
        return {"refused": REFUSED_DISAGREE, "values": sorted(values)}
    value = next(iter(values))
    if any(has_bracket for _, has_bracket in in_corner):
        bracket = True

    evidence: List[str] = []
    if from_crop:
        evidence.append("corner_crop")
    if in_corner:
        evidence.append("sheet_field_in_corner")
    # То же значение у местного знака: строка текста и поле без места могут
    # быть его прочтениями, а не знака в углу.
    if value not in local:
        for sign in (roughness_sign(line) for line in transcript or []):
            if sign is not None and sign[0] == value:
                evidence.append("transcript_line")
                if sign[1]:
                    bracket = True
                break
        if from_crop and not in_corner and value in confirming:
            evidence.append("sheet_field")
    if len(evidence) < 2:
        return {"refused": REFUSED_NOT_CONFIRMED, "value": value}
    return {"value": value, "bracket_sign": bracket, "evidence": evidence}


def needs_corner_reading(fields: Sequence[Dict[str, Any]], corner: Sequence[float]) -> bool:
    """Нужна ли вырезка угла: поля листа со знаком в самом углу ещё нет."""
    return not any(
        roughness_sign(field.get("value")) and _known(field.get("bbox"))
        and _inside(field["bbox"], corner, _CORNER_SLACK)
        for field in fields
    )


def mark_general_roughness(
    fields: List[Dict[str, Any]], value: str, corner: Sequence[float],
) -> bool:
    """
    Ставит род на поле листа с этим значением. `False` — такого поля нет:
    значение пришло только с вырезки угла и из текста листа, и поле создаёт
    вызывающий. Прежние поля не меняются, кроме добавленных ключей `kind`
    и `kind_label`.
    """
    best: Optional[Dict[str, Any]] = None
    for field in fields:
        sign = roughness_sign(field.get("value"))
        if sign is None or sign[0] != value:
            continue
        bbox = field.get("bbox")
        if _known(bbox) and not _inside(bbox, corner, _CORNER_SLACK):
            continue                          # такое же значение у местного знака
        if best is None or _known(bbox):
            best = field
    if best is None:
        return False
    _set_kind(best, KIND_ROUGHNESS_UNSPECIFIED)
    return True


# ===========================================================================
# Фаска и изображение, на котором стоит размер
# ===========================================================================

def chamfer(text: Any) -> bool:
    """Надпись — фаска под 45° в записи ГОСТ 2.307: «2×45°», «1,6x45°»."""
    return bool(_CHAMFER.match(drawing_text.normalize(str(text or ""))))


def mark_chamfers(fields: List[Dict[str, Any]]) -> int:
    """Род «фаска» у полей в записи «катет×45°». Возвращает число подписей."""
    marked = 0
    for field in fields:
        if not field.get("kind") and chamfer(field.get("value")):
            _set_kind(field, KIND_CHAMFER)
            marked += 1
    return marked


def view_title(text: Any) -> Optional[str]:
    """Заголовок изображения в едином написании («Б (1:2)», «В–В») или None."""
    raw = " ".join(str(text or "").split())
    if not _VIEW_TITLE.match(raw):
        return None
    # «Б (4:1) 4 места» — заголовок и число мест одной надписью.
    return _title_key(_PLACES.sub("", raw))


def view_titles(
    fields: Sequence[Dict[str, Any]], transcript: Optional[Sequence[str]],
) -> Dict[str, Dict[str, Any]]:
    """
    Заголовки изображений листа: ключ заголовка -> поле, которым он прочитан.

    Заголовок принимается, только если его одинаково прочитали оба прохода —
    чтение листа (поле) и переписанный текст (строка). У «Крышки» знак
    «развёрнуто» перед масштабом один проход прочитал как «B (1:2,5)», другой
    как «Q (1:2,5)»: такой буквы на листе нет, и подписывать ею размеры
    нельзя. Без переписанного текста заголовков нет вовсе.
    """
    lines = {view_title(line) for line in transcript or []} - {None}
    titles: Dict[str, Dict[str, Any]] = {}
    for field in fields:
        key = view_title(field.get("value"))
        if key is None or key not in lines:
            continue
        current = titles.get(key)
        if current is None or (not _known(current.get("bbox")) and _known(field.get("bbox"))):
            titles[key] = field
    return titles


def title_text(field: Dict[str, Any]) -> str:
    """Заголовок как на листе, без числа мест: «Б (4:1)»."""
    return _PLACES.sub("", " ".join(str(field.get("value") or "").split()))


def view_membership(
    fields: List[Dict[str, Any]],
    titles: Dict[str, Dict[str, Any]],
    answer: Optional[Dict[str, Any]],
    transcript: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Ставит полю `view` — заголовок изображения, на котором стоит размер.

    Модель отвечает, какие размеры стоят на изображении с каким заголовком
    (`answer` — `{"views": [{"title", "values"}]}`), но её слово принимается
    только при трёх условиях. Заголовок — из `titles` (`view_titles`). Размер с
    таким значением на листе один: у двух «15±0,1» не узнать, о котором
    речь; одинаковые поля к этому моменту уже сведены в одно, поэтому
    повтор считается и по строкам переписанного текста. Место надписи с
    ответом согласно: заголовок стоит над своим изображением (ГОСТ 2.305),
    поэтому из заголовков, которые не ниже надписи, ближайшим должен быть
    названный, и не дальше `_TITLE_REACH`. Соперником считается любая
    надпись, похожая на заголовок, — и прочитанная лишь одним проходом, и
    одна буква: подписать можно только подтверждённым заголовком, а
    заслонить его может и неподтверждённый. Поле, у которого уже есть род
    (знак в углу листа), ни к какому изображению не относится.
    Места нет — подписи нет. Проверка местом — запрет, а не доказательство:
    она отсекает явно чужое, но размер соседнего вида, стоящий вплотную под
    заголовком, пропустит, если модель назвала его.

    Возвращает `{"labelled": n, "refused": [{"value", "title", "reason"}]}`.
    """
    result: Dict[str, Any] = {"labelled": 0, "refused": []}
    views = (answer or {}).get("views")
    if not isinstance(views, list):
        return result

    by_value: Dict[str, List[Dict[str, Any]]] = {}
    for field in fields:
        if field.get("category") in _DIMENSION_CATEGORIES and not field.get("kind"):
            by_value.setdefault(_value_key(field.get("value")), []).append(field)
    written: Dict[str, int] = {}
    for line in transcript or []:
        written[_value_key(line)] = written.get(_value_key(line), 0) + 1
    rivals = _title_places(fields)

    claims: Dict[str, List[str]] = {}
    as_written: Dict[str, str] = {}
    for view in views:
        if not isinstance(view, dict) or not isinstance(view.get("values"), list):
            continue
        title = view_title(view.get("title"))
        for value in view["values"]:
            key = _value_key(value)
            as_written.setdefault(key, str(value))
            claims.setdefault(key, []).append(title or str(view.get("title") or ""))

    for key, claimed in claims.items():
        title = claimed[0]

        def refuse(reason: str) -> None:
            result["refused"].append(
                {"value": as_written[key], "title": title, "reason": reason})

        if len(set(claimed)) > 1 or len(by_value.get(key, [])) > 1 or written.get(key, 0) > 1:
            refuse(REFUSED_AMBIGUOUS)
        elif title not in titles:
            refuse(REFUSED_UNKNOWN_TITLE)
        elif not by_value.get(key):
            refuse(REFUSED_UNKNOWN_VALUE)
        else:
            field = by_value[key][0]
            if not _known(field.get("bbox")) or not _known(titles[title].get("bbox")):
                refuse(REFUSED_NO_POSITION)
            elif _title_above(field["bbox"], rivals) != title:
                refuse(REFUSED_POSITION)
            else:
                field["view"] = title_text(titles[title])
                result["labelled"] += 1
    return result


# ===========================================================================
# Помощники
# ===========================================================================

def _set_kind(field: Dict[str, Any], kind: str) -> None:
    field["kind"] = kind
    field["kind_label"] = KIND_LABELS[kind]


def _known(bbox: Any) -> bool:
    """Место надписи известно: модель не вернула «весь лист» вместо координат."""
    return (isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and [float(c) for c in bbox] != _UNKNOWN_BBOX)


def _centre(bbox: Sequence[float]) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2


def _inside(bbox: Sequence[float], zone: Sequence[float], slack: float = 0.0) -> bool:
    x, y = _centre(bbox)
    return (float(zone[0]) - slack <= x <= float(zone[2]) + slack
            and float(zone[1]) - slack <= y <= float(zone[3]) + slack)


def _title_key(text: str) -> str:
    """«Б(1:2)», «Б (1 : 2)» и «б (1:2)» — один заголовок; «A» и «А» — одна буква."""
    folded = "".join(text.split()).upper().replace("—", "–").replace("-", "–")
    return folded.translate(_LATIN_TO_CYRILLIC).replace(".", ",")


_LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTX", "АВСЕНКМОРТХ")


def _value_key(text: Any) -> str:
    """Размер без пробелов и с одной записью дроби: «⌀ 32» и «⌀32» — одно значение."""
    return "".join(drawing_text.normalize(str(text or "")).split()).replace(".", ",").casefold()


def _title_places(fields: Sequence[Dict[str, Any]]) -> List[Tuple[str, Sequence[float]]]:
    """
    Все надписи листа, похожие на заголовок изображения, с известным местом:
    заголовки по ГОСТ и одиночные буквы (заголовок вида по стрелке).
    """
    places: List[Tuple[str, Sequence[float]]] = []
    for field in fields:
        raw = " ".join(str(field.get("value") or "").split())
        key = view_title(raw) or (f"буква {raw}" if _BARE_LETTER.match(raw) else None)
        if key and _known(field.get("bbox")):
            places.append((key, field["bbox"]))
    return places


def _title_above(bbox: Any, places: Sequence[Tuple[str, Sequence[float]]]) -> Optional[str]:
    """
    Ближайший заголовок из тех, что не ниже надписи и не дальше
    `_TITLE_REACH`. `None` — такого заголовка нет.
    """
    x, y = _centre(bbox)
    best: Optional[Tuple[float, str]] = None
    for key, place in places:
        if float(place[1]) > y:
            continue
        title_x, title_y = _centre(place)
        distance = ((title_x - x) ** 2 + (title_y - y) ** 2) ** 0.5
        if distance > _TITLE_REACH:
            continue
        if best is None or distance < best[0]:
            best = (distance, key)
    return best[1] if best else None
