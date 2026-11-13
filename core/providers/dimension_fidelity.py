"""
Числовая достоверность размеров и знаков на чертеже
(`DRAWING_DIMENSION_FIDELITY`).

Проверка показала шесть видов потери именно в числах и знаках — там, где
ошибка меняет смысл надписи, а не её вид:

* запятая десятичной дроби пропадает у шага резьбы («M28x1,5» → «M28x15»),
  у шероховатости («Ra 1,6» → «Ра 16») и у мелкого предельного отклонения
  («2⁺¹˙⁵» → «2+15»). Значение становится в десять раз больше, и по верной
  записи надпись не находится;
* надстрочные цифры отклонения приходят от модели как есть
  («⌀16⁺⁰˙⁰¹⁸»), и дальше по конвейеру это не число;
* знак «±» у транскрипции листа подменяется на «+» («30±10*» → «30+10*»),
  а у вырезки вида он верный: в индекс попадали оба варианта;
* справочная звёздочка по ГОСТ 2.307 («(Y+12)*») теряется у одного из двух
  прочтений той же надписи;
* марка стали («12Х18Н10Т») разбирается как размер с посадкой: правило
  посадки видит в ней «H10» и выдаёт допуск, которого на листе нет;
* масштаб «1:1» читается как «7:1» — курсивная «1» по ГОСТ похожа на «7».

Что здесь делается и чего здесь не делается. Запятая не «додумывается»:
она восстанавливается только тогда, когда записанное без неё значение по
ГОСТ невозможно, а с запятой — ровно одно допустимое. Шаг резьбы сверяется
с рядом ГОСТ 8724 и с диаметром (длина крепежа «М12х50» остаётся длиной),
шероховатость Ra — с предпочтительным рядом ГОСТ 2789 (ряд Rz другой, и Rz
здесь не трогается), масштаб — с рядом ГОСТ 2.302 и только в графе
«Масштаб» и в заголовке вида, отклонение — с самим размером (отклонение не
бывает больше размера; диапазон «4−16» и цепочки номеров не в счёт). Где
однозначного прочтения нет, строка остаётся как есть: выдуманное число хуже
честно нечитаемого.

Отдельно — выбор между двумя прочтениями одной надписи (`prefer`): из
«30+10» и «30±10» остаётся то, в котором знак и звёздочка сохранены.
Вариант с «+» вместо «±» сам по себе не исчезает — он исчезает только
рядом с тем же размером, прочитанным со знаком.

Флаг выключен по умолчанию; каждая правка пишется в лог.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------- надстрочные
# Надстрочные цифры и знаки, которыми модель передаёт мелкое отклонение
# справа от размера: «2⁺¹˙⁵», «⌀16⁺⁰˙⁰¹⁸».
_SUPERSCRIPT: Dict[str, str] = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "−",
}
# Разделитель дроби внутри надстрочной записи: точка в середине строки.
_SUPERSCRIPT_POINT = "·˙‧∙"
# Раскрывается только запись отклонения: справа от числа или скобки, со
# знаком «⁺» или «⁻» перед каждой группой цифр. Иначе под правило попадали
# бы «10 м²», «x²» и одиночная точка списка «• Примечание».
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_RUN = re.compile(
    r"(?<=[\d)])((?:[⁺⁻][" + _SUPERSCRIPT_DIGITS + "][" + _SUPERSCRIPT_DIGITS
    + _SUPERSCRIPT_POINT + "]*)+)"
)

# --------------------------------------------------------------- резьба
# Ряд шагов метрической резьбы по ГОСТ 8724 (мм).
THREAD_PITCHES: Tuple[str, ...] = (
    "0,2", "0,25", "0,3", "0,35", "0,4", "0,45", "0,5", "0,6", "0,7", "0,75",
    "0,8", "1", "1,25", "1,5", "1,75", "2", "2,5", "3", "3,5", "4", "4,5",
    "5", "5,5", "6", "8",
)
# «M28x15», «M24×15»; знак умножения — и латинский «x», и «×».
_THREAD = re.compile(r"(?<![\w,.])(M\s?(\d{1,3}))(\s?[xX×Хх]\s?)(\d{1,4})(?![\d,.])")
# Шаг резьбы не бывает больше четверти диаметра: у самой крупной резьбы
# М6х1 это 1/6. Без такой проверки длина крепежа («Болт М12х50») читалась
# бы как потерянная запятая шага («M12x5»).
_MAX_PITCH_SHARE = 0.25
# Длины крепёжных изделий по ГОСТ 7798, 17475 и соседним: «М12х50» — это
# болт длиной 50 мм, а не резьба с шагом 5 мм.
FASTENER_LENGTHS: FrozenSet[int] = frozenset({
    6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50,
    55, 60, 65, 70, 75, 80, 85, 90, 100, 110, 120, 130, 140, 150, 160, 170,
    180, 190, 200, 220, 240, 260, 280, 300,
})

# --------------------------------------------------------------- шероховатость
# Предпочтительные значения Ra по ГОСТ 2789 (мкм). Ряд шире, но запятую
# восстанавливаем только до предпочтительного значения: иначе «Ra 20» (есть
# в полном ряду) превращалось бы в «Ra 2».
ROUGHNESS_PREFERRED: Tuple[str, ...] = (
    "0,012", "0,025", "0,05", "0,1", "0,2", "0,4", "0,8",
    "1,6", "3,2", "6,3", "12,5", "25", "50", "100",
)
# Начало слова. Кроме обычного («не буква и не цифра слева») годится и
# заглавная буква сразу после строчной: распознавание склеивает слова, и
# «чемРа 16» — это «чем Ra 1,6». Цифра слева запрещена: иначе в
# шестнадцатеричном «unit-…94ba04» находился бы «ba0».
_WORD_START = r"(?:(?<![А-Яа-яЁёA-Za-z\d])|(?<=[а-яёa-z])(?=[A-ZА-ЯЁ]))"

# Запятая восстанавливается только у Ra. Ряд Rz по ГОСТ 2789 другой, и
# «Rz 16», «Rz 32», «Rz 63», «Rz 125» — настоящие его значения.
_ROUGHNESS = re.compile(_WORD_START + r"(Ra)(\s?)(\d{2,4})(?![\d,.])")
# Цифра слева запрещена наровне с буквой: без этого в шестнадцатичном
# идентификаторе «unit-…94ba04» находился «ba0» и становился «Ra 0,4».
# Знак шероховатости кириллицей и похожими буквами: «Ра 16», «Ка 3.2»,
# «Rа 1,6». Один образец на весь разбор: его зовут и `drawing_text.normalize`
# для коротких надписей модели, и правка абзацев MinerU и строк транскрипции,
# которые нормализовать целиком нельзя.
# Вторая буква — только строчная: «КЗ 31 кА» (ток короткого замыкания),
# «КА 1» (реле по ГОСТ 2.710), «ВА 47-29» (серия выключателя) и «РЗ 2» —
# не шероховатость, а обозначения, которых на электрических листах полно.
_ROUGHNESS_LETTERS_PLAIN = re.compile(
    r"(?<![А-Яа-яЁёA-Za-z])([RКВВK])\s?([aаzз])(?=\s?\d)"
)
ROUGHNESS_LETTERS = re.compile(
    _WORD_START + r"([RrРрВвКкKk])\s?([aаzз])(?=\s?\d)"
)

# --------------------------------------------------------------- отклонение
# Размер со слипшимся отклонением без запятой: «2+15», «3,5+15», «40−025».
# Знак — только «+», «±» и настоящий минус «−»: обычный дефис между двумя
# числами на чертеже почти всегда диапазон («4-16» — диаметр кабеля), и
# трогать его нельзя.
# Отрицательный знак отдельно: «4−16» — диапазон диаметра кабеля, и
# «больше размера» для минуса не признак потерянной запятой. У минуса
# правится только запись с ведущим нулём («40−025»). Взгляд вперёд
# `(?![−\-]\d)` отсекает цепочки обозначений («3185−001−12345678»).
_DEVIATION = re.compile(
    r"(?<![\w,.])(\d{1,4}(?:[.,]\d{1,3})?)([+−±])(\d{2,4})(?![\d,.])(?![−\-]\d)"
)

# --------------------------------------------------------------- марка стали
# «12Х18Н10Т», «08Х18Н10Т», «04Х19Н11М3» и те же марки, прочитанные
# латиницей («12X18H10T»). Внутри такой марки правило посадки видит «H10»
# и выдаёт допуск, которого на чертеже нет.
# Образец узкий намеренно: марка кончается буквой легирующего элемента
# («…Н10Т», «…Н11М3») или начинается с «Св-». Без этого под правило попадал
# бы размер паза с посадкой — «18х11Н9», «40x5h11», — и он терял бы допуск.
_GRADE_BODY = r"(?<![\w-])(?:Св-)?\d{2}[ХXхx]\d{1,2}[НHнh]\d{1,2}"
GRADE = re.compile(_GRADE_BODY + r"(?:[А-Яа-яA-Za-z]\d?)+(?![\w])")
# Марка без хвоста («08Х18Н10», «12Х18Н9») по виду не отличается от размера
# паза с посадкой («12х18Н9»), поэтому считается маркой только там, где
# рядом сказано, что речь о материале. Марка с хвостом в этом образце стоит
# первой: иначе от «08Х18Н10Т» закрывалось бы только «08Х18Н10».
_GRADE_IN_MATERIAL = re.compile(
    _GRADE_BODY + r"(?:[А-Яа-яA-Za-z]\d?)+(?![\w])"
    r"|" + _GRADE_BODY + r"(?![\w])"
)
_MATERIAL_CONTEXT = re.compile(
    r"сталь|материал|марка|прокат|лист\s|Св-|ГОСТ\s?5632|ГОСТ\s?2246"
    r"|ГОСТ\s?5582|ГОСТ\s?19903|ГОСТ\s?19904", re.IGNORECASE
)

# ------------------------------------------------------- обозначение шва
# Обозначение сварного шва по ГОСТ 2.312 и ГОСТ 14771: тип соединения
# (С стыковое, У угловое, Т тавровое, Н нахлёсточное), номер и способ
# сварки. Способов с инертным газом два: «ИН» — неплавящимся электродом,
# «ИНп» — плавящимся. «ИНн» не бывает, и на прод-разборе штанги
# «Н1-ИНп» пришло как «H1-ИНн»: латинская «H» вместо «Н» и «н» вместо «п».
#
# Первая буква способа — только «И»: латинского двойника у неё нет, и если
# разрешить здесь «H», под правило попадут настоящие обозначения с «НН» —
# «Т1-НН» (сторона низшего напряжения трансформатора по ГОСТ 2.710),
# прочитанное латиницей как «T1-HH», превращалось в «Т1-ИН».
_WELD_TYPES = {"С": "С", "C": "С", "У": "У", "Y": "У", "Т": "Т", "T": "Т",
               "Н": "Н", "H": "Н"}
WELD_METHODS: Tuple[str, ...] = ("ИН", "ИНп")
# Буквы, которые по ГОСТ 2.304 путаются с «п» в конце способа сварки:
# только они и читаются как «п». Любая другая буква означает, что это не
# обозначение шва, и запись остаётся как на листе.
_WELD_TAIL_CONFUSION = frozenset("нНиИпП")
_WELD = re.compile(
    r"(?<![\wА-Яа-яЁё])(\d{0,2})([СCУYТTНH])(\d{1,2})-([Ии][НHн])([а-яa-zА-ЯA-Z]?)"
    r"(?![\wА-Яа-яЁё])"
)

# --------------------------------------------------------------- масштаб
# Ряд масштабов по ГОСТ 2.302.
SCALES: Tuple[str, ...] = (
    "1:1",
    "1:2", "1:2,5", "1:4", "1:5", "1:10", "1:15", "1:20", "1:25", "1:40",
    "1:50", "1:75", "1:100", "1:200", "1:400", "1:500", "1:800", "1:1000",
    "2:1", "2,5:1", "4:1", "5:1", "10:1", "20:1", "40:1", "50:1", "100:1",
)
_RATIO = r"(\d{1,4}(?:,\d)?)\s?:\s?(\d{1,4}(?:,\d)?)"
# Масштаб правится только там, где он и стоит: значение графы «Масштаб»
# (`repair_scale`) и заголовок вида в скобках — «Б(2:1)», «А (1:2)».
# Без этой привязки под правило попадали бы «Поз. 7:1» и «лист 7:15».
_SCALE = re.compile(r"(?<![\d:,])" + _RATIO + r"(?![\d:,])")
# Перед скобкой — буква вида («Б(2:1)», «А (1:2)»). Одной скобки мало:
# «(7:15)» бывает временем, «(2:7)» — просто отношением.
_VIEW_SCALE = re.compile(
    r"(?:(?<=[А-ЯЁA-Z]\()|(?<=[А-ЯЁA-Z] \())" + _RATIO + r"(?=\))"
)
# Курсивная «1» по ГОСТ 2.304 читается как «7»; обратной замены не бывает.
_SCALE_CONFUSION = {"7": "1"}


def _decimal(value: str) -> str:
    """«1.5» -> «1,5»; «2,0» -> «2». Запись для сверки с рядом по ГОСТ."""
    text = value.replace(".", ",")
    if "," in text:
        text = text.rstrip("0").rstrip(",")
    return text or "0"


def _comma_variants(digits: str) -> List[str]:
    """Прочтения числа без разделителя с запятой на каждом месте."""
    return [
        _decimal(digits[:index] + "," + digits[index:])
        for index in range(1, len(digits))
    ]


def _leading_zero(value: str) -> bool:
    """«02,5» — не запись числа: целая часть с ведущим нулём."""
    whole = value.split(",")[0]
    return len(whole) > 1 and whole.startswith("0")


def _only_valid(digits: str, allowed: Sequence[str]) -> Optional[str]:
    """
    Единственное прочтение числа с запятой, которое есть в ряду по ГОСТ.
    `None`, если само число в ряду есть (правка не нужна), прочтений нет
    или их больше одного.
    """
    if _decimal(digits) in allowed:
        return None
    found = {value for value in _comma_variants(digits) if value in allowed}
    if len(found) != 1:
        return None
    return found.pop()


# =========================================================================
# Правки одной надписи
# =========================================================================

def expand_superscripts(text: str) -> str:
    """
    «2⁺¹˙⁵» -> «2+1,5». Точка в середине строки внутри надстрочной записи —
    это запятая дроби, а не знак умножения.
    """
    def replace(match: re.Match) -> str:
        out = []
        for character in match.group():
            if character in _SUPERSCRIPT_POINT:
                out.append(",")
            else:
                out.append(_SUPERSCRIPT.get(character, character))
        return "".join(out)

    return _SUPERSCRIPT_RUN.sub(replace, text)


def fix_thread_pitch(text: str) -> str:
    """
    «M28x15» -> «M28x1,5»: шага 15 мм у метрической резьбы не бывает.

    «Болт М12х50» при этом остаётся болтом: второе число — длина крепежа,
    и признаков у неё два. Шаг не бывает больше четверти диаметра (5 мм на
    резьбе М12 невозможны), и длина стоит в своём ряду по ГОСТ 7798.

    Цена такой осторожности — пропуски там, где ряды пересекаются: «М20х25»
    и «М30х35» не правятся, хотя шаги 2,5 и 3,5 на этих диаметрах
    настоящие, — 25 и 35 есть и в ряду длин. Так и задумано: пропустить
    правку безопаснее, чем сломать запись болта.
    """
    def replace(match: re.Match) -> str:
        prefix, mark, written = match.group(1), match.group(3), match.group(4)
        diameter = int(match.group(2))
        if int(written) in FASTENER_LENGTHS and int(written) >= diameter:
            return match.group()
        fixed = _only_valid(written, THREAD_PITCHES)
        if fixed is None:
            return match.group()
        if float(fixed.replace(",", ".")) > diameter * _MAX_PITCH_SHARE:
            return match.group()
        return prefix + mark + fixed

    return _THREAD.sub(replace, text)


def fix_roughness_letters(text: str, wide: bool = True) -> str:
    """
    «Ра 16», «Ка 3.2», «Rа 1,6» -> знак шероховатости латиницей.

    `wide=False` — прежний, более узкий образец (без кириллической «Р» и
    без склеенных слов): с выключенным флагом разбор остаётся тем же.
    """
    pattern = ROUGHNESS_LETTERS if wide else _ROUGHNESS_LETTERS_PLAIN
    return pattern.sub(
        lambda m: "R" + ("a" if m.group(2) in "aа" else "z"), text or ""
    )


def fix_roughness(text: str) -> str:
    """«Ра 16» -> «Ra 1,6»: латинские буквы знака и запятая на месте."""
    text = fix_roughness_letters(text)

    def replace(match: re.Match) -> str:
        fixed = _only_valid(match.group(3), ROUGHNESS_PREFERRED)
        if fixed is None:
            return match.group()
        return match.group(1) + match.group(2) + fixed

    return _ROUGHNESS.sub(replace, text)


def fix_deviation(text: str) -> str:
    """
    «2+15» -> «2+1,5», «40−025» -> «40−0,25». Правится только невозможное:
    отклонение с ведущим нулём без разделителя и — у «+» и «±» — отклонение
    больше самого размера. «30±10», «100+25», «4−16» остаются как есть.
    """
    def replace(match: re.Match) -> str:
        size, sign, digits = match.group(1), match.group(2), match.group(3)
        try:
            base = float(size.replace(",", "."))
        except ValueError:
            return match.group()
        if base <= 0:
            return match.group()
        # Невозможных записей две. Ведущий нуль без разделителя — всегда
        # потерянная запятая («40−025»). Отклонение больше размера — тоже,
        # но только у «+» и «±»: «4−16» с настоящим минусом — это диапазон
        # диаметра кабеля, а не отклонение.
        lost = digits.startswith("0") or (sign in "+±" and float(digits) >= base)
        if not lost:
            return match.group()
        found = [
            value for value in _comma_variants(digits)
            if "," in value and not _leading_zero(value)
            and float(value.replace(",", ".")) < base
        ]
        if not found:
            return match.group()
        # Запятая ставится как можно правее: «15» — это «1,5», а не «0,15»;
        # прочтений меньше размера может быть несколько, и ближе к записи
        # на листе то, в котором цифр до запятой больше.
        return size + sign + found[-1]

    return _DEVIATION.sub(replace, text)


def fix_weld_mark(text: str) -> str:
    """
    «H1-ИНн» -> «Н1-ИНп»: тип шва пишется кириллицей, а способов сварки в
    инертном газе по ГОСТ 14771 ровно два — «ИН» и «ИНп». Как «п» читается
    только буква, которая с «п» путается в шрифте по ГОСТ 2.304
    (`_WELD_TAIL_CONFUSION`). Любая другая буква в конце означает, что это
    не обозначение шва, и запись остаётся как на листе. Пустой хвост
    («У4-ИН») не дополняется.
    """
    def replace(match: re.Match) -> str:
        count, kind, number, method, tail = match.groups()
        fixed_method = "ИН"
        if tail and ("ИН" + tail) not in WELD_METHODS:
            if tail not in _WELD_TAIL_CONFUSION:
                return match.group()
            fixed_method = "ИНп"
        elif tail:
            fixed_method = "ИН" + tail
        return count + _WELD_TYPES[kind] + number + "-" + fixed_method

    return _WELD.sub(replace, text or "")


def repair_scale(text: str) -> str:
    """
    Значение графы «Масштаб»: «7:1» -> «1:1». Курсивная «1» по ГОСТ 2.304
    читается как «7», а ряд масштабов по ГОСТ 2.302 узкий, и прочтение из
    ряда ровно одно. Зовётся только для этой графы — «Поз. 7:1» в тексте
    листа правилу не подчиняется.
    """
    return _fix_ratio(_SCALE, text)


def fix_view_scale(text: str) -> str:
    """Масштаб вида в скобках: «Б(7:1)» -> «Б(1:1)». «Б(2:1)» уже из ряда."""
    return _fix_ratio(_VIEW_SCALE, text)


def _fix_ratio(pattern: re.Pattern, text: str) -> str:
    def replace(match: re.Match) -> str:
        current = f"{match.group(1)}:{match.group(2)}"
        if current in SCALES:
            return match.group()
        found = set()
        for index, character in enumerate(current):
            other = _SCALE_CONFUSION.get(character)
            if not other:
                continue
            candidate = current[:index] + other + current[index + 1:]
            if candidate in SCALES:
                found.add(candidate)
        if len(found) != 1:
            return match.group()
        return found.pop()

    return pattern.sub(replace, text or "")


def is_steel_grade(text: str) -> bool:
    """Есть ли в строке марка стали («12Х18Н10Т», «Сталь 08Х18Н10»)."""
    return bool(_grade_pattern(text).search(text or ""))


def _grade_pattern(text: str) -> re.Pattern:
    """
    Образец марки: обычный или, когда рядом сказано о материале, ещё и без
    хвоста легирующих элементов.
    """
    if text and _MATERIAL_CONTEXT.search(text):
        return _GRADE_IN_MATERIAL
    return GRADE


def mask_steel_grades(text: str) -> str:
    """
    Марки стали заменены точками той же длины. Нужна правилам посадки и
    допуска: в «12Х18Н10Т» нет ни того, ни другого, а места букв и цифр
    должны остаться теми же, чтобы совпадали смещения в строке.
    """
    return _grade_pattern(text).sub(lambda m: "." * len(m.group()), text or "")


def repair(text: str, where: str = "") -> str:
    """
    Надпись чертежа с восстановленными числами и знаками. Порядок важен:
    надстрочные цифры сначала превращаются в обычную запись, и только потом
    проверяются как отклонение.
    """
    if not text:
        return text
    fixed = expand_superscripts(str(text))
    fixed = fix_thread_pitch(fixed)
    fixed = fix_roughness(fixed)
    fixed = fix_deviation(fixed)
    fixed = fix_weld_mark(fixed)
    fixed = fix_view_scale(fixed)
    if fixed != text:
        logger.info("Число в надписи восстановлено%s: %r -> %r",
                    f" ({where})" if where else "", text, fixed)
    return fixed


def repair_lines(lines: Sequence[str]) -> List[str]:
    """Строки транскрипции листа с теми же правками."""
    return [repair(line, "транскрипция") for line in lines]


_HTML_TEXT_RE = re.compile(r">([^<>]+)<")


def repair_blocks(blocks: Sequence[Any]) -> int:
    """
    Текст блоков и ячейки таблиц MinerU — на месте. Число правок.

    Своя таблица MinerU и текст модели лежат в выдаче рядом: пока в ячейке
    оставался «M24×15», верная запись из чтения модели не помогала — по
    «M24x1,5» находилась одна из двух записей. Правится ячейка, а не
    выбрасывается таблица: остальные её числа верны.
    """
    changed = 0

    def fix(value: Any) -> Any:
        nonlocal changed
        if not isinstance(value, str) or not value.strip():
            return value
        fixed = repair(value, "таблица MinerU")
        changed += int(fixed != value)
        return fixed

    for block in blocks:
        if getattr(block, "type", "") == "text" and getattr(block, "text", ""):
            block.text = fix(block.text)
        elif getattr(block, "type", "") == "table" and getattr(block, "table_data", None):
            table = block.table_data
            table["headers"] = [fix(h) for h in (table.get("headers") or [])]
            table["rows"] = [[fix(c) for c in row] for row in (table.get("rows") or [])]
            block.table_data = table
            if getattr(block, "table_html", None):
                # Та же правка в исходном HTML таблицы: в счёт не идёт,
                # иначе одна исправленная ячейка считалась бы дважды.
                block.table_html = _HTML_TEXT_RE.sub(
                    lambda m: ">" + repair(m.group(1)) + "<", block.table_html
                )
    return changed


# =========================================================================
# Выбор между двумя прочтениями одной надписи
# =========================================================================

_SIGNS = "±+-−∓"
_SIGN_SLOT = "\x00"
_SPACES = re.compile(r"\s+")


def sign_key(text: str) -> str:
    """
    Запись надписи без знака отклонения и без справочной звёздочки:
    «30±10*», «30+10» и «30 - 10» дают один ключ. По нему два прочтения
    одного размера встречаются, и из них выбирается достоверное.
    """
    lowered = _SPACES.sub("", (text or "").lower()).replace("*", "")
    return "".join(_SIGN_SLOT if ch in _SIGNS else ch for ch in lowered)


_MINUS_BETWEEN_DIGITS = re.compile(r"\d\s?[−-]\s?\d")


def _rank(text: str) -> Tuple[int, int]:
    """
    Чем достовернее прочтение, тем больше. Знак, затем звёздочка. Обычный
    дефис между цифрами считается минусом наравне с «−»: модель подменяет
    на «+» и его, а `drawing_text.normalize` дефис в минус не переводит.
    """
    if "±" in text or "∓" in text:
        sign = 2
    elif "−" in text or _MINUS_BETWEEN_DIGITS.search(text):
        sign = 1
    else:
        sign = 0
    return sign, int("*" in text)


def prefer(first: str, second: str) -> str:
    """
    Из двух прочтений одной надписи — то, в котором знак сохранён, а если
    справочная звёздочка есть только у другого, она дописывается. «±»
    старше «−», «−» старше «+»: модель подменяет «±» на «+», обратной
    подмены не бывает. При равенстве остаётся первое — порядок задаёт тот,
    кто вызывает. Выбор между «−» и «+» доходит до выдачи только там, где
    место надписи известно (`_mergeable`): без рамки оба прочтения остаются.
    """
    better, worse = (
        (second, first) if _rank(second)[0] > _rank(first)[0] else (first, second)
    )
    if "*" in worse and "*" not in better:
        # Справочный размер по ГОСТ 2.307: звёздочка есть на листе, просто
        # одно из двух прочтений её потеряло.
        return better + "*"
    return better


def merge_readings(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Поля чертежа, в которых два прочтения одного размера сведены в одно.

    Сводятся только надписи на одном месте листа: пересечение рамок не
    меньше половины меньшей из них. Иначе «+0,2» и «−0,2», написанные на
    листе двумя отдельными надписями, слились бы в одну.
    """
    kept: List[Dict[str, Any]] = []
    for field in fields:
        key = sign_key(str(field.get("value") or ""))
        twin = None
        for other in kept:
            if sign_key(str(other.get("value") or "")) != key:
                continue
            if other.get("category") != field.get("category"):
                continue
            if _mergeable(other, field):
                twin = other
                break
        if twin is None:
            kept.append(field)
            continue
        kept_value, other_value = str(twin["value"]), str(field["value"])
        winner = prefer(kept_value, other_value)
        if winner != kept_value:
            # Прочтение со знаком берётся целиком — вместе с рамкой и
            # уверенностью того прочтения, которое знак сохранило.
            if _rank(other_value)[0] > _rank(kept_value)[0]:
                twin.update(field)
            twin["value"] = winner
        logger.info("Из двух прочтений размера оставлено %r (было ещё %r)",
                    winner, other_value if winner != other_value else kept_value)
    return kept


# Рамка во весь лист — это «место неизвестно»: такую рамку ставит
# `drawing_processor._bbox`, когда модель не вернула своей.
_WHOLE_SHEET = [0.0, 0.0, 1.0, 1.0]


def _mergeable(first: Dict[str, Any], second: Dict[str, Any]) -> bool:
    """
    Можно ли считать два прочтения одной надписью. На одном месте листа —
    да, любой разницы в знаке. Если место неизвестно (рамки нет или она во
    весь лист), сводятся только две известные ошибки модели: «+» вместо
    «±» и потерянная звёздочка. Иначе «+0,2» и «−0,2», написанные на листе
    двумя отдельными надписями, слились бы в одну.
    """
    one, two = str(first.get("value") or ""), str(second.get("value") or "")
    if _same_place(first.get("bbox"), second.get("bbox")):
        return True
    ranks = sorted((_rank(one)[0], _rank(two)[0]))
    return ranks in ([0, 0], [0, 2], [2, 2])


def _same_place(first: Any, second: Any) -> bool:
    """
    Одно ли это место листа. Рамки нет или она во весь лист — место
    неизвестно, и это не «то же место».
    """
    boxes = []
    for box in (first, second):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            return False
        if values == _WHOLE_SHEET:
            return False
        boxes.append(values)
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = boxes
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = width * height
    smaller = min(abs(ax2 - ax1) * abs(ay2 - ay1), abs(bx2 - bx1) * abs(by2 - by1))
    if smaller <= 0:
        return False
    return overlap / smaller >= 0.5
