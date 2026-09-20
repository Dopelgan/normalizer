"""
Отличаем формулу от текста, завёрнутого в LaTeX.

Зачем это есть. Модель распознавания таблиц MinerU отдаёт содержимое ячеек
в LaTeX. Для колонки с формулами это ровно то, что нужно. Но когда в ячейке
русская проза, та же модель заворачивает в разметку каждую букву по
отдельности, попутно путая кириллицу с латиницей:

    \mathsf { v } [ \mathsf { M } / \mathsf { c } ] - \mathsf { c } \mathsf { K } ...

Это не формула и не текст, а мусор: ни прочитать, ни найти поиском.
Восстановить из него исходную строку нельзя — буквы потеряны при
распознавании, а не при разметке. Поэтому задача модуля не чинить, а
**узнавать** такие строки, чтобы выше по конвейеру их заменили текстом,
прочитанным нормальным OCR.

`decode` нужен для сравнения со строками OCR и как последнее средство,
когда заменить нечем: голый текст без разметки хотя бы читается глазами.

Модуль ничего не импортирует из конвейера и никуда не ходит.
"""

from __future__ import annotations

import re
from typing import List

# Команды переключения гарнитуры. В формуле их единицы, в «прозе, завёрнутой
# в LaTeX» — по одной на букву, и это главный признак.
_TEXT_FONT_COMMANDS = (
    "mathsf", "textsf", "mathrm", "textrm", "mathtt", "texttt",
    "textbf", "textit", "operatorname",
)

# Снимается при декодировании, но признаком прозы не служит: `\boldsymbol`
# в формуле помечает вектор, и три вектора в строке — это всё ещё формула.
_MATH_FONT_COMMANDS = ("boldsymbol", "textcircled", "breve", "Breve", "mathcal")

_FONT_RE = re.compile(r"\\(" + "|".join(_TEXT_FONT_COMMANDS) + r")\b")

# Сколько переключений гарнитуры делают строку подозрительной. Настоящая
# формула обходится одним-двумя (\boldsymbol у вектора, \mathrm у индекса).
_FONT_LIMIT = 3

# Длина «слова», собранного из одиночных букв, после которой строка
# считается прозой — независимо от того, переключалась ли гарнитура.
# Проверено на выдаче: в настоящих формулах (`formulas.png`, формульные
# ячейки `physical_formulas.png`) самая длинная такая цепочка — 3 буквы,
# у прозы вроде «A K T T E X H M \in C K O \Gamma 0 O C M O T P A» — 8.
_RUN_LIMIT = 6

# Разметка, которая при снятии не несёт содержимого.
_DROP_RE = re.compile(
    r"\\(?:" + "|".join(_TEXT_FONT_COMMANDS + _MATH_FONT_COMMANDS)
    + r"|displaystyle|scriptstyle|textstyle"
    r"|left|right|big|Big|bigg|Bigg|not|quad|qquad)\b|\\[,;!:]"
)

# Немногие команды, у которых есть однозначный символ. Больше не нужно:
# модуль не переводит формулы, он снимает разметку с текста.
_SYMBOLS = {
    r"\varOmega": "Ω", r"\varPi": "П", r"\varphi": "φ",
    r"\Omega": "Ω", r"\Delta": "Δ", r"\Theta": "Θ", r"\Pi": "П",
    r"\alpha": "α", r"\beta": "β", r"\omega": "ω", r"\phi": "φ", r"\pi": "π",
    r"\mu": "μ", r"\nu": "ν", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ",
    r"\lambda": "λ", r"\times": "×", r"\cdot": "·", r"\approx": "≈",
    r"\pm": "±", r"\leq": "≤", r"\geq": "≥", r"\infty": "∞", r"\nabla": "∇",
    r"\circ": "°", r"\square": "□", r"\mapsto": "→", r"\in": "∈", r"\cup": "∪",
    r"\land": "∧", r"\exists": "∃",
}

_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
_SPACES_RE = re.compile(r"\s+")
# Одиночные буквы, разделённые пробелами: «K O O p a m H a r c a».
_SINGLE_LETTERS_RE = re.compile(r"(?:(?<=\s)|^)((?:[^\W\d_]\s+){2,}[^\W\d_])(?=\s|$)")


def font_switches(latex: str) -> int:
    """Сколько раз в строке переключается гарнитура."""
    return len(_FONT_RE.findall(latex or ""))


def is_mangled_text(latex: str) -> bool:
    """
    Строка — это проза, завёрнутая в LaTeX, а не формула.

    Признак прямой: в настоящей формуле гарнитура переключается один-два
    раза, а здесь — на каждой букве. Второй признак самостоятелен: длинная
    цепочка одиночных букв остаётся прозой и без единого переключения —
    `A K T T E X H M \in C K O \Gamma 0 O C M O T P A` разметки почти не
    несёт, а формулой не является. Раньше такая строка проходила насквозь,
    потому что цепочка засчитывалась только вместе с переключением.
    """
    if not latex or "\\" not in latex:
        return False
    if font_switches(latex) >= _FONT_LIMIT:
        return True
    return _longest_letter_run(latex) >= _RUN_LIMIT


def decode(latex: str) -> str:
    """
    Снимает разметку, оставляя то, что под ней лежит.

    Это не перевод LaTeX в текст: `\frac` и индексы теряются намеренно —
    функция применяется к строкам, которые формулой не являются. Одиночные
    буквы, разнесённые разметкой, склеиваются обратно в слова.
    """
    if not latex:
        return ""

    text = latex
    for command, symbol in _SYMBOLS.items():
        text = text.replace(command, symbol)
    text = _DROP_RE.sub(" ", text)
    # Всё, что осталось от команд, содержимого не несёт.
    text = _COMMAND_RE.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ").replace("\\", " ")
    text = _SPACES_RE.sub(" ", text).strip()
    text = _SINGLE_LETTERS_RE.sub(lambda m: m.group(1).replace(" ", ""), text)
    return _SPACES_RE.sub(" ", text).strip()


def strip_for_match(latex: str) -> str:
    """Представление для сравнения со строкой OCR: без разметки и пробелов."""
    return "".join(decode(latex).split())


def _longest_letter_run(latex: str) -> int:
    """Самая длинная цепочка одиночных букв, разделённых пробелами."""
    longest = 0
    for match in _SINGLE_LETTERS_RE.finditer(latex):
        longest = max(longest, len(match.group(1).replace(" ", "")))
    return longest


def mangled_cells(table_data: dict) -> List[str]:
    """Ячейки таблицы, в которых лежит проза под разметкой."""
    if not table_data:
        return []
    cells = list(table_data.get("headers") or [])
    for row in table_data.get("rows") or []:
        cells.extend(row)
    return [str(c) for c in cells if is_mangled_text(str(c))]
