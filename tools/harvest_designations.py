"""
Сбор справочника обозначений по корпусу разобранных документов.

Справочник нужен `core/providers/designation_repair.py`: код организации
ЕСКД и марка стали — это части обозначения, у которых буквы и есть
содержание, и правятся они только до настоящего обозначения. Список
собирается здесь, а не пишется руками, чтобы его можно было пересобрать,
когда в базе появятся документы другого предприятия.

Берутся только написания, встреченные в корпусе не меньше `--min-count`
раз: код, встреченный единично, сам может быть порчей другого кода («ГАДУ»,
«ТАЛУ», «САДУ» — это испорченное «ТАДУ»), и справочником он не становится.
Марка проверяется ещё и формой: две цифры, не меньше двух легирующих букв,
хотя бы одна буква не «Х» и хотя бы одна с цифрой после неё — иначе в
список попадают обрывки обозначений («00СБ», «22ТУ»).

Запуск (пример; выгрузка — текст фрагментов корпуса, TSV):

    python tools/harvest_designations.py фрагменты.tsv --text-column 7

Печатается то, что можно перенести в `ESKD_CODES` и `GRADES`. Решение о
переносе — за человеком: список настоящих обозначений важнее полноты.

Режим `--names` собирает справочник названий исполнений для
`DRAWING_DESIGNATION_REPAIR` (файл `DESIGNATION_KNOWN_NAMES_PATH`):

    python tools/harvest_designations.py фрагменты.tsv --names \
        --text-column 2 --method-column 1 --output config/known_designations.txt

Название — слово из частей через дефис в прописных («ВОЛНА-ПЛЭ-КР»).
Голосуют только источники с текстовым слоем (`--text-methods`, по способу
разбора фрагмента): в PDF с текстом, спецификации, Word буквы набраны, а
не прочитаны. Прочтения чертежей и сканов (OCR, модель) не голосуют —
именно их и правит справочник. Написания, неразличимые на листе
(«ПЛЗ»/«ПЛЭ»/«ПЛ3»), считаются одной группой; в справочник идёт
написание, которое в группе в большинстве и встречено не меньше
`--min-count` раз (по умолчанию в этом режиме 2). Спор поровну — ничего.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ESKD = re.compile(r"(?<![0-9A-Za-zА-Яа-яЁё])([А-ЯЁ]{4})[  ]?\d{6}\.\d{3}")
_GRADE = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])((?:Св-)?\d{2}"
    r"(?:[АБВГДЕЖКЛМНПРСТФХЦЧЮЯ]\d{0,2}){2,6})(?![0-9A-Za-zА-Яа-яЁё])")
_ALLOY = re.compile(r"[А-ЯЁ]")
_ALLOY_WITH_DIGIT = re.compile(r"[А-ЯЁ]\d")


def _is_grade(token: str) -> bool:
    """Форма марки по ГОСТ 5632 и ГОСТ 2246, а не обрывок обозначения."""
    body = token[3:] if token.startswith("Св-") else token
    tail = body[2:]
    letters = _ALLOY.findall(tail)
    return (len(letters) >= 2 and any(letter != "Х" for letter in letters)
            and bool(_ALLOY_WITH_DIGIT.search(tail)))


# Название исполнения: не меньше двух частей через дефис, прописными.
_NAME = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё-])([А-ЯЁA-Z0-9]{2,}(?:-[А-ЯЁA-Z0-9]{2,})+)"
    r"(?![0-9A-Za-zА-Яа-яЁё-])")
_CYRILLIC_LETTER = re.compile(r"[А-ЯЁ]")
# Способы разбора, при которых текст набран, а не прочитан с изображения.
TEXT_METHODS = ("cad_source", "text_layer", "text_layer_extraction",
                "vector_text_layer", "docx", "doc", "word")


def harvest_names(rows, text_column: int, method_column: int,
                  methods, min_count: int) -> list:
    """Названия, написанные в большинстве источников с текстовым слоем."""
    from core.providers.designation_repair import _name_fold

    allowed = {method.strip() for method in methods if method.strip()}
    groups: dict = collections.defaultdict(collections.Counter)
    for row in rows:
        if len(row) <= max(text_column, method_column):
            continue
        if row[method_column].strip() not in allowed:
            continue
        text = row[text_column].replace("\\\\n", "\n").replace("\\n", "\n")
        for match in _NAME.finditer(text):
            name = match.group(1)
            if not _CYRILLIC_LETTER.search(name):
                continue
            groups[_name_fold(name)][name] += 1
    names = []
    for spellings in groups.values():
        ranked = spellings.most_common()
        best, count = ranked[0]
        total = sum(spellings.values())
        if count >= min_count and count * 2 > total:
            names.append(best)
    return sorted(names)


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="выгрузка текстов фрагментов, TSV")
    parser.add_argument("--text-column", type=int, default=1,
                        help="номер столбца с текстом, с единицы")
    parser.add_argument("--min-count", type=int, default=None,
                        help="сколько раз написание должно встретиться "
                             "(50; в режиме --names 2)")
    parser.add_argument("--names", action="store_true",
                        help="справочник названий исполнений")
    parser.add_argument("--method-column", type=int, default=0,
                        help="номер столбца со способом разбора (для --names)")
    parser.add_argument("--text-methods", default=",".join(TEXT_METHODS),
                        help="способы разбора с текстовым слоем, через запятую")
    parser.add_argument("--output", default="",
                        help="куда записать справочник (--names), иначе stdout")
    args = parser.parse_args(argv)

    csv.field_size_limit(10 ** 8)
    if args.names:
        if args.method_column < 1:
            parser.error("--names требует --method-column: голосуют только "
                         "источники с текстовым слоем")
        with open(args.path, encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            names = harvest_names(
                rows, args.text_column - 1, args.method_column - 1,
                args.text_methods.split(","),
                2 if args.min_count is None else args.min_count)
        body = ("# Справочник названий (tools/harvest_designations.py --names)\n"
                + "".join(name + "\n" for name in names))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(body)
        else:
            sys.stdout.write(body)
        return 0
    if args.min_count is None:
        args.min_count = 50
    codes: collections.Counter = collections.Counter()
    grades: collections.Counter = collections.Counter()
    column = args.text_column - 1
    with open(args.path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) <= column:
                continue
            text = row[column].replace("\\\\n", "\n").replace("\\n", "\n")
            for match in _ESKD.finditer(text):
                codes[match.group(1)] += 1
            for match in _GRADE.finditer(text):
                if _is_grade(match.group(1)):
                    grades[match.group(1)] += 1

    print("# ESKD_CODES (встречено не меньше %d раз)" % args.min_count)
    for code, count in codes.most_common():
        mark = "" if count >= args.min_count else "  # редко, не переносить"
        print('    "%s",  # %d%s' % (code, count, mark))
    print("# GRADES (встречено не меньше %d раз)" % args.min_count)
    for grade, count in grades.most_common():
        mark = "" if count >= args.min_count else "  # редко, не переносить"
        print('    "%s",  # %d%s' % (grade, count, mark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
