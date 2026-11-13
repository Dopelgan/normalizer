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
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import sys

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


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="выгрузка текстов фрагментов, TSV")
    parser.add_argument("--text-column", type=int, default=1,
                        help="номер столбца с текстом, с единицы")
    parser.add_argument("--min-count", type=int, default=50,
                        help="сколько раз написание должно встретиться")
    args = parser.parse_args(argv)

    csv.field_size_limit(10 ** 8)
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
