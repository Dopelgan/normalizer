"""
Сбор справочника названий исполнений (`tools/harvest_designations.py --names`).

Названия придуманы. Проверяется: считаются только источники с текстовым
слоем, написание берётся, только если оно в большинстве и встречено не
меньше двух раз, а прочтения чертежей (OCR, модель) не голосуют.
"""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "harvest_designations.py"
_spec = importlib.util.spec_from_file_location("harvest_designations", _PATH)
harvest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harvest)


def _tsv(tmp_path, rows):
    path = tmp_path / "frags.tsv"
    path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
    return str(path)


def test_names_majority_from_text_layer(tmp_path, capsys):
    rows = [
        ("text_layer", "Клапан ВОЛНА-ПЛЭ-КР исполнение"),
        ("text_layer", "ВОЛНА-ПЛЭ-КР и МЗК-ДР"),
        ("text_layer_extraction", "Исполнение ВОЛНА-ПЛЗ-КР"),
        ("text_layer", "МЗК-ДР, МЗК-ДР"),
        # Прочтения чертежей не голосуют, сколько бы их ни было.
        ("vision_plus_vlm", "ВОЛНА-ПЛЗ-КР ВОЛНА-ПЛЗ-КР ВОЛНА-ПЛЗ-КР СОСНА-КЛЗ-ИХ СОСНА-КЛЗ-ИХ"),
        ("tesseract", "ВОЛНА-ПЛЗ-КР ВОЛНА-ПЛЗ-КР"),
        # Один раз — мало; спор поровну — нет большинства.
        ("text_layer", "БУК-ТР-01 ЕЛЬ-КЗ ЕЛЬ-КЭ"),
        # Числа и даты — не названия.
        ("text_layer", "12-34-56 12-34-56 2020-01 2020-01"),
    ]
    out = tmp_path / "known.txt"
    code = harvest.main([_tsv(tmp_path, rows), "--names", "--text-column", "2",
                         "--method-column", "1", "--output", str(out)])
    assert code == 0
    names = [line for line in out.read_text(encoding="utf-8").splitlines()
             if line and not line.startswith("#")]
    assert names == ["ВОЛНА-ПЛЭ-КР", "МЗК-ДР"]


def test_names_requires_method_column(tmp_path):
    path = _tsv(tmp_path, [("text_layer", "ВОЛНА-ПЛЭ-КР")])
    try:
        harvest.main([path, "--names"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("без столбца способа разбора режим названий не работает")
