"""
Листы, которые ветка чертежей не читала, не теряются молча.

`DRAWING_MAX_SHEETS` обрезал альбом по шестому листу, и паспорт на 7-8
чертёжных листов терял последние. Номера страниц были только внутри текста
пометки `DRAWING_SHEETS_SKIPPED:7,8` — списком через запятую, а не числом;
прочитать их машинно было неоткуда.
"""

from types import SimpleNamespace

import pytest

from core.config import settings
from ingest.tasks import mark_unread_sheets, skipped_sheet_pages


def _fragment(page, type_="text", warnings=()):
    return SimpleNamespace(type=type_, position=SimpleNamespace(page=page),
                           missing_regions=[], warnings=list(warnings))


def test_cap_covers_the_albums_that_lost_sheets():
    # В базе потеряны только листы 7 и 8: предела в восемь хватает на весь
    # известный корпус. Замер — в комментарии к настройке.
    assert settings.DRAWING_MAX_SHEETS >= 8


@pytest.mark.parametrize("warning, pages", [
    ("DRAWING_SHEETS_SKIPPED:8", [8]),
    ("DRAWING_SHEETS_SKIPPED:7,8", [7, 8]),
    ("DRAWING_SHEETS_SKIPPED:7, 8", [7, 8]),
    ("DRAWING_SHEETS_SKIPPED:", []),
    ("DRAWING_COMPLETENESS_UNKNOWN", []),
])
def test_skipped_pages_are_a_list_not_a_count(warning, pages):
    # Пометка несёт номера страниц, а не их число: «:7,8» — это страницы 7
    # и 8, а не восемь листов.
    assert skipped_sheet_pages([_fragment(1, "drawing", [warning])]) == pages


def test_pages_from_several_fragments_merge():
    fragments = [_fragment(1, "drawing", ["DRAWING_SHEETS_SKIPPED:7,8"]),
                 _fragment(2, "drawing", ["DRAWING_SHEETS_SKIPPED:8"])]
    assert skipped_sheet_pages(fragments) == [7, 8]


def test_unread_sheets_marked_on_text_fragments_only():
    # `missing_regions` чертежа ведёт документ к человеку; непрочитанный
    # лист дочитывает система.
    fragments = [_fragment(6), _fragment(7), _fragment(7), _fragment(8, "drawing")]
    marked = mark_unread_sheets(fragments, [7, 8])
    assert marked == 1
    assert fragments[1].missing_regions[0] == {
        "kind": "sheet", "bbox": [0.0, 0.0, 1.0, 1.0], "page": 7,
        "reason": "sheet_not_read",
    }
    assert fragments[2].missing_regions == []
    assert fragments[3].missing_regions == []
    assert fragments[0].missing_regions == []


def test_nothing_marked_without_skipped_sheets():
    fragments = [_fragment(1), _fragment(2)]
    assert mark_unread_sheets(fragments, []) == 0
    assert all(not f.missing_regions for f in fragments)
