"""
Две молчаливые потери текста на проде 13.11.

A. Дедупликатор повторов удалял настоящий текст: предложение, делящее с
   соседним фрагментом один оборот, считалось его повтором.
B. Распознавание доходило до пятидесятой страницы из полутораста, а документ
   выдавался разобранным целиком.

Строки и номера страниц здесь — из «Приложения №15 … Порядок взаимодействия»
и из стопки на 158 листов, на которых обе потери и нашлись.
"""

from types import SimpleNamespace

import pytest

from core.config import settings
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.mineru_parser import (
    MinerUParserProvider,
    _contained_share,
    _stems,
    drop_repeated_recovered_blocks,
    page_ranges,
)
from core.providers.text_layer import METHOD_OCR_LAYER_RECOVERED as REC
from core.providers.text_layer import TextLayer, TextLine
from ingest.tasks import build_warnings, mark_unread_pages

# Пункт 6 приложения: предложение, которое пропало, и его сосед, за повтор
# которого оно было принято.
LOST = ("проводится расследование. Покупатель письменно извещает Поставщика "
        "о необходимости прибытия для расследования выявленных несоответствий.")
NEIGHBOUR = ("Извещение Поставщика о необходимости прибытия для расследования "
             "выявленных несоответствий может быть осуществлено по выбору "
             "Покупателя следующими способами:")
PAGE_PROSE = (
    "7. Представитель Поставщика для участия в расследовании несоответствий, "
    "составлении акта расследования выявленных несоответствий должен иметь "
    "полномочия в виде надлежащим образом оформленной доверенности, форма "
    "которой является приложением № 4 (Форма 4.22) к договору поставки. "
    "8. Представитель Поставщика вправе участвовать в расследовании выявленных "
    "несоответствий. В случае неприбытия представителя Поставщика, извещенного "
    "о необходимости явки для участия в расследовании выявленных "
    "Несоответствий согласно п.6 настоящего Порядка, либо отказа представителя "
    "Поставщика от подписания Акта расследования выявленных несоответствий, "
    "действительным считается Акт расследования выявленных несоответствий, "
    "подписанный Покупателем в одностороннем порядке."
)
CONTAIN = 0.9


def _block(text, method, page=1, type_="text"):
    return ParsedBlock(type=type_, text=text, page=page, bbox=[0, 0, 1, 1], order=0,
                       confidence=0.9, method=method)


# --------------------------------------------------------------------- A
def test_sentence_sharing_one_phrase_is_not_a_repeat():
    # Слова потерянного предложения нашлись по странице вразнобой — старое
    # правило считало это повтором (доля 0.82 при пороге 0.6). Нового порога
    # не достигает ни одно сравнение с отдельным блоком.
    blocks = [
        _block(PAGE_PROSE, "mineru_ocr"),
        _block(NEIGHBOUR, "mineru_ocr"),
        _block(LOST, REC),
    ]
    kept, counts = drop_repeated_recovered_blocks(blocks, CONTAIN)
    assert counts["total"] == 0
    assert any(f.text == LOST for f in kept)
    # Та же пара чисел, на которой разошлись правила: по объединению слов
    # страницы без порядка — выше прежнего порога 0.6, по отдельному блоку с
    # порядком слов — ниже нового 0.9.
    stems = _stems(LOST)
    union = set(_stems(PAGE_PROSE)) | set(_stems(NEIGHBOUR))
    assert sum(s in union for s in stems) / len(stems) >= 0.6
    assert max(_contained_share(stems, _stems(other))
               for other in (PAGE_PROSE, NEIGHBOUR)) < CONTAIN


def test_garbled_copy_of_the_same_block_is_still_dropped():
    # Та же строка, прочитанная Tesseract хуже и через графику: содержится в
    # одном блоке страницы целиком, окончания и мусор в счёт не идут.
    copy = ("$ \\ < Извещение Поставщика о необходимости прибытия для "
            "расследования выявленных несоответствий может быть осуществленно "
            "по выбору Покупателя следующими способами")
    blocks = [
        _block(PAGE_PROSE, "mineru_ocr"),
        _block(NEIGHBOUR, "mineru_ocr"),
        _block(copy, REC),
    ]
    kept, counts = drop_repeated_recovered_blocks(blocks, CONTAIN)
    assert counts["repeat"] == 1
    assert counts["total"] == 1
    assert all(f.text != copy for f in kept)


def test_latin_garbage_is_kept_and_counted_uncertain():
    # «Cozobopa MEXHONOZUYECKUX» — русские слова, прочитанные латиницей. На
    # текст это похоже, доказанным повтором не является: блок остаётся, а
    # факт сомнения записывается.
    junk = "Cozobopa MEXHONOZUYECKUX usdenua"
    blocks = [_block(PAGE_PROSE, "mineru_ocr"), _block(junk, REC)]
    kept, counts = drop_repeated_recovered_blocks(blocks, CONTAIN)
    assert counts["total"] == 0
    assert counts["kept_uncertain"] == 1
    assert any(f.text == junk for f in kept)


def test_pure_graphics_noise_is_still_dropped():
    # Ни русского, ни латинского слова — обводка чертежа, читать нечего.
    blocks = [_block(PAGE_PROSE, "mineru_ocr"), _block("a С» 125) \\/ R263 1\\/) G", REC)]
    kept, counts = drop_repeated_recovered_blocks(blocks, CONTAIN)
    assert counts["graphics"] == 1
    assert len(kept) == 1


# --------------------------------------------------------------------- B
class _Truncating:
    """Распознавание, которое читает по `step` страниц за проход."""

    def __init__(self, total, step):
        self.total, self.step = total, step
        self.truncated_at = None
        self.failed_pages = []
        self.starts = []

    def text_layer(self, uri, file_type=None, first_page=1):
        self.starts.append(first_page)
        last = min(self.total, first_page + self.step - 1)
        pages = {
            page: [TextLine(text=f"строка {page}", bbox=[0, 0, 1, 1], confidence=0.9,
                            page=page)]
            for page in range(first_page, last + 1)
        }
        self.truncated_at = None if last >= self.total else (last, self.total)
        return TextLayer(pages=pages, page_count=last)


def _provider(monkeypatch, tesseract):
    provider = MinerUParserProvider.__new__(MinerUParserProvider)
    provider.tesseract = tesseract
    return provider


def test_truncated_layer_reports_unread_pages(monkeypatch):
    # Один проход на 50 страниц из 158: непрочитанные страницы названы, а не
    # потеряны. Это и есть тот документ, который выдавался completed.
    monkeypatch.setattr(settings, "OCR_LAYER_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_LAYER_CONTINUATION_PASSES", 0)
    tesseract = _Truncating(total=158, step=50)
    layer, unread = _provider(monkeypatch, tesseract)._ocr_layer(
        "s3://x.pdf", "pdf", total_pages=158
    )
    assert layer is not None and layer.page_count == 50
    assert unread == list(range(51, 159))
    assert page_ranges(unread) == "51-158"


def test_continuation_passes_read_the_whole_document(monkeypatch):
    # Продолжение прохода дочитывает документ само: человек для этого не нужен.
    monkeypatch.setattr(settings, "OCR_LAYER_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_LAYER_CONTINUATION_PASSES", 3)
    tesseract = _Truncating(total=158, step=50)
    layer, unread = _provider(monkeypatch, tesseract)._ocr_layer(
        "s3://x.pdf", "pdf", total_pages=158
    )
    assert tesseract.starts == [1, 51, 101, 151]
    assert layer.page_count == 158
    assert unread == []


def test_continuation_stops_when_a_pass_makes_no_progress(monkeypatch):
    # Проход, остановленный бюджетом времени, не продвинулся ни на страницу:
    # повторять его бессмысленно, а молчать о непрочитанном нельзя.
    monkeypatch.setattr(settings, "OCR_LAYER_ENABLED", True)
    monkeypatch.setattr(settings, "OCR_LAYER_CONTINUATION_PASSES", 5)

    class _Stuck(_Truncating):
        def text_layer(self, uri, file_type=None, first_page=1):
            self.starts.append(first_page)
            self.truncated_at = (50, self.total)
            return TextLayer(pages={1: [TextLine(text="строка", bbox=[0, 0, 1, 1],
                                                 confidence=0.9, page=1)]}, page_count=1)

    tesseract = _Stuck(total=158, step=50)
    _, unread = _provider(monkeypatch, tesseract)._ocr_layer(
        "s3://x.pdf", "pdf", total_pages=158
    )
    assert tesseract.starts == [1, 51]
    assert unread == list(range(51, 159))


def test_blank_page_inside_the_read_part_is_not_unread():
    # Пустая страница скана и непрочитанная страница — разные вещи.
    unread = MinerUParserProvider._unread_pages({1: [], 3: []}, 5, (3, 5))
    assert unread == [4, 5]


# ------------------------------------------------- пометки для выдачи
def test_unread_pages_reach_the_warnings_once():
    result = ParseResult(unread_text_pages=list(range(51, 159)), source_page_count=158)
    warnings = build_warnings(result, [])
    assert "TEXT_NOT_READ: pages 51-158" in warnings
    assert sum(w.startswith("TEXT_NOT_READ") for w in warnings) == 1


def _fragment(page, type_="text"):
    return SimpleNamespace(type=type_, position=SimpleNamespace(page=page),
                           missing_regions=[])


def test_unread_pages_marked_on_text_fragments_only():
    # `missing_regions` чертежа отправляет документ человеку; качество
    # разбора поводом для этого не является.
    fragments = [_fragment(1), _fragment(51), _fragment(51), _fragment(52, "drawing")]
    marked = mark_unread_pages(fragments, [51, 52])
    assert marked == 1
    assert fragments[1].missing_regions[0]["reason"] == "ocr_pages_not_read"
    assert fragments[1].missing_regions[0]["page"] == 51
    assert fragments[2].missing_regions == []
    assert fragments[3].missing_regions == []
    assert fragments[0].missing_regions == []


@pytest.mark.parametrize("pages, text", [
    ([], ""),
    ([7], "7"),
    ([51, 52, 53, 100], "51-53, 100"),
])
def test_page_ranges(pages, text):
    assert page_ranges(pages) == text
