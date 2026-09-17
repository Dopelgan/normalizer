"""Нормализатор: структура фрагментов, полнота, происхождение."""

import pytest

from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer, parse_expression
from core.normalizer.completeness import formula_completeness, table_completeness

META = {"doc_id": "doc-001", "doc_type": "article"}


def text_block(text, page=1, bbox=None, order=0, **kwargs):
    return ParsedBlock(type="text", text=text, page=page,
                       bbox=bbox or [0.0, 0.0, 1.0, 1.0], order=order, **kwargs)


class TestTextFragments:
    def test_fragment_matches_contract(self):
        result = ParseResult(
            blocks=[text_block("Настоящий документ описывает требования к изделию. " * 3)],
            parser_name="mineru", source_kind="vector_pdf",
        )
        fragments = TextNormalizer().normalize(result, META)

        assert fragments
        fragment = fragments[0]
        assert fragment.fragment_id == "doc-001-frag-001"
        assert fragment.type == "text"
        assert fragment.content.text
        assert fragment.position.page == 1
        assert fragment.position.order == 1
        assert 0 <= fragment.confidence <= 1
        assert 0 <= fragment.completeness <= 1
        # Блок пришёл от MinerU и ничего не сообщил о способе получения —
        # значит это распознавание, а не чтение текстового слоя. Раньше
        # любому тексту векторного PDF подписывался text_layer_extraction.
        assert fragment.provenance.method == "mineru_ocr"
        assert fragment.provenance.strategy_level == 5
        assert fragment.provenance.source == "vector_pdf"
        assert fragment.graph_nodes == []

    @pytest.mark.parametrize("method,expected_method,expected_level", [
        ("cad_source", "cad_source", 1),
        ("text_layer", "text_layer_extraction", 2),
        ("text_layer_recovered", "text_layer_extraction", 2),
        ("office_extraction", "office_extraction", 2),
        ("plain_text", "plain_text_extraction", 2),
        ("tabular_extraction", "tabular_extraction", 3),
        ("tesseract", "tesseract", 4),
        ("mineru_ocr", "mineru_ocr", 5),
        ("restored_layout", "restored_layout", 6),
        ("vlm_escalation", "vlm_escalation", 7),
        (None, "mineru_ocr", 5),
    ])
    def test_provenance_follows_actual_method(self, method, expected_method, expected_level):
        """
        Происхождение берётся из блока, а не угадывается по типу фрагмента,
        а уровень соответствует лестнице стратегий 1..7.
        """
        result = ParseResult(
            blocks=[text_block("Текст достаточной длины для отдельного фрагмента. " * 2,
                               method=method)],
            parser_name="mineru", source_kind="vector_pdf",
        )
        fragment = TextNormalizer().normalize(result, META)[0]
        assert fragment.provenance.method == expected_method
        assert fragment.provenance.strategy_level == expected_level

    def test_fragment_ids_unique_and_ordered(self):
        blocks = [
            text_block(f"Абзац номер {i} с достаточной длиной текста. " * 2, page=1 + i // 3, order=i)
            for i in range(6)
        ]
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=blocks, parser_name="mineru"), META
        )
        ids = [f.fragment_id for f in fragments]
        assert len(ids) == len(set(ids))
        assert [f.position.order for f in fragments] == list(range(1, len(fragments) + 1))

    def test_empty_parse_result(self):
        assert TextNormalizer().normalize(ParseResult(blocks=[]), META) == []

    def test_short_noise_dropped(self):
        blocks = [
            text_block("ок", order=0),
            text_block("Полноценный абзац достаточной длины для индексации. " * 2, order=1),
        ]
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=blocks, parser_name="mineru"), META
        )
        assert all(len(f.content.text) >= 30 for f in fragments)


class TestCompleteness:
    def test_partial_page_coverage(self):
        """Блок покрывает 0.5 * 0.6 = 0.3 площади листа."""
        block = text_block("Текст с ограниченной областью на листе. " * 2, bbox=[0.0, 0.0, 0.5, 0.6])
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="mineru"), META
        )
        assert fragments[0].completeness == pytest.approx(0.3, abs=0.01)

    def test_fallback_penalty(self):
        block = text_block("Распознанный сканом текст достаточной длины. " * 2,
                           bbox=[0.0, 0.0, 0.5, 0.6], is_fallback=True)
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="tesseract_full",
                        source_kind="scanned_pdf", is_fallback=True), META
        )
        assert fragments[0].completeness == pytest.approx(0.24, abs=0.01)
        assert fragments[0].provenance.method == "tesseract"
        assert fragments[0].provenance.source == "scanned_pdf"

    def test_png_is_not_called_a_scanned_pdf(self):
        """
        Источник берётся из разбора, а не прописан константой. В выдаче по
        `formulas.png` стояло `scanned_pdf`, хотя это картинка, — и по
        происхождению нельзя было понять, что вообще произошло.
        """
        block = text_block("Распознанная картинка. " * 3, is_fallback=True)
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="plain_ocr",
                        source_kind="image", is_fallback=True), META
        )
        assert fragments[0].provenance.source == "image"
        assert fragments[0].provenance.strategy_level == 4

    def test_tesseract_inside_layout_is_level_five(self):
        """
        `tesseract_full` — это фоллбэк внутри разбора с детекцией областей:
        MinerU звали, он ответил без текста. Раньше он подписывался уровнем
        4 и был неотличим от собственно уровня 4.
        """
        block = text_block("Страницу дочитал OCR. " * 3, is_fallback=True)
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="tesseract_full",
                        source_kind="image", is_fallback=True), META
        )
        assert fragments[0].provenance.strategy_level == 5

    def test_no_bbox_is_penalized_not_inflated(self):
        """Когда координат нет, полнота не должна быть безосновательной 1.0."""
        block = text_block("Текст без координат, bbox на весь лист. " * 2, bbox=[0, 0, 1, 1])
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="mineru"), META
        )
        assert fragments[0].completeness == pytest.approx(0.9, abs=0.01)

    def test_exhaustive_extraction_is_not_penalised_for_missing_bbox(self):
        """
        У DOCX, TXT и XLSX координат нет и быть не может. Штраф ставился за
        «покрытие листа не проверить», и полный разбор абзаца объявлялся
        неполным на 0.9 — то есть уезжал в очередь на проверку ни за что.
        """
        block = text_block("Абзац из DOCX, координат у него нет. " * 2, bbox=[0, 0, 1, 1])
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="docx", exhaustive=True), META
        )
        assert fragments[0].completeness == 1.0

    def test_table_completeness_counts_filled_cells(self):
        assert table_completeness({"headers": ["A", "B"], "rows": [["1", "2"]]}) == 1.0
        assert table_completeness({"headers": ["A", "B"], "rows": [["1", ""]]}) == 0.5
        assert table_completeness({"headers": ["A"], "rows": []}) == 0.5
        assert table_completeness(None) == 0.0

    def test_formula_completeness(self):
        assert formula_completeness("x = 1", None, None) == 0.5
        assert formula_completeness("x = 1", "<math/>", {"base_var": "x"}) == 1.0
        assert formula_completeness("", None, None) == 0.0


class TestOtherFragmentTypes:
    def test_table_fragment(self):
        block = ParsedBlock(
            type="table", page=3, bbox=[0.1, 0.1, 0.9, 0.5],
            table_data={"headers": ["Параметр", "Значение"], "rows": [["Ra", "3.2"]]},
        )
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="mineru"), META
        )
        assert len(fragments) == 1
        assert fragments[0].type == "table"
        assert fragments[0].content.table_data.headers == ["Параметр", "Значение"]
        assert fragments[0].completeness == 1.0
        assert fragments[0].provenance.method == "mineru_table"

    def test_formula_fragment(self):
        block = ParsedBlock(type="formula", text="S = a * b", page=2, bbox=[0.2, 0.2, 0.6, 0.3])
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="mineru"), META
        )
        assert fragments[0].type == "formula"
        assert fragments[0].content.parsed_expression.base_var == "S"
        assert fragments[0].content.formula_mathml is None

    def test_image_fragment(self):
        block = ParsedBlock(type="image", image_ref="assets/doc-001/fig1.jpg",
                            page=2, bbox=[0.2, 0.2, 0.8, 0.8])
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="mineru"), META
        )
        assert fragments[0].type == "image"
        assert fragments[0].content.image_ref == "assets/doc-001/fig1.jpg"
        assert fragments[0].completeness == 1.0

    def test_image_without_ref_skipped(self):
        block = ParsedBlock(type="image", page=1, bbox=[0, 0, 1, 1])
        assert TextNormalizer().normalize(
            ParseResult(blocks=[block], parser_name="mineru"), META
        ) == []


class TestFragmentOrder:
    def test_text_does_not_jump_ahead_of_the_table_it_follows(self):
        """
        `order` у куска текста — номер после нарезки, у таблицы — номер блока
        на странице. Сортировка по этому полю ставила первый кусок текста
        впереди таблицы, хотя нарезан он из блока, который шёл после неё.
        """
        table = ParsedBlock(
            type="table", page=1, order=5, bbox=[0.1, 0.1, 0.9, 0.4],
            table_data={"headers": ["Параметр"], "rows": [["Ra"]]},
        )
        after = text_block(
            "Текст после таблицы, достаточно длинный для отдельного фрагмента. " * 2,
            page=1, order=6,
        )
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[table, after], parser_name="mineru"), META
        )
        assert [f.type for f in fragments] == ["table", "text"]

    def test_chunks_of_one_block_keep_their_own_order(self):
        """Куски одного блока различаются только номером куска — по нему и идут."""
        long_text = " ".join(f"Предложение номер {i} в длинном абзаце." for i in range(1, 120))
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[text_block(long_text, page=1, order=7)], parser_name="mineru"),
            META,
        )
        assert len(fragments) > 1
        assert [f.position.order for f in fragments] == list(range(1, len(fragments) + 1))
        assert fragments[0].content.text.startswith("Предложение номер 1")


    def test_text_and_tables_keep_document_order(self):
        """
        Абзацы, между которыми стоит таблица, не склеиваются в один кусок.
        Раньше нарезка получала все текстовые блоки одним списком, склейка
        вставала на место своего первого блока, и обе таблицы страницы
        уезжали за неё: текст-текст-текст-таблица-таблица вместо порядка
        документа.
        """
        def table(order, y0):
            return ParsedBlock(
                type="table", page=1, order=order, bbox=[0.1, y0, 0.9, y0 + 0.05],
                table_data={"headers": ["Параметр"], "rows": [["Ra"]]},
            )

        blocks = [
            text_block("Первый абзац про требования к валу.", order=0, bbox=[0.1, 0.10, 0.9, 0.15]),
            table(1, 0.20),
            text_block("Второй абзац, после первой таблицы.", order=2, bbox=[0.1, 0.30, 0.9, 0.35]),
            table(3, 0.40),
            text_block("Третий абзац, после второй таблицы.", order=4, bbox=[0.1, 0.50, 0.9, 0.55]),
        ]
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=blocks, parser_name="mineru", source_kind="scanned_pdf"), META
        )
        assert [f.type for f in fragments] == ["text", "table", "text", "table", "text"]
        assert fragments[0].content.text.startswith("Первый")
        assert fragments[2].content.text.startswith("Второй")
        assert fragments[4].content.text.startswith("Третий")

    def test_order_holds_without_coordinates(self):
        """
        У скана, разобранного только по `content_list`, bbox стоит на весь
        лист. Разрыв по вертикальному промежутку тут не срабатывает вовсе —
        порядок держится на разрыве цепочки по нетекстовому блоку.
        """
        blocks = [
            text_block("Вводный абзац документа.", order=0),
            ParsedBlock(
                type="formula", page=1, order=1, bbox=[0.0, 0.0, 1.0, 1.0],
                text="S = a \\cdot b",
            ),
            text_block("Пояснение к формуле выше.", order=2),
            ParsedBlock(
                type="table", page=1, order=3, bbox=[0.0, 0.0, 1.0, 1.0],
                table_data={"headers": ["Параметр"], "rows": [["Ra"]]},
            ),
            text_block("Заключительный абзац страницы.", order=4),
        ]
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=blocks, parser_name="mineru", source_kind="scanned_pdf"), META
        )
        assert [f.type for f in fragments] == ["text", "formula", "text", "table", "text"]


class TestDegradation:
    def test_degraded_parse_marks_every_fragment_for_review(self):
        """
        У страницы, прочитанной OCR, полнота 0.72 при пороге 0.5 — флага
        нет, документ выглядит нормальным. Деградация разбора должна
        отправлять на проверку независимо от полноты.
        """
        block = text_block("Вполне полный на вид текст страницы. " * 3,
                           bbox=[0.0, 0.0, 0.9, 0.9])
        result = ParseResult(
            blocks=[block], parser_name="plain_ocr", source_kind="image",
            degraded=["mineru_unavailable: нет связи с MinerU"],
        )
        _fragments, flags = TextNormalizer().normalize_with_flags(result, META)
        assert all(f["needs_review"] for f in flags)

    def test_clean_parse_is_not_marked(self):
        block = text_block("Вполне полный на вид текст страницы. " * 3,
                           bbox=[0.0, 0.0, 0.9, 0.9])
        result = ParseResult(blocks=[block], parser_name="mineru", source_kind="image")
        _fragments, flags = TextNormalizer().normalize_with_flags(result, META)
        assert not any(f["needs_review"] for f in flags)


class TestFlags:
    def test_needs_review_flag_is_not_in_contract(self):
        block = text_block("Текст в маленькой области листа. " * 2, bbox=[0.0, 0.0, 0.2, 0.2])
        fragments, flags = TextNormalizer().normalize_with_flags(
            ParseResult(blocks=[block], parser_name="mineru"), META
        )
        assert flags[0]["needs_review"] is True
        assert "needs_review" not in fragments[0].model_dump()


class TestParseExpression:
    def test_assignment(self):
        parsed = parse_expression("F = m * a")
        assert parsed.base_var == "F"
        assert parsed.operations[0]["value"] == "m * a"

    def test_not_an_assignment(self):
        assert parse_expression("a + b") is None
        assert parse_expression(None) is None
