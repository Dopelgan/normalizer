"""Geometry and source readings are the gold, never a global space/letter rewrite."""
import hashlib
from types import SimpleNamespace

import pytest

from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from core.providers import mixed_script, text_layer


@pytest.fixture(autouse=True)
def structural_v2(monkeypatch):
    """Эти проверки — про профиль structural-rag-v2 (флаг STRUCTURAL_RAG_V2)."""
    from core.config import settings
    monkeypatch.setattr(settings, "STRUCTURAL_RAG_V2", True)


def span(text, left, right, size=10):
    return {'text':text,'bbox':[left,10,right,20],'size':size,'font':'Example','color':0}


def read_line(*spans):
    page = SimpleNamespace(rotation=0, get_text=lambda kind:{'blocks':[{
        'type':0,'lines':[{'spans':list(spans),'bbox':[10,10,180,20],'dir':[1,0]}]}]})
    return text_layer._lines_from_page(page,1,200,100)[0]


def test_spatial_word_gap_recovers_boundary_without_splitting_a_font_run():
    separated = read_line(span('Поставка',10,50),span('разрешена',54,110))
    joined = read_line(span('По',10,20),span('ставка',20,50))
    assert separated.text == 'Поставка разрешена'
    assert joined.text == 'Поставка'
    assert separated.raw_text == 'Поставкаразрешена'
    edit = separated.text_edits[0]
    assert edit['source_start'] == edit['source_end'] == len('Поставка')
    assert edit['replacement'] == ' '
    assert separated.text[edit['output_start']:edit['output_end']] == ' '
    assert not joined.text_edits


def test_existing_nbsp_units_decimals_hyphen_and_identifiers_survive():
    assert read_line(span('12,5\u00a0',10,36),span('мм',36,50)).text == '12,5\u00a0мм'
    assert read_line(span('12,',10,28),span('5',28,33)).text == '12,5'
    assert read_line(span('12,5',10,33),span('мм',37,50)).text == '12,5 мм'
    assert read_line(span('IP',10,20),span('67',20,31)).text == 'IP67'
    assert read_line(span('кабель-',10,45),span('канал',45,70)).text == 'кабель-канал'


def test_geometry_correction_observation_survives_block_and_fragment():
    line = read_line(span('До',10,20),span('согласования',24,85))
    block = ParsedBlock(type='text',page=1,text='')
    text_layer._rebuild_text_block(block,[line],SimpleNamespace(rebuilt_blocks=0))
    fragment = TextNormalizer().normalize(ParseResult(blocks=[block],exhaustive=True),
        {'doc_id':'source','s3_fileid':'source.pdf'})[0]
    observed = fragment.content.structured_payload['text_observations'][0]
    assert observed['raw'] == 'Досогласования'
    assert observed['normalized'] == 'До согласования'
    assert observed['rule'] == 'pdf-span-word-gap-v1'
    assert observed['page'] == 1 and observed['bbox'] == line.bbox
    assert observed['edits'][0]['source_start'] == 2


def test_long_page_reconstruction_keeps_the_middle_clause_and_word_boundaries():
    prefix = 'Технические требования и условия поставки. ' * 45
    clause = 'Отгрузка без согласования запрещена.'
    suffix = 'Перечень документов сохраняется полностью. ' * 45
    source = prefix + clause + ' ' + suffix
    # A high character ratio is insufficient: the evaluator requires the actual clause.
    incomplete = prefix + suffix
    assert len(incomplete) / len(source) > 0.98
    assert clause not in incomplete
    line = read_line(span(source,10,180))
    block = ParsedBlock(type='text',page=1,text=incomplete)
    text_layer._rebuild_text_block(block,[line],SimpleNamespace(rebuilt_blocks=0))
    assert block.text.count(clause) == 1
    assert 'согласования запрещена' in block.text
    assert block.text == source.rstrip()


def test_mixed_script_audit_locates_repeated_occurrences_without_touching_notation():
    from core.models.contract import Content,Fragment,Position,Provenance
    raw = 'Код ККS; ККS; English IP67 Rизм 12X18H10T'
    fragment = Fragment(fragment_id='source',type='text',content=Content(text=raw),
        position=Position(page=3),confidence=1,completeness=1,
        provenance=Provenance(method='text_layer',strategy_level=2,source='vector_pdf'))
    mixed_script.repair_fragment(fragment)
    assert fragment.content.text == 'Код KKS; KKS; English IP67 Rизм 12X18H10T'
    evidence = fragment.content.structured_payload['text_correction_evidence']
    assert len(evidence) == 1
    audit = evidence[0]
    # Полный исходный текст в свидетельство не копируется: хватает мест и хеша.
    assert 'raw' not in audit and 'normalized' not in audit
    assert audit['raw_sha256'] == hashlib.sha256(raw.encode('utf-8')).hexdigest()
    assert audit['raw_length'] == len(raw)
    assert audit['page'] == 3 and audit['rule'] == 'mixed-script-v1'
    assert len(audit['edits']) == 2
    for edit in audit['edits']:
        assert raw[edit['source_start']:edit['source_end']] == edit['original'] == 'ККS'
        assert edit['replacement'] == 'KKS'
    assert audit['edits'][0]['source_start'] != audit['edits'][1]['source_start']
    # Исходный текст восстанавливается из выданного текста и правок.
    restored = fragment.content.text
    for edit in reversed(audit['edits']):
        start = edit['source_start']
        restored = restored[:start] + edit['original'] + restored[start + len(edit['replacement']):]
    assert restored == raw


def test_repeated_mixed_script_repair_does_not_duplicate_evidence():
    from core.models.contract import Content,Fragment,Position,Provenance
    raw = 'Код ККS'
    fragment = Fragment(fragment_id='source',type='text',content=Content(text=raw),
        position=Position(page=1),confidence=1,completeness=1,
        provenance=Provenance(method='text_layer',strategy_level=2,source='vector_pdf'))
    mixed_script.repair_fragment(fragment)
    mixed_script.repair_fragment(fragment)
    # Повтор по исходному тексту (например, при повторной сборке) — та же запись.
    fragment.content.text = raw
    mixed_script.repair_fragment(fragment)
    assert len(fragment.content.structured_payload['text_correction_evidence']) == 1
    assert fragment.content.text == 'Код KKS'
