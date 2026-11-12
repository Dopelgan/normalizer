"""
Регрессия (прод, 11–12.11.2026): авто-импорт админки отправил 122 файла в
карантин G-4 «Классификатор не уверен (уверенность 0.50), а разбор спорных
случаев не настроен — решение за администратором». 116 из них — сканы PDF
без текстового слоя: паспорта изделий и свидетельства о поверке «СБ210
зав.№№ …», программы контроля и испытаний «ПК …», «ПИ …», удостоверения о
ПИ, допсоглашения «ДС …». Правила не находили в имени ни одного маркера,
голый запасной «деловой, 0.50» без маркера не принимается, а модели G-4 в
контуре нет. Администратор вручную принял все такие файлы.

Имена ниже — реальные имена с прода. Отрицательные случаи (паспорт
гражданина, водительское удостоверение, безымянный скан) обязаны оставаться
вне базы знаний без решения администратора.
"""

import pytest

from core.gateway import rules
from core.gateway.profile import CATEGORY_BUSINESS, CATEGORY_TECHNICAL, GatewayProfile
from core.gateway.service import ACCEPT, DataGateway


class FakeStorage:
    def __init__(self, files):
        self.files = dict(files)

    def read_bytes(self, uri):
        return self.files[uri]

    def size(self, uri):
        return len(self.files[uri])


class SilentClassifier:
    """Модели нет — так работает прод."""

    def classify_document(self, path, sample):
        return None

    def classify_image(self, path, sample):
        return None

    def resolve_ambiguous(self, path, sample, company_profile):
        return None


def blank_pdf() -> bytes:
    """Скан: страницы есть, текстового слоя нет."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for _ in range(2):
        document.new_page(width=600, height=800)
    data = document.tobytes()
    document.close()
    return data


def evaluate_scan(name: str):
    path = f"s3://knowledge-base/imports/0f0e/{name}"
    gateway = DataGateway(
        profile=GatewayProfile(),
        storage=FakeStorage({path: blank_pdf()}),
        classifier=SilentClassifier(),
    )
    return gateway.evaluate(path, path, None)


PRODUCTION_SCANS = [
    ("СБ210 зав.№№ 220303-220317 св-ва о поверке.pdf", CATEGORY_TECHNICAL),
    ("СБ210 зав.№№ 220303-220317 паспорта.pdf", CATEGORY_TECHNICAL),
    ("СБ210 зав.№№ 220255-220274 св-ва о поверке Ростест.pdf", CATEGORY_TECHNICAL),
    ("КТК-02 зав.№№ 220071-220080 паспорта.pdf", CATEGORY_TECHNICAL),
    ("СП-02 зав.№ 220769 паспорт штамп ДЛЯ АЭС.pdf", CATEGORY_TECHNICAL),
    ("СМ-02 зав.№№ 220631 паспорт.pdf", CATEGORY_TECHNICAL),
    ("Паспорт зав№220263.pdf", CATEGORY_TECHNICAL),
    ("РТ-МП-514-442-2022__Методика поверки.pdf", CATEGORY_TECHNICAL),
    ("ПК 3.18.22 (ред 1).pdf", CATEGORY_TECHNICAL),
    ("ПК 3.18.22 + 3.19.22 (ред 0).pdf", CATEGORY_TECHNICAL),
    ("ПК РПР-ПК-НТЛ-6-2025.pdf", CATEGORY_TECHNICAL),
    ("ПИ 3.10.pdf", CATEGORY_TECHNICAL),
    ("ПИ 3.17.pdf", CATEGORY_TECHNICAL),
    ("Удостоверение РПР-ПК-НТЛ-11-2025.pdf", CATEGORY_TECHNICAL),
    ("Удостоверение о ПИ РПР-ПК-НТЛ-10-2025-ПИ.pdf", CATEGORY_TECHNICAL),
    ("СМ1638.11.00.00.00.00.pdf", CATEGORY_TECHNICAL),
    ("ДС 1 от 27.12.2022 с одной стороны.pdf", CATEGORY_BUSINESS),
    ("ДС1 от 25.07.2022.pdf", CATEGORY_BUSINESS),
    ("ДС3 место передачи.pdf", CATEGORY_BUSINESS),
    ("Приложение к ДС2.pdf", CATEGORY_BUSINESS),
    ("Уп лист 1_1 ДС-1.pdf", CATEGORY_BUSINESS),
    ("исх 1082-113 от 22.08.2022.pdf", CATEGORY_BUSINESS),
    ("Заключение_о_бесперспекивности_ПИР.pdf", CATEGORY_BUSINESS),
]


@pytest.mark.parametrize("name,category", PRODUCTION_SCANS)
def test_production_scan_is_accepted_by_name(name, category):
    verdict = evaluate_scan(name)
    assert verdict.outcome == ACCEPT, (name, verdict.layer, verdict.reason)
    assert verdict.category == category
    assert verdict.layer == "G-2"


@pytest.mark.parametrize("name", [
    "Паспорт гражданина РФ Иванов.pdf",
    "скан паспорта.pdf",
    "Паспорт.pdf",
    "Паспорт Иванова И.И..pdf",
    "Водительское удостоверение.pdf",
    "Водительское удостоверение РФ АВ 123456.pdf",
    "Удостоверение личности.pdf",
    "Свидетельство о рождении.pdf",
    "Свидетельство о браке Петровых.pdf",
    "scan_001.pdf",
    "Документ1.pdf",
    "Резюме Петров.pdf",
])
def test_personal_or_nameless_scan_is_not_accepted(name):
    """Без маркера принимаемой категории — решает администратор, а не приём."""
    verdict = evaluate_scan(name)
    assert verdict.outcome != ACCEPT, (name, verdict.layer, verdict.reason)


@pytest.mark.parametrize("value", [
    "ПК 12.05.24",          # дата сборки компьютера, а не шифр
    "пк 3.18.22",           # строчными — не шифр
    "ДСП 16 мм",            # не допсоглашение
    "НДС 20",               # не допсоглашение
    "удостоверение 5",      # голое удостоверение
])
def test_new_markers_do_not_fire(value):
    match = rules.classify(value, allowed=rules.PATH_CATEGORIES)
    assert not match.matched, (value, match.hits)


def test_official_letter_number_without_sign_yields_eskd():
    """Письмо со ссылкой на обозначение остаётся письмом (деловым)."""
    match = rules.classify("исх 1082-113 от 22.08.2022 по ТАДУ 305179.001.01 СБ")
    assert match.category == CATEGORY_BUSINESS
