"""
Классификация приёма: счёт по весам и разбор реальных спорных случаев.

Файлы, на которых классификатор не мог принять решение, присланы отдельным
набором. Сами файлы в репозиторий не кладутся — в них суммы договоров,
названия контрагентов и фамилии. Здесь воспроизведены их признаки: имена,
обороты писем и состав таблицы, то есть ровно то, по чему классификатор и
принимает решение.

Что было не так на этом наборе:

* четыре из семи — сканы PDF без текстового слоя. Текста нет, имя файла
  говорящее («Аттестат аккредитации», «Лицензия … (конструирование)»), но
  слов «аттестат» и «лицензия» в словарях не было вовсе;
* маркеры «исх. №» и «вх. №» не срабатывали никогда: подготовка строки
  заменяла точку пробелом, а шаблон точку требовал;
* коммерческий паспорт заказа с дюжиной упоминаний договора объявлялся
  технической документацией из-за одного слова «Сборочный»: побеждал
  первый сработавший маркер, а технические шли в списке первыми;
* книга .xlsm читалась не openpyxl, а как простой текст — то есть по
  случайной кириллице из сжатого потока.
"""

import io

import pytest

from core.gateway import rules, text_probe
from core.gateway.profile import (
    CATEGORY_BUSINESS,
    CATEGORY_CORRESPONDENCE,
    CATEGORY_TECHNICAL,
    GatewayProfile,
)
from core.gateway.service import ACCEPT, DataGateway


class FakeStorage:
    def __init__(self, files):
        self.files = dict(files)

    def read_bytes(self, uri):
        return self.files[uri]

    def size(self, uri):
        return len(self.files[uri])


class SilentClassifier:
    """Модели нет — решают правила. Ровно так работает штатная поставка."""

    def classify_document(self, path, sample):
        return None

    def classify_image(self, path, sample):
        return None

    def resolve_ambiguous(self, path, sample, company_profile):
        return None


def gateway(files):
    return DataGateway(
        profile=GatewayProfile(), storage=FakeStorage(files),
        classifier=SilentClassifier(),
    )


def blank_pdf(pages: int = 2) -> bytes:
    """Скан: страницы есть, текстового слоя нет."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for _ in range(pages):
        document.new_page(width=600, height=800)
    data = document.tobytes()
    document.close()
    return data


def pdf_with_text(*lines: str) -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    for row, line in enumerate(lines):
        page.insert_text((50, 60 + row * 22), line, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def docx_bytes(*lines: str) -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def xlsm_bytes(rows) -> bytes:
    """Книга с макросами: тот же формат, что xlsx, плюс проект VBA."""
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Паспорт"
    for row in rows:
        sheet.append(list(row))
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


# ===========================================================================
# Счёт по весам
# ===========================================================================

class TestWeightedScoring:
    def test_repeated_marker_outweighs_single_foreign_one(self):
        """
        Регресс. Коммерческий паспорт заказа: «договор» дюжину раз против
        одного «Сборочный». Раньше побеждал первый сработавший маркер, и
        решала не суть документа, а порядок списков в коде.
        """
        text = (
            "Паспорт заказа № 943-1. Заказчик. Стоимость договора. "
            "Договор поставки, дата договора, оригинал договора, "
            "доп соглашения к договору, цепочка договоров. Оплата после "
            "отгрузки. Сборочный узел."
        )
        match = rules.classify(text)
        assert match.category == CATEGORY_BUSINESS
        assert match.confidence >= 0.6
        assert match.scores[CATEGORY_BUSINESS] > match.scores[CATEGORY_TECHNICAL]

    def test_even_split_is_not_a_decision(self):
        """
        Поровну признаков двух категорий — это спорный случай. Уверенность
        обязана упасть ниже порога, чтобы файл ушёл на разбор (G-4), а не
        проскочил с выдуманным числом.
        """
        match = rules.classify("Чертёж вала. Договор поставки.")
        assert match.confidence < GatewayProfile().min_confidence

    def test_single_clear_marker_is_confident(self):
        match = rules.classify("Приказ № 17 об утверждении регламента")
        assert match.category == CATEGORY_BUSINESS
        assert match.confidence >= 0.8

    def test_signals_explain_the_decision(self):
        """Решение приёма должно быть проверяемым, а не принятым на веру."""
        signals = rules.classify("Настоящий договор поставки").as_signals()
        assert signals["matched_marker"] == "договор"
        assert CATEGORY_BUSINESS in signals["rule_scores"]
        assert signals["rule_hits"]["договор"] == 1

    def test_nothing_matched_is_empty(self):
        assert not rules.classify("qwerty 12345").matched
        assert not rules.classify("").matched


class TestMarkerRegressions:
    def test_outgoing_number_marker_works(self):
        """
        Регресс. Подготовка строки заменяла точку пробелом, а шаблон
        требовал «исх\\.», — маркер не срабатывал ни разу за всё время.
        """
        assert "исх. №" in rules.classify("Исх. № 1427-185 от 10.09.2026").hits
        assert "исх. №" in rules.classify("Исх№ 1427-185").hits

    def test_underscores_still_split_words(self):
        assert rules.classify("личное_отпуск.jpg").matched

    def test_official_letter_is_a_business_document(self):
        """
        Служебное письмо с исходящим номером подшивается в дело и хранится
        наравне с приказом. «Переписка» оставлена почтовым веткам.
        """
        match = rules.classify(
            "Исх. № 1427-185 от 10.09.2026. Уважаемые коллеги! "
            "Об участии в инспекции. Генеральный директор."
        )
        assert match.category == CATEGORY_BUSINESS

    def test_mail_thread_is_still_correspondence(self):
        match = rules.classify(
            "Re: смета\nКому: ООО Ромашка\nЗдравствуйте!\nС уважением, Иванов"
        )
        assert match.category == CATEGORY_CORRESPONDENCE

    def test_path_cannot_declare_correspondence(self):
        match = rules.classify(
            "documents/переписка/приложение 2.pdf", allowed=rules.PATH_CATEGORIES
        )
        assert match.category != CATEGORY_CORRESPONDENCE


# ===========================================================================
# Присланный набор спорных файлов
# ===========================================================================

class TestUnsureSamples:
    """Семь файлов, на которых классификатор не мог принять решение."""

    @pytest.mark.parametrize("name", [
        "Аттестат аккредитации.pdf",
        "Лицензия ЦО-11-101-14790 (конструирование).pdf",
        "ДИ023-24 Инженер-конструктор.pdf",
        "Заявка от 11.02.2026 №3.pdf",
    ])
    def test_scan_without_text_is_decided_by_name(self, name):
        """
        У скана текстового слоя нет, и содержимого для правил тоже нет:
        решает имя файла. Раньше эти четыре уходили в карантин с
        формулировкой «классификатор не уверен».
        """
        data = blank_pdf()
        uri = f"documents/{name}"
        verdict = gateway({uri: data}).evaluate(name, uri, len(data))
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.outcome == ACCEPT
        assert verdict.confidence >= GatewayProfile().min_confidence
        assert verdict.signals["matched_in"] == "путь"

    def test_letter_with_outgoing_number(self):
        data = docx_bytes(
            "Об участии в инспекции",
            "Исх. № 1427-185 от 10.09.2026",
            "Уважаемые коллеги!",
            "Генеральный директор",
        )
        uri = "documents/Исх№ 1427-185 Об участии в инспекции.docx"
        verdict = gateway({uri: data}).evaluate("письмо.docx", uri, len(data))
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.outcome == ACCEPT
        assert verdict.signals["matched_in"] == "текст"

    def test_letter_about_packaging(self):
        data = pdf_with_text(
            "ООО «ТД Групп» ИНН 7839103384 / КПП 771501001",
            "Письмо № ТДГ-352 от 09.04.2026",
            "Уважаемый Дмитрий Владимирович!",
            "Заместитель генерального директора",
        )
        uri = "documents/Письмо №ТДГ-352 О требованиях к упаковке.pdf"
        verdict = gateway({uri: data}).evaluate("письмо.pdf", uri, len(data))
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.outcome == ACCEPT

    def test_order_passport_is_business_not_technical(self):
        """
        Коммерческий паспорт заказа: договор, суммы, сроки оплаты — и одно
        слово «Сборочный» в номенклатуре. Раньше выигрывало оно.
        """
        data = xlsm_bytes([
            ("Паспорт заказа № 943-1", "Согласование ТУ"),
            ("Заказчик", "ООО НПП"),
            ("Стоимость договора", 199268450),
            ("Договор", "№ОСН0925-2"),
            ("Дата договора", "2025-09-03"),
            ("Оригинал договора", "получен"),
            ("Доп соглашения к договору", "нет"),
            ("Оплата после отгрузки", "180 дней"),
            ("План качества", "требуется"),
            ("Сборочный узел", "поз. 4"),
        ])
        uri = "documents/Проект паспорта.xlsm"
        verdict = gateway({uri: data}).evaluate("паспорт.xlsm", uri, len(data))
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.outcome == ACCEPT
        assert verdict.signals["text_source"] == "xlsx"


# ===========================================================================
# Чтение таблиц
# ===========================================================================

class TestSpreadsheetProbe:
    def test_xlsm_is_read_by_openpyxl(self):
        """
        Регресс. Отбор шёл по `file_type == "xlsx"`, и xlsm, xls и ods
        уезжали в чтение как простой текст: zip и BIFF декодировались
        в cp1251, давая случайную кириллицу и случайные же совпадения.
        """
        data = xlsm_bytes([("Договор поставки", "№ 12")])
        probe = text_probe.extract("xlsm", data)
        assert probe.source == text_probe.SOURCE_XLSX
        assert "Договор поставки" in probe.text

    def test_ods_is_read_as_xml(self):
        probe = text_probe.extract("ods", _ods_bytes("Договор поставки"))
        assert probe.source == text_probe.SOURCE_ODS
        assert "Договор поставки" in probe.text

    def test_csv_is_still_plain_text(self):
        probe = text_probe.extract("csv", "договор;поставки\n1;2\n".encode("utf-8"))
        assert probe.source == text_probe.SOURCE_PLAIN


def _ods_bytes(value: str) -> bytes:
    import zipfile

    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content '
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        f"<office:body><text:p>{value}</text:p></office:body>"
        "</office:document-content>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "mimetype", "application/vnd.oasis.opendocument.spreadsheet"
        )
        archive.writestr("content.xml", content)
    return buffer.getvalue()
