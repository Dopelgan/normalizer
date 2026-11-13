"""
Правила классификации приёма (слой G-2): счёт по весам вместо первого
совпадения.

Раньше категорию решал первый сработавший маркер в жёстком порядке
«техническое → деловое → переписка → личное». Одно случайное слово решало
судьбу документа: в коммерческом паспорте заказа, где «договор» встречается
дюжину раз, слово «Сборочный» из одной строки делало файл технической
документацией.

Здесь считается вес. У каждого маркера своя цена, повторы поднимают её по
логарифму (дюжина «договоров» весомее одного «сборочного», но не в
двенадцать раз), а уверенность берётся из отрыва лидера от второго места:
две категории с равным счётом — это не решение, а спорный случай, и он
обязан уйти на разбор, а не проскочить с выдуманной уверенностью.

Модуль намеренно не зависит ни от чего, кроме стандартной библиотеки:
словари правят чаще, чем код вокруг них, и проверка правил должна быть
дешёвой.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set, Tuple

# Категории слоя G-2. Продублированы из core.gateway.profile намеренно:
# модуль правил не должен тянуть за собой настройки и pydantic.
CATEGORY_BUSINESS = "business_document"
CATEGORY_TECHNICAL = "technical_documentation"
CATEGORY_CORRESPONDENCE = "correspondence"
CATEGORY_PERSONAL = "personal"

# Порядок при равном счёте: у двух категорий с одинаковым весом побеждает
# та, что выше в списке. Нужен, чтобы результат не зависел от порядка
# обхода словаря.
CATEGORY_ORDER: Tuple[str, ...] = (
    CATEGORY_TECHNICAL,
    CATEGORY_BUSINESS,
    CATEGORY_CORRESPONDENCE,
    CATEGORY_PERSONAL,
)

# Категории, которые нельзя объявить по одному лишь пути. Путь — слабый
# признак: каталог «Договоры и ДС» говорит о деловом документе, но каталог
# «Переписка» не делает перепиской лежащее в нём приложение к договору.
PATH_FORBIDDEN_CATEGORIES: Tuple[str, ...] = (CATEGORY_CORRESPONDENCE,)

# Категории, которые вообще можно вывести из пути.
PATH_CATEGORIES: Tuple[str, ...] = tuple(
    c for c in CATEGORY_ORDER if c not in PATH_FORBIDDEN_CATEGORIES
)

# Во сколько раз повтор маркера может поднять его вес. Дюжина «договоров»
# весомее одного, но не в двенадцать раз: иначе длинный документ побеждает
# короткий одним лишь объёмом.
_REPEAT_CAP = 3.0

# Сколько совпадений маркера ещё считается: дальше вес упёрся в потолок, а
# перебор на документе в двадцать тысяч знаков стоит времени.
_COUNT_LIMIT = 16


@dataclass(frozen=True)
class Marker:
    """Признак категории: что искать и сколько это стоит."""

    category: str
    weight: float
    pattern: "re.Pattern"
    title: str

    def count(self, haystack: str) -> int:
        found = 0
        for _ in self.pattern.finditer(haystack):
            found += 1
            if found >= _COUNT_LIMIT:
                break
        return found


def _marker(category: str, weight: float, pattern: str, title: str = "") -> Marker:
    return Marker(
        category=category,
        weight=weight,
        pattern=re.compile(pattern, re.IGNORECASE),
        title=title or pattern,
    )


# ===========================================================================
# Словари
# ===========================================================================
# Вес 1.0 — слово, которое само по себе называет род документа («договор»,
# «чертёж», «лицензия»). Вес 0.5 и ниже — признак, который встречается и в
# чужих документах («акт», «график», «деталь»): он подтверждает категорию,
# но не объявляет её в одиночку.

# Обозначение изделия по ЕСКД (ГОСТ 2.201): код организации — три-четыре
# заглавные буквы, затем группы цифр через точку. Форматов на проде два:
# «ТАДУ 687000.001», «ТАДУ 305179.001.01.01.011», «АДУ 305179.001.15.02» —
# классификационная характеристика из шести цифр; и «АЕГН.01.00017.111.0015» —
# короткие группы, но не меньше четырёх. Сканы КД приходят без текстового
# слоя, и обозначение в имени — единственный их признак.
#
# Регистр важен намеренно: «июня 12.05.20» и «от 12.05.2024» — даты. Коды
# классификаторов («ОКПД 25.11.23.110»), стандартов и камер («DSCN 0012.034»)
# под формат подходят, поэтому исключены явно.
_ESKD_STOP = (
    "ОКПД", "ОКПО", "ОКВЭД", "ОКАТО", "ОКТМО", "ОКОФ", "ОКУД", "ОКВ", "ОКС",
    "ГОСТ", "СНИП", "СТО", "ОСТ", "РСТ", "DSCN", "DSC", "IMG", "PIC", "SAM",
)
ESKD_DESIGNATION = Marker(
    category=CATEGORY_TECHNICAL,
    weight=1.2,
    pattern=re.compile(
        r"(?<![А-ЯЁA-Zа-яёa-z\d])"
        r"(?!(?:" + "|".join(_ESKD_STOP) + r")[ .\d])"
        r"(?:"
        r"[А-ЯЁA-Z]{3,4}[ .]"
        r"(?:\d{6}(?:\.\d{2,5}){1,6}|\d{2,5}(?:\.\d{2,5}){3,6})"
        # Слитная запись без разделителя после кода: «СМ1638.11.00.00.00.00».
        # Код бывает из двух букв, поэтому групп после номера не меньше
        # четырёх — так не ловятся ни даты, ни версии.
        r"|[А-ЯЁA-Z]{2,4}\d{3,6}(?:\.\d{2,3}){4,6}"
        # Обозначение технологического документа (ГОСТ 3.1201): код
        # организации, затем пять и пять цифр — «ТАДУ 25290.00022». Ровно
        # четыре буквы и ровно две группы по пять: ни дат, ни телефонов.
        r"|[А-ЯЁA-Z]{4}[ .]\d{5}\.\d{5}"
        r")"
        r"(?![\d])"
    ),
    title="обозначение ЕСКД",
)

# Код документа КД после обозначения (ГОСТ 2.102): «СБ», «ПЭ3», «РЭ»… и номер
# изменения. Сами по себе ничего не значат — «Подстанция ПС 110 кВ», «изм. 3»
# в протоколе, — поэтому считаются только рядом с обозначением (см. classify).
# Латинская «C» в «CБ» встречается в реальных именах файлов.
ESKD_DOCUMENT_CODE = Marker(
    category=CATEGORY_TECHNICAL,
    weight=0.4,
    pattern=re.compile(
        r"(?<![А-ЯЁA-Zа-яёa-z])(?:[СC]Б|ПМ|ТУ|П?Э[3З]|ВП|РЭ|ПС|ФО|МЧ|ГЧ|КД)(?![А-ЯЁA-Zа-яёa-z])"
    ),
    title="код документа КД",
)
ESKD_REVISION = _marker(CATEGORY_TECHNICAL, 0.3, r"\bизм\.?\s*\d+", "изм. N")

# Маркеры, рядом с которыми обозначение ЕСКД не считается вовсе. Личное в
# имени важнее шифра («личное_ТАДУ 687000.001.pdf» — не КД), а письмо или
# почтовая ветка, ссылающиеся на чертёж, остаются письмом.
_ESKD_YIELDS_TO_TITLES = frozenset({"исх. №", "исх. номер от", "вх. №", "письмо №"})

_TECHNICAL: Tuple[Marker, ...] = (
    ESKD_DESIGNATION,
    ESKD_DOCUMENT_CODE,
    ESKD_REVISION,
    _marker(CATEGORY_TECHNICAL, 1.2, r"\bгост\b", "ГОСТ"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bост\s*\d", "ОСТ"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bту\s*\d", "ТУ"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bчерт[её]ж\w*", "чертёж"),
    _marker(CATEGORY_TECHNICAL, 0.8, r"\bспецификац\w+", "спецификация"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bтехническ\w+\s+услови\w+", "технические условия"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bтехнологическ\w+\s+процесс\w*", "техпроцесс"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bруководств\w+\s+по\s+эксплуатац\w+", "руководство по эксплуатации"),
    _marker(CATEGORY_TECHNICAL, 0.9, r"\bизвещени\w+\s+об\s+изменени\w+", "извещение об изменении"),
    _marker(CATEGORY_TECHNICAL, 0.8, r"\bшероховат\w+", "шероховатость"),
    _marker(CATEGORY_TECHNICAL, 0.8, r"\bсборочн\w+\s+черт\w*", "сборочный чертёж"),
    _marker(CATEGORY_TECHNICAL, 0.8, r"\bмаршрутн\w+\s+карт\w+", "маршрутная карта"),
    _marker(CATEGORY_TECHNICAL, 0.7, r"\bэскиз\w*", "эскиз"),
    _marker(CATEGORY_TECHNICAL, 0.5, r"\bтехническ\w+", "технический"),
    _marker(CATEGORY_TECHNICAL, 0.5, r"\bдопуск\w*", "допуск"),
    _marker(CATEGORY_TECHNICAL, 0.4, r"\bсборочн\w+", "сборочный"),
    _marker(CATEGORY_TECHNICAL, 0.4, r"\bдеталь\w*", "деталь"),
    _marker(CATEGORY_TECHNICAL, 0.4, r"\bузел\b", "узел"),
    _marker(CATEGORY_TECHNICAL, 0.4, r"\bсхема\w*", "схема"),
    # Документы качества и метрологии. Приходят сканами без текстового слоя,
    # и имя файла — единственный их признак: «СБ210 зав.№№ 220303-220317
    # св-ва о поверке.pdf», «ПК 3.18.22 (ред 1).pdf», «ПИ 3.10.pdf». Без них
    # правила давали голый «деловой, 0.50», и 116 обычных документов
    # предприятия ушли в карантин G-4 (прод, 11–12.11.2026).
    #
    # «Паспорт» отдельно не маркер: так же называется документ гражданина.
    # Паспорт изделия узнаётся по заводскому номеру рядом или по «паспорт
    # изделия».
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bзав(?:\.|одск\w*)?\s*(?:№|номер)", "заводской номер"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bпаспорт\w*\s+(?:на\s+)?издели\w*", "паспорт изделия"),
    _marker(
        CATEGORY_TECHNICAL, 1.0,
        r"\bсв(?:идетельств\w*|\s*ва|\.)\s+о\s+(?:поверк|калибровк)\w*",
        "свидетельство о поверке",
    ),
    _marker(
        CATEGORY_TECHNICAL, 1.0,
        r"\bметодик\w*\s+(?:поверк|калибровк|измерени|испытани|контрол)\w*",
        "методика поверки",
    ),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bпрограмм\w*\s+(?:и\s+методик\w*\s+)?(?:контрол|испытани)\w*", "программа контроля/испытаний"),
    # «ПК 3.18.22», «ПИ 3.10», «ПК РПР-ПК-НТЛ-6-2025» — шифры программы
    # контроля и программы испытаний. Регистр важен: «пк» строчными —
    # компьютер; «ПК 12.05.24» — дата, а не шифр (поэтому и «ПК 3.10.22»,
    # похожий на дату, остаётся администратору).
    Marker(
        category=CATEGORY_TECHNICAL,
        weight=1.0,
        pattern=re.compile(
            r"(?<![А-ЯЁA-Zа-яёa-z\d])П[КИ]\s*"
            r"(?:"
            r"(?!(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.\d{2,4}(?!\.?\d))"
            r"\d{1,3}(?:\.\d{1,3})+(?!\.?\d)"
            r"|РПР\b"
            r")"
        ),
        title="ПК/ПИ (шифр программы)",
    ),
    # Удостоверение о качестве / о ПИ с шифром — документ изделия. Голое
    # «удостоверение» не маркер: водительское, личности.
    Marker(
        category=CATEGORY_TECHNICAL,
        weight=1.0,
        pattern=re.compile(
            r"(?<![А-ЯЁA-Zа-яёa-z])[Уу]достоверени\w*\s+"
            r"(?:о\s+(?:ПИ|качеств\w*|приёмк\w*|приемк\w*)\b|[А-ЯЁ]{2,4}\s+П[КИ]\s+[А-ЯЁ]{2,4}\s+\d)"
        ),
        title="удостоверение о качестве/ПИ",
    ),
    # Технологическая документация (ГОСТ 3.1102): «КТД на ремонт» —
    # комплект технологической документации, «КТПС», «Карта ТПС» — карта
    # технологического процесса сварки. Регистр важен.
    Marker(
        category=CATEGORY_TECHNICAL,
        weight=1.0,
        pattern=re.compile(
            r"(?<![А-ЯЁA-Zа-яёa-z\d])(?:КТД|КТПС|[Кк]арт\w*\s+ТПС)(?![А-ЯЁA-Zа-яёa-z])"
        ),
        title="КТД/КТПС",
    ),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bтехнологическ\w+\s+карт\w*", "технологическая карта"),
    # «СТО 06.18» — стандарт организации. Регистр важен; «СТО 12.05.24» —
    # дата (станция техобслуживания), поэтому похожее на дату исключено,
    # как у шифров ПК/ПИ.
    Marker(
        category=CATEGORY_TECHNICAL,
        weight=1.0,
        pattern=re.compile(
            r"(?<![А-ЯЁA-Zа-яёa-z\d])СТО\s+"
            r"(?!(?:0?[1-9]|[12]\d|3[01])\.(?:0?[1-9]|1[0-2])\.\d{2,4}(?!\.?\d))"
            r"\d{2}(?:\.\d{2,3})+(?!\.?\d)"
        ),
        title="СТО (стандарт организации)",
    ),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bруководств\w+\s+по\s+качеств\w+", "руководство по качеству"),
    # «СТП ТАДУ 15.01-25…» — стандарт предприятия. Регистр важен.
    Marker(
        category=CATEGORY_TECHNICAL,
        weight=1.0,
        pattern=re.compile(r"(?<![А-ЯЁA-Zа-яёa-z\d])СТП\s+[А-ЯЁ]{2,4}\b"),
        title="СТП",
    ),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\bquality\s+(?:assurance|control|management|plan|manual|program\w*)", "quality assurance"),
    _marker(CATEGORY_TECHNICAL, 1.0, r"\b(?:verification|calibration)\s+(?:certificate|procedure|report|method\w*)|\b(?:test|inspection)\s+(?:procedure|program\w*|certificate)", "test/verification document"),
)

_BUSINESS: Tuple[Marker, ...] = (
    # Договорная и распорядительная часть.
    _marker(CATEGORY_BUSINESS, 1.0, r"\bдоговор\w*", "договор"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bконтракт\w*", "контракт"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bприказ\w*", "приказ"),
    _marker(CATEGORY_BUSINESS, 0.9, r"\bраспоряжени\w+", "распоряжение"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bрегламент\w*", "регламент"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bустав\b", "устав"),
    _marker(CATEGORY_BUSINESS, 0.9, r"\bсоглашени\w+", "соглашение"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bинструкц\w+", "инструкция"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bдолжностн\w+\s+инструкц\w+", "должностная инструкция"),
    # «ДИ023-24» в имени файла — та же должностная инструкция, только
    # сокращённая до шифра. Сканы приходят именно так, и без этого правила
    # они не опознаются вовсе: текстового слоя у них нет.
    _marker(CATEGORY_BUSINESS, 0.9, r"\bди\s*-?\s*\d{2,}", "ДИ (шифр)"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bтрудов\w+\s+договор", "трудовой договор"),
    _marker(CATEGORY_BUSINESS, 0.6, r"\bположение\b", "положение"),
    # Разрешительные документы организации. Их не было вовсе, а приходят они
    # сканами без текстового слоя — только имя файла и говорит, что это.
    _marker(CATEGORY_BUSINESS, 1.0, r"\bлицензи\w+", "лицензия"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bаттестат\w*", "аттестат"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bаккредитац\w+", "аккредитация"),
    _marker(CATEGORY_BUSINESS, 0.9, r"\bсертификат\w*", "сертификат"),
    # Свидетельство о рождении, браке, смерти — документ человека, не
    # организации: такой скан решает администратор.
    _marker(
        CATEGORY_BUSINESS, 0.9,
        r"\bсвидетельств\w*\b(?!\s+об?\s+(?:рождени|заключени\w*\s+брак|брак|расторжени|смерт|"
        r"перемен\w*\s+имен|установлени\w*\s+отцовств|усыновл))",
        "свидетельство",
    ),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bразрешени\w+", "разрешение"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bдеклараци\w+\s+о\s+соответств\w+", "декларация о соответствии"),
    # Первичные и расчётные документы.
    _marker(CATEGORY_BUSINESS, 1.0, r"\bсч[ёе]т[- ]фактур\w*", "счёт-фактура"),
    _marker(CATEGORY_BUSINESS, 0.9, r"\bнакладн\w+", "накладная"),
    _marker(CATEGORY_BUSINESS, 0.9, r"\bупд\b", "УПД"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bсч[ёе]т\b", "счёт"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bсмет\w+", "смета"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bкалькуляц\w+", "калькуляция"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bплат[её]жн\w+\s+поручени\w+", "платёжное поручение"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bкоммерческ\w+\s+предложени\w+", "коммерческое предложение"),
    _marker(CATEGORY_BUSINESS, 0.6, r"\bзаказчик\w*", "заказчик"),
    _marker(CATEGORY_BUSINESS, 0.5, r"\bпоставщик\w*", "поставщик"),
    _marker(CATEGORY_BUSINESS, 0.5, r"\bоплат\w+", "оплата"),
    _marker(CATEGORY_BUSINESS, 0.5, r"\bинн\b", "ИНН"),
    _marker(CATEGORY_BUSINESS, 0.4, r"\bкпп\b", "КПП"),
    # Служебная переписка организации — её деловой документ, а не личная
    # почта: письмо с исходящим номером подшивается в дело и хранится
    # наравне с приказом. Категория «переписка» оставлена почтовым веткам.
    _marker(CATEGORY_BUSINESS, 1.0, r"\bисх\.?\s*(?:№|n\b)", "исх. №"),
    _marker(CATEGORY_BUSINESS, 1.0, r"\bвх\.?\s*(?:№|n\b)", "вх. №"),
    _marker(CATEGORY_BUSINESS, 0.9, r"\bписьм\w*\s*№", "письмо №"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bуведомлени\w+", "уведомление"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bпретензи\w+", "претензия"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bдоверенност\w+", "доверенность"),
    _marker(CATEGORY_BUSINESS, 0.4, r"\bуважаем(?:ый|ая|ые)\b", "уважаемый (обращение)"),
    _marker(CATEGORY_BUSINESS, 0.3, r"\bдиректор\w*", "директор"),
    # Заявки, заявления, отчётность.
    _marker(CATEGORY_BUSINESS, 0.9, r"\bзаявк\w+", "заявка"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bзаявлени\w+", "заявление"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bпротокол\w*", "протокол"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bотч[ёе]т\w*", "отчёт"),
    _marker(CATEGORY_BUSINESS, 0.8, r"\bтехническ\w+\s+задани\w+", "техническое задание"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bтз\b", "ТЗ"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bплан\w*\s+качеств\w+", "план качества"),
    _marker(CATEGORY_BUSINESS, 0.7, r"\bпаспорт\w*\s+заказ\w+", "паспорт заказа"),
    _marker(CATEGORY_BUSINESS, 0.6, r"\bинспекц\w+", "инспекция"),
    _marker(CATEGORY_BUSINESS, 0.6, r"\bреестр\w*", "реестр"),
    _marker(CATEGORY_BUSINESS, 0.6, r"\bприложение\s*№?\s*\d", "приложение №"),
    _marker(CATEGORY_BUSINESS, 0.5, r"\bакт\b", "акт"),
    _marker(CATEGORY_BUSINESS, 0.4, r"\bграфик\b", "график"),
    _marker(CATEGORY_BUSINESS, 0.4, r"\bведомост\w+", "ведомость"),
    # «ДС1», «ДС 1 от 27.12.2022», «Приложение к ДС2» — дополнительное
    # соглашение к договору. Регистр важен.
    Marker(
        category=CATEGORY_BUSINESS,
        weight=1.0,
        # «Справка ДС 15», «Квитанция ДС 5» — детский сад; «ДС 12.05.2024» — дата.
        pattern=re.compile(
            r"(?<![А-ЯЁA-Zа-яёa-z\d])(?<!(?i:справка) )(?<!(?i:квитанция) )"
            r"ДС\s*\d{1,3}(?!\d)(?!\.\d)"
        ),
        title="ДС (доп. соглашение)",
    ),
    # «исх 1082-113 от 22.08.2022» — исходящий номер без знака «№».
    _marker(CATEGORY_BUSINESS, 1.0, r"\bисх\.?\s*\d[\d\s]*от\s+\d", "исх. номер от"),
    # Только заключения о проекте/работе: «заключение о состоянии здоровья» —
    # документ человека.
    _marker(
        CATEGORY_BUSINESS, 0.8,
        r"\bзаключени\w*\s+о\s+(?:бесперспек|нецелесообразн|целесообразн|соответстви|"
        r"техническ|возможност|результат)\w*",
        "заключение о",
    ),
)

# Переписка — это почтовая ветка, а не любое письмо. Служебное письмо с
# исходящим номером считается деловым документом (см. выше). Сюда попадает
# то, что выгружено из почтового клиента или мессенджера.
_CORRESPONDENCE: Tuple[Marker, ...] = (
    _marker(CATEGORY_CORRESPONDENCE, 1.2, r"(?:^|\n)\s*(?:re|fwd|fw)\s*:", "Re:/Fwd:"),
    _marker(CATEGORY_CORRESPONDENCE, 1.0, r"\bкому\s*:", "Кому:"),
    _marker(CATEGORY_CORRESPONDENCE, 1.0, r"\bот\s+кого\s*:", "От кого:"),
    _marker(CATEGORY_CORRESPONDENCE, 1.0, r"\bтема\s+письма\s*:", "Тема письма:"),
    _marker(CATEGORY_CORRESPONDENCE, 1.0, r"\bпереслан\w+\s+сообщени\w+", "пересланное сообщение"),
    _marker(CATEGORY_CORRESPONDENCE, 0.9, r"\bпереписк\w+", "переписка"),
    _marker(CATEGORY_CORRESPONDENCE, 0.8, r"\bс\s+уважением\b", "с уважением"),
    _marker(CATEGORY_CORRESPONDENCE, 0.7, r"\bдобр\w+\s+(?:день|вечер|утро)\b", "добрый день"),
    _marker(CATEGORY_CORRESPONDENCE, 0.7, r"\bздравствуйте\b", "здравствуйте"),
)

_PERSONAL: Tuple[Marker, ...] = (
    _marker(CATEGORY_PERSONAL, 1.0, r"\bличное\b", "личное"),
    _marker(CATEGORY_PERSONAL, 1.0, r"\bсвадьб\w+", "свадьба"),
    _marker(CATEGORY_PERSONAL, 1.0, r"\bдень\s+рождения\b", "день рождения"),
    _marker(CATEGORY_PERSONAL, 0.9, r"\bкорпоратив\w*", "корпоратив"),
    _marker(CATEGORY_PERSONAL, 0.8, r"\bфото\s+с\b", "фото с"),
    _marker(CATEGORY_PERSONAL, 0.7, r"\bсемья\b", "семья"),
    # «Отпуск» сам по себе личным документ не делает: заявление на отпуск —
    # кадровый документ, и деловые маркеры весят больше.
    _marker(CATEGORY_PERSONAL, 0.7, r"\bотпуск\w*", "отпуск"),
)

MARKERS: Tuple[Marker, ...] = _TECHNICAL + _BUSINESS + _CORRESPONDENCE + _PERSONAL


# ===========================================================================
# Счёт
# ===========================================================================

@dataclass
class RuleMatch:
    """Итог разбора правил: кто победил, с каким отрывом и почему."""

    category: Optional[str] = None
    confidence: float = 0.0
    marker: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    hits: Dict[str, int] = field(default_factory=dict)

    @property
    def matched(self) -> bool:
        return self.category is not None

    def as_signals(self) -> Dict[str, object]:
        """То, что уходит в сигналы приёма: решение можно проверить."""
        if not self.matched:
            return {}
        return {
            "matched_marker": self.marker,
            "rule_scores": {
                key: round(value, 3)
                for key, value in sorted(self.scores.items(), key=lambda kv: -kv[1])
            },
            "rule_hits": dict(sorted(self.hits.items(), key=lambda kv: -kv[1])),
        }


def prepare(value: str) -> str:
    """
    Строка, пригодная для поиска по границам слов.

    В именах файлов слова разделяют не пробелы, а `_`, `-` и слэши, причём
    `_` для регулярного выражения — такой же символ слова, как буква: в
    «личное_отпуск.jpg» маркер границы слова без нормализации не находится.

    Точка при этом НЕ трогается. Раньше её тоже заменяли пробелом, и
    маркеры «исх. №» и «вх. №» не срабатывали никогда — а это единственный
    надёжный признак служебного письма.
    """
    return re.sub(r"[_\-/\\]+", " ", value or "")


def classify(haystack: str, allowed: Optional[Sequence[str]] = None) -> RuleMatch:
    """
    Категория по весам маркеров.

    `allowed` ограничивает набор категорий — так путь к файлу не может
    объявить документ перепиской.
    """
    prepared = prepare(haystack)
    if not prepared.strip():
        return RuleMatch()

    permitted: Optional[Set[str]] = set(allowed) if allowed is not None else None

    scores: Dict[str, float] = {}
    hits: Dict[str, int] = {}
    best_marker: Dict[str, Tuple[float, str]] = {}

    counts: Dict[Marker, int] = {}
    for marker in MARKERS:
        if permitted is not None and marker.category not in permitted:
            continue
        count = marker.count(prepared)
        if count:
            counts[marker] = count

    if not _eskd_counts(counts):
        for marker in _ESKD_GROUP:
            counts.pop(marker, None)

    for marker, count in counts.items():
        value = marker.weight * min(_REPEAT_CAP, 1.0 + math.log2(count))
        scores[marker.category] = scores.get(marker.category, 0.0) + value
        hits[marker.title] = count
        top = best_marker.get(marker.category)
        if top is None or value > top[0]:
            best_marker[marker.category] = (value, marker.title)

    if not scores:
        return RuleMatch()

    ranked = sorted(
        scores.items(), key=lambda kv: (-kv[1], CATEGORY_ORDER.index(kv[0]))
    )
    category, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0

    return RuleMatch(
        category=category,
        confidence=_confidence(best, second, len(hits)),
        marker=best_marker[category][1],
        scores=scores,
        hits=hits,
    )


_ESKD_GROUP = (ESKD_DESIGNATION, ESKD_DOCUMENT_CODE, ESKD_REVISION)


def _eskd_counts(counts: Dict[Marker, int]) -> bool:
    """
    Учитывать ли признаки ЕСКД. Код документа и «изм. N» без обозначения не
    значат ничего, а обозначение уступает личному, переписке и служебному
    письму: это ссылка на чертёж, а не сам чертёж.
    """
    if ESKD_DESIGNATION not in counts:
        return False
    for marker in counts:
        if marker.category in (CATEGORY_PERSONAL, CATEGORY_CORRESPONDENCE):
            return False
        if marker.title in _ESKD_YIELDS_TO_TITLES:
            return False
    return True


def _confidence(best: float, second: float, distinct_hits: int) -> float:
    """
    Уверенность — это отрыв лидера, а не его собственный счёт.

    Документ, в котором поровну признаков договора и чертежа, классификатор
    не опознал: какой бы большой ни был счёт у лидера, решение спорное и
    должно уйти на разбор. Поэтому основной вклад даёт отрыв от второго
    места, а число разных сработавших маркеров лишь немного добавляет
    сверху.
    """
    dominance = 0.0 if best <= 0 else max(0.0, (best - second) / best)
    confidence = 0.45 + 0.40 * dominance + 0.05 * min(distinct_hits, 3)
    return round(min(0.95, confidence), 3)
