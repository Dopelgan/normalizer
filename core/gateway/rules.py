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

_TECHNICAL: Tuple[Marker, ...] = (
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
    _marker(CATEGORY_BUSINESS, 0.9, r"\bсвидетельств\w+", "свидетельство"),
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

    for marker in MARKERS:
        if permitted is not None and marker.category not in permitted:
            continue
        count = marker.count(prepared)
        if not count:
            continue
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
