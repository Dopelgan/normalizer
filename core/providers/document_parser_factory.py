"""Фабрика провайдера разбора документов."""

from typing import Optional

from core.providers.document_parser import DocumentParserProvider
from core.providers.storage import StorageProvider


class DocumentParserFactory:
    """
    Отдаёт разбор через лестницу стратегий: маршрутизатор сам выбирает
    самый дешёвый достаточный уровень, а задача обработки документа про
    уровни ничего не знает.
    """

    @staticmethod
    def get_parser(storage: Optional[StorageProvider] = None) -> DocumentParserProvider:
        # Импорт внутри функции: лестница тянет за собой парсеры всех
        # уровней, и на импорте модуля это лишняя задержка старта.
        from core.ladder.router import LadderParserProvider

        return LadderParserProvider(storage=storage)
