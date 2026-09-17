"""Интерфейс парсера документов."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.models.parse_result import ParseResult


class ParserUnavailable(RuntimeError):
    """ML-сервис парсинга недоступен — уместен фоллбэк на OCR."""


class ParserFailed(RuntimeError):
    """Сервис ответил, но результат разобрать не удалось — это дефект, не фоллбэк."""


class DocumentParserProvider(ABC):
    @abstractmethod
    def parse(
        self,
        uri: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        """Парсит документ по URI и возвращает структурированный результат."""
