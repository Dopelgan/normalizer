"""Модели запросов ручек Parser (контракт RAG <-> Parser)."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models.contract import Operation
from core.models.intake import IntakeFile


class ParseRequest(BaseModel):
    """POST /internal/v1/parse — синхронная обработка."""

    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(..., min_length=1)
    dialog_id: str = Field(..., min_length=1)
    s3_fileid: List[str] = Field(..., min_length=1)


class BackgroundParseRequest(BaseModel):
    """
    POST /internal/v1/parse/background — фоновая обработка.

    Принимает два вида тела:

    * `files: [{s3_fileid, operation}]` — операция на каждый файл. Так
      присылает Quality Gate, и так же можно прислать напрямую;
    * `s3_fileid: [...]` плюс одна `operation` на весь пакет — прежний вид,
      оставлен ради совместимости.

    `dialog_id` в фоновом контракте больше нет: связь lifecycle-события,
    запроса Parser и результата держит `event_id`. Пришедший `dialog_id`
    молча игнорируется (`extra="ignore"`) — старый вызов не ломается, но и
    полем контракта он больше не является.

    `event_id` приходит от Backend через RAG и не заменяется новым UUID.
    Он необязателен: в цепочке приёма (Data Gateway -> Quality Gate ->
    Parser) события жизненного цикла нет, и выдумывать его нечем.
    """

    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(..., min_length=1)
    event_id: str = ""
    s3_fileid: List[str] = Field(default_factory=list)
    operation: Operation = "create"
    files: List[IntakeFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_not_empty(self) -> "BackgroundParseRequest":
        if not self.files and not self.s3_fileid:
            raise ValueError("Пустой пакет: нужен либо files, либо s3_fileid.")
        return self

    @property
    def items(self) -> List[IntakeFile]:
        """Пакет в едином виде — файл плюс его операция."""
        if self.files:
            return list(self.files)
        return [
            IntakeFile(s3_fileid=fileid, operation=self.operation)
            for fileid in self.s3_fileid
        ]


class HealthResponse(BaseModel):
    status: str
    service: str = "parser"
    dependencies: dict = Field(default_factory=dict)
