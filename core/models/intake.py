"""Модели ручек приёма: Data Gateway и Quality Gate."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from core.config import settings
from core.models.contract import Operation


class IntakeFile(BaseModel):
    """
    Файл пакета вместе с операцией над ним.

    Операция принадлежит файлу, а не пакету: в одном запросе приходят и
    новые документы, и обновления уже принятых, и решать по всему пакету
    сразу значит либо переразбирать лишнее, либо не переразобрать нужное.
    """

    s3_fileid: str = Field(..., min_length=1, max_length=256)
    operation: Operation = "create"


class IntakeRequest(BaseModel):
    """Запрос на приём пакета файлов — вход всей цепочки."""

    request_id: str = Field(..., min_length=1, max_length=128)
    files: List[IntakeFile] = Field(..., min_length=1)

    @field_validator("files")
    @classmethod
    def _check_files(cls, value: List[IntakeFile]) -> List[IntakeFile]:
        if len(value) > settings.INTAKE_MAX_FILES:
            raise ValueError(
                f"В пакете {len(value)} файлов при потолке "
                f"{settings.INTAKE_MAX_FILES}. Разбейте на несколько запросов."
            )
        seen = {item.s3_fileid for item in value}
        if len(seen) != len(value):
            raise ValueError(
                "Один и тот же s3_fileid встречается в пакете дважды — "
                "неясно, какую операцию по нему выполнять."
            )
        return value

    @property
    def fileids(self) -> List[str]:
        return [item.s3_fileid for item in self.files]

    @property
    def operations(self) -> Dict[str, str]:
        return {item.s3_fileid: item.operation for item in self.files}

    def subset(self, fileids: List[str]) -> List[IntakeFile]:
        """Те же файлы со своими операциями, но только перечисленные."""
        wanted = set(fileids)
        return [item for item in self.files if item.s3_fileid in wanted]


class FileVerdict(BaseModel):
    """Решение по одному файлу на одном этапе."""

    s3_fileid: str
    outcome: str                      # accept | quarantine | reject
    stage: str                        # data_gateway | quality_gate
    layer: Optional[str] = None
    reason: str = ""
    category: Optional[str] = None
    confidence: float = 1.0
    warnings: List[str] = Field(default_factory=list)
    routing: Dict[str, Any] = Field(default_factory=dict)
    decision_id: Optional[str] = None


class IntakeResponse(BaseModel):
    """
    Итог приёма пакета. `forwarded` — файлы, дошедшие до нормализации;
    остальные остались в карантине или отклонены, и по каждому есть причина.
    """

    request_id: str
    accepted: List[str] = Field(default_factory=list)
    quarantined: List[str] = Field(default_factory=list)
    rejected: List[str] = Field(default_factory=list)
    verdicts: List[FileVerdict] = Field(default_factory=list)
    forwarded: bool = False
    forward_error: Optional[str] = None


class ResolveRequest(BaseModel):
    """Решение администратора по карантинной записи."""

    outcome: str = Field(..., pattern="^(accept|reject)$")
    resolved_by: str = "admin"


class QuarantineItem(BaseModel):
    decision_id: str
    s3_fileid: str
    stage: str
    layer: Optional[str] = None
    category: Optional[str] = None
    reason: str
    confidence: float
    created_at: Optional[str] = None
