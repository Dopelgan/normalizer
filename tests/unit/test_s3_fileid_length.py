"""Единый предел длины s3_fileid во всех входных контрактах и БД."""

import pytest
from pydantic import ValidationError

from core.db.models import DocumentDB, IntakeDecisionDB
from core.models.api import BackgroundParseRequest, ParseRequest
from core.models.intake import IntakeFile, S3_FILEID_MAX_LENGTH


MAX_FILEID = "я" * S3_FILEID_MAX_LENGTH
TOO_LONG_FILEID = MAX_FILEID + "я"


@pytest.mark.parametrize(
    "build",
    [
        lambda value: IntakeFile(s3_fileid=value),
        lambda value: ParseRequest(
            request_id="request-1", dialog_id="dialog-1", s3_fileid=[value]
        ),
        lambda value: BackgroundParseRequest(
            request_id="request-1", s3_fileid=[value]
        ),
        lambda value: BackgroundParseRequest(
            request_id="request-1", files=[{"s3_fileid": value}]
        ),
    ],
)
def test_all_input_contracts_accept_2048_characters(build):
    build(MAX_FILEID)


@pytest.mark.parametrize(
    "build",
    [
        lambda value: IntakeFile(s3_fileid=value),
        lambda value: ParseRequest(
            request_id="request-1", dialog_id="dialog-1", s3_fileid=[value]
        ),
        lambda value: BackgroundParseRequest(
            request_id="request-1", s3_fileid=[value]
        ),
        lambda value: BackgroundParseRequest(
            request_id="request-1", files=[{"s3_fileid": value}]
        ),
    ],
)
def test_all_input_contracts_reject_more_than_2048_characters(build):
    with pytest.raises(ValidationError):
        build(TOO_LONG_FILEID)


def test_database_columns_fit_full_identifier_and_resolved_path():
    assert DocumentDB.__table__.c.s3_fileid.type.length == 2048
    assert IntakeDecisionDB.__table__.c.s3_fileid.type.length == 2048
    assert DocumentDB.__table__.c.source_path.type.length == 4096
    assert IntakeDecisionDB.__table__.c.source_path.type.length == 4096
