"""Measure the exact UTF-8 HTTP envelope; never return a successful prefix."""

from fastapi.responses import JSONResponse, Response

from core.models.contract import ParseResponse, ResultsResponse, SyncPendingResponse
from core.transport_budget import MAX_RESPONSE_BYTES, encode_payload, resource_error


def _error_envelope(model):
    """Ответ 413 в форме модели ручки: тот же request_id, ошибка, без документов.

    Ответ /parse/results — ResultsResponse. Ответ синхронной ручки —
    SyncPendingResponse: это ParseResponse (request_id, dialog_id, documents)
    со статусом и ошибкой, тот же вид, что у её ответа 504.
    """
    keep = {name: getattr(model, name) for name in ("request_id", "dialog_id", "event_id", "operation")
            if name in type(model).model_fields}
    body = {**keep, "status": "error", "documents": [], "error": resource_error()}
    if isinstance(model, ResultsResponse):
        return ResultsResponse(**body).model_dump(mode="json")
    if isinstance(model, (ParseResponse, SyncPendingResponse)):
        return SyncPendingResponse(**body).model_dump(mode="json")
    return body


def bounded_response(model, *, status_code=200):
    encoded = encode_payload(model.model_dump(mode="json"), MAX_RESPONSE_BYTES)
    if encoded is None:
        return JSONResponse(status_code=413, content=_error_envelope(model))
    return Response(content=encoded, status_code=status_code, media_type="application/json")
