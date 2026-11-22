"""Bound encoded result envelopes without retaining an oversized JSON body."""

import json

MAX_RESPONSE_BYTES = 16_000_000


def encode_payload(payload, limit, *, ensure_ascii=False, compact=True):
    chunks, size = [], 0
    encoder = json.JSONEncoder(ensure_ascii=ensure_ascii, allow_nan=False,
                               separators=(",", ":") if compact else None)
    for chunk in encoder.iterencode(payload):
        encoded = chunk.encode("utf-8")
        size += len(encoded)
        if size > limit:
            return None
        chunks.append(encoded)
    return b"".join(chunks)


def resource_error():
    return {"reason_code": "TABLE_EXTRACTION_FAILED",
            "message": "PARSER_RESPONSE_RESOURCE_LIMIT: full response exceeds byte budget; no prefix returned",
            "retryable": False}
