"""
Проверка общего bearer-токена на HTTP-ручках сервисов (ingest, gateway, quality).

Флаг `NORMALIZER_REQUIRE_TOKEN` выключен — ручки открыты, как раньше. Включён —
каждая ручка с зависимостью `require_api_token` ждёт заголовок
`Authorization: Bearer <NORMALIZER_API_TOKEN>`, иначе 401. Сравнение — за
постоянное время. Пробы `/health` и `/metrics` зависимость не получают.

Флаг включён, а токен пуст — сервис не стартует (`ensure_configured` на
старте приложения): открытая ручка при включённой защите хуже явного отказа.
"""

import hmac
from typing import Optional

from fastapi import Header, HTTPException

from core.config import settings


class ApiTokenMisconfigured(RuntimeError):
    """Защита включена, а токен не задан."""


def ensure_configured() -> None:
    """Отказ запуска, если защита включена без токена."""
    if settings.NORMALIZER_REQUIRE_TOKEN and not settings.NORMALIZER_API_TOKEN.strip():
        raise ApiTokenMisconfigured(
            "NORMALIZER_REQUIRE_TOKEN=true, но NORMALIZER_API_TOKEN пуст: "
            "сервис не запускается без токена."
        )


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="Нужен заголовок Authorization: Bearer <токен>.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_api_token(authorization: Optional[str] = Header(default=None)) -> None:
    """Зависимость FastAPI: пропускает запрос с верным токеном или при выключенном флаге."""
    if not settings.NORMALIZER_REQUIRE_TOKEN:
        return
    expected = settings.NORMALIZER_API_TOKEN.strip()
    if not expected:
        # Старт с такой настройкой уже запрещён; здесь — на случай смены на лету.
        raise _unauthorized()
    scheme, _, supplied = (authorization or "").strip().partition(" ")
    if scheme.lower() != "bearer" or not supplied.strip():
        raise _unauthorized()
    if not hmac.compare_digest(supplied.strip().encode("utf-8"), expected.encode("utf-8")):
        raise _unauthorized()


def outbound_headers() -> dict:
    """Заголовок для своих же вызовов между сервисами (gateway → ingest)."""
    token = settings.NORMALIZER_API_TOKEN.strip()
    return {"Authorization": f"Bearer {token}"} if token else {}
