"""Service-to-service authentication for private inference routes."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status


def require_internal_key(
    request: Request,
    x_internal_key: str | None = Header(default=None),
) -> None:
    expected = request.app.state.settings.internal_api_key
    supplied = x_internal_key or ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "UNAUTHORIZED",
                "message": "A valid internal service key is required.",
            },
        )
