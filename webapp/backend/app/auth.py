"""Optional API-key protection for network-exposed backend routes."""

import hmac
from collections.abc import Sequence

from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

API_KEY_HEADER = b"x-api-key"
PROTECTED_API_PREFIXES = (
    "/api/detect",
    "/api/history",
)


class ApiKeyAuthMiddleware:
    """Reject protected requests before FastAPI parses an uploaded body."""

    def __init__(
        self,
        app: ASGIApp,
        api_key: str | None,
        protected_prefixes: Sequence[str],
    ) -> None:
        self.app = app
        self.api_key = api_key.encode("ascii") if api_key is not None else None
        self.protected_prefixes = tuple(protected_prefixes)

    def _is_protected(self, path: str) -> bool:
        return any(
            path == prefix or path.startswith(f"{prefix}/")
            for prefix in self.protected_prefixes
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.api_key is None
            or scope.get("type") != "http"
            or scope.get("method") == "OPTIONS"
            or not self._is_protected(scope.get("path", ""))
        ):
            await self.app(scope, receive, send)
            return

        provided = None
        for name, value in scope.get("headers", []):
            if name.lower() == API_KEY_HEADER:
                provided = value
                break

        if provided is None or not hmac.compare_digest(provided, self.api_key):
            response = JSONResponse(
                status_code=401,
                content={"detail": "缺少或无效的 X-API-Key。"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def document_api_key(app: FastAPI, protected_prefixes: Sequence[str]) -> None:
    """Add Swagger authorization only when the deployment enables API-key auth."""
    original_openapi = app.openapi
    prefixes = tuple(protected_prefixes)

    def openapi() -> dict:
        schema = original_openapi()
        security_schemes = schema.setdefault("components", {}).setdefault(
            "securitySchemes", {}
        )
        security_schemes["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
        for path, path_item in schema.get("paths", {}).items():
            if not any(
                path == prefix or path.startswith(f"{prefix}/")
                for prefix in prefixes
            ):
                continue
            for operation in path_item.values():
                if isinstance(operation, dict) and "responses" in operation:
                    operation["security"] = [{"ApiKeyAuth": []}]
        return schema

    app.openapi = openapi
