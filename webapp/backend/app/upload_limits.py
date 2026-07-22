"""Upload limits shared by the ASGI boundary and detection routes."""

import threading
from collections.abc import Mapping

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

MIB = 1024 * 1024
MAX_UPLOAD_BYTES = 100 * MIB
MAX_BATCH_FILES = 100

# Multipart headers and boundaries add a small amount beyond the raw file bytes.
MAX_SINGLE_REQUEST_BYTES = MAX_UPLOAD_BYTES + MIB
MAX_BATCH_PAYLOAD_BYTES = 500 * MIB
MAX_BATCH_REQUEST_BYTES = 512 * MIB
DETECT_REQUEST_LIMITS = {
    "/api/detect": MAX_SINGLE_REQUEST_BYTES,
    "/api/detect/batch": MAX_BATCH_REQUEST_BYTES,
}


class ContentLengthLimitMiddleware:
    """Reject declared oversized detection bodies before multipart parsing starts.

    Chunked requests have no Content-Length and continue to the route-level capped reader.
    A production reverse proxy should also enforce a body limit for that transport mode.
    """

    def __init__(self, app: ASGIApp, path_limits: Mapping[str, int]) -> None:
        self.app = app
        self.path_limits = dict(path_limits)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self.path_limits.get(scope.get("path", ""))
        if scope.get("type") != "http" or scope.get("method") != "POST" or limit is None:
            await self.app(scope, receive, send)
            return

        content_length = None
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    content_length = int(value)
                except ValueError:
                    content_length = None
                break

        if content_length is not None and content_length > limit:
            response = JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"请求体过大（{content_length} 字节），超过该接口 "
                        f"{limit} 字节上限。"
                    )
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class DetectionConcurrencyLimitMiddleware:
    """Reject excess detection requests before multipart parsing and extraction."""

    def __init__(self, app: ASGIApp, max_active: int) -> None:
        if max_active < 1:
            raise ValueError("max_active must be positive")
        self.app = app
        self.max_active = max_active
        self._active = 0
        self._lock = threading.Lock()

    @staticmethod
    def _applies(scope: Scope) -> bool:
        path = scope.get("path", "")
        return (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and (path == "/api/detect" or path.startswith("/api/detect/"))
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return

        with self._lock:
            if self._active >= self.max_active:
                accepted = False
            else:
                self._active += 1
                accepted = True

        if not accepted:
            response = JSONResponse(
                status_code=429,
                content={"detail": "检测服务繁忙，请稍后重试。"},
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)
        finally:
            with self._lock:
                self._active -= 1
