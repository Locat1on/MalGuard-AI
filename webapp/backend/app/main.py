import logging
import os
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app import history
from app.auth import (
    ApiKeyAuthMiddleware,
    PROTECTED_API_PREFIXES,
    document_api_key,
)
from app.predictor import predictor
from app.schemas import HealthStatus
from app.settings import settings
from app.upload_limits import (
    ContentLengthLimitMiddleware,
    DETECT_REQUEST_LIMITS,
    DetectionConcurrencyLimitMiddleware,
)
from app.routers import detect, metrics
from app.routers import history as history_router

history.init_db()

REQUEST_LOGGER = logging.getLogger("malguard.requests")

app = FastAPI(title="MalGuard AI backend")

app.add_middleware(
    ContentLengthLimitMiddleware,
    path_limits=DETECT_REQUEST_LIMITS,
)
app.add_middleware(
    DetectionConcurrencyLimitMiddleware,
    max_active=settings.detection_concurrency,
)
app.add_middleware(
    ApiKeyAuthMiddleware,
    api_key=settings.api_key,
    protected_prefixes=PROTECTED_API_PREFIXES,
)

app.include_router(detect.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(history_router.router, prefix="/api")
if settings.api_key is not None:
    document_api_key(app, PROTECTED_API_PREFIXES)


@app.middleware("http")
async def observe_request(request: Request, call_next):
    """Attach a request id and server duration without logging uploaded filenames/content."""
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        REQUEST_LOGGER.exception(
            "request_id=%s method=%s path=%s status=500 duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误。"},
        )
    else:
        duration_ms = (time.perf_counter() - started) * 1000
        REQUEST_LOGGER.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "Retry-After",
        "WWW-Authenticate",
        "X-Process-Time-Ms",
        "X-Request-ID",
        "X-Total-Count",
    ],
)


def _health_status() -> HealthStatus:
    mode = "real" if predictor.models_loaded else ("stub" if predictor.stub_enabled else "unavailable")
    return HealthStatus(
        ok=True,
        ready=predictor.models_loaded,
        mode=mode,
        modelsLoaded=predictor.models_loaded,
        familyModelLoaded=predictor.family_model_loaded,
        llmConfigured=bool(os.environ.get("OPENROUTER_API_KEY")),
        modelLoadError=predictor.model_load_error,
        familyModelLoadError=predictor.family_model_load_error,
        modelProvenanceVerified=predictor.model_provenance_verified,
        modelProvenanceWarning=predictor.model_provenance_warning,
        inferenceConcurrency=predictor.inference_concurrency,
        detectionConcurrency=settings.detection_concurrency,
        apiKeyRequired=settings.api_key is not None,
    )


@app.get("/api/health", response_model=HealthStatus)
async def health() -> HealthStatus:
    """Liveness plus optional-component state; always responds while the API process is alive."""
    return _health_status()


@app.get("/api/ready", response_model=HealthStatus)
async def ready() -> HealthStatus | JSONResponse:
    """Readiness for real detection. Optional family/LLM components do not gate readiness."""
    status = _health_status()
    if status.ready:
        return status
    return JSONResponse(status_code=503, content=status.model_dump())
