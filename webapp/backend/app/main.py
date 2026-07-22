import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app import history
from app.predictor import predictor
from app.schemas import HealthStatus
from app.routers import detect, metrics
from app.routers import history as history_router

history.init_db()

app = FastAPI(title="MalGuard AI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(history_router.router, prefix="/api")


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
