from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import history
from app.predictor import predictor
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


@app.get("/api/health")
async def health() -> dict[str, bool]:
    # Surfaces real model-load state rather than a constant True — a stale/incompatible
    # checkpoint (e.g. an architecture mismatch after a model refactor) would otherwise be
    # invisible here even though predict() is silently falling back to the hash-based stub.
    return {
        "ok": True,
        "modelsLoaded": predictor.models_loaded,
        "familyModelLoaded": predictor.family_model_loaded,
    }
