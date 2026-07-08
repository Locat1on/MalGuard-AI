from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import detect, metrics

app = FastAPI(title="MalGuard AI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")


@app.get("/api/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
