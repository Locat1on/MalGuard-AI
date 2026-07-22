import json

from fastapi import APIRouter, HTTPException

from app.predictor import PROJECT_ROOT
from app.schemas import ModelMetric

router = APIRouter()

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
METRICS_FILE = CHECKPOINTS_DIR / "metrics.json"
MANIFEST_FILE = CHECKPOINTS_DIR / "evaluation_manifest.json"

STUB_METRICS = [
    ModelMetric(model="LightGBM (EMBER 静态特征基线) [占位数据]", accuracy=0, precision=0, recall=0, f1=0),
    ModelMetric(model="MLP 深度模型 (本系统) [占位数据]", accuracy=0, precision=0, recall=0, f1=0),
]


@router.get("/metrics", response_model=list[ModelMetric])
async def get_metrics() -> list[ModelMetric]:
    if METRICS_FILE.exists():
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        return [ModelMetric(**row) for row in data]
    return STUB_METRICS


@router.get("/metrics/provenance")
async def get_metrics_provenance() -> dict:
    if not MANIFEST_FILE.exists():
        raise HTTPException(status_code=404, detail="evaluation manifest is not available")
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
