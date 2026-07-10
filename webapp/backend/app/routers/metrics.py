import json

from fastapi import APIRouter

from app.predictor import PROJECT_ROOT
from app.schemas import ModelMetric

router = APIRouter()

METRICS_FILE = PROJECT_ROOT / "checkpoints" / "metrics.json"

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
