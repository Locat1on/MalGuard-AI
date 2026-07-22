import json

from fastapi import APIRouter, HTTPException

from app.predictor import PROJECT_ROOT
from app.schemas import ModelMetric

router = APIRouter()

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
METRICS_FILE = CHECKPOINTS_DIR / "metrics.json"
MANIFEST_FILE = CHECKPOINTS_DIR / "evaluation_manifest.json"

@router.get("/metrics", response_model=list[ModelMetric])
async def get_metrics() -> list[ModelMetric]:
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="正式评估指标尚未生成，请先运行 compare_models.py。",
        )
    try:
        data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data:
            raise ValueError("metrics.json must contain a non-empty list")
        return [ModelMetric(**row) for row in data]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="正式评估指标文件不可用，请重新运行 compare_models.py。",
        ) from error


@router.get("/metrics/provenance")
async def get_metrics_provenance() -> dict:
    if not MANIFEST_FILE.exists():
        raise HTTPException(status_code=404, detail="evaluation manifest is not available")
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
