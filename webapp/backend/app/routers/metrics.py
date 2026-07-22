import json

from fastapi import APIRouter, HTTPException

from app.predictor import PROJECT_ROOT
from app.schemas import EvaluationManifest, ModelMetric

router = APIRouter()

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MANIFEST_FILE = CHECKPOINTS_DIR / "evaluation_manifest.json"


def _load_manifest() -> EvaluationManifest:
    if not MANIFEST_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="正式评估结果尚未生成，请先运行 compare_models.py。",
        )
    try:
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        return EvaluationManifest.model_validate(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="正式评估来源清单不可用，请重新运行 compare_models.py。",
        ) from error


@router.get("/metrics", response_model=list[ModelMetric])
async def get_metrics() -> list[ModelMetric]:
    manifest = _load_manifest()
    return [ModelMetric.model_validate(row.model_dump()) for row in manifest.results]


@router.get("/metrics/provenance", response_model=EvaluationManifest)
async def get_metrics_provenance() -> EvaluationManifest:
    return _load_manifest()
