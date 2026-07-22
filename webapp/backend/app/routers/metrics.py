import json

from fastapi import APIRouter, HTTPException

from app.predictor import PROJECT_ROOT
from app.schemas import ModelMetric

router = APIRouter()

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MANIFEST_FILE = CHECKPOINTS_DIR / "evaluation_manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="正式评估结果尚未生成，请先运行 compare_models.py。",
        )
    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("evaluation_manifest.json must contain an object")
        return manifest
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="正式评估来源清单不可用，请重新运行 compare_models.py。",
        ) from error


@router.get("/metrics", response_model=list[ModelMetric])
async def get_metrics() -> list[ModelMetric]:
    manifest = _load_manifest()
    try:
        results = manifest["results"]
        if not isinstance(results, list) or not results:
            raise ValueError("evaluation results must contain a non-empty list")
        return [ModelMetric(**row) for row in results]
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="正式评估指标不可用，请重新运行 compare_models.py。",
        ) from error


@router.get("/metrics/provenance")
async def get_metrics_provenance() -> dict:
    return _load_manifest()
