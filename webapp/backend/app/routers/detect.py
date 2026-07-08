from fastapi import APIRouter, HTTPException, UploadFile

from app.predictor import FeatureExtractionError, predictor
from app.schemas import DetectionResult

router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile) -> DetectionResult:
    content = await file.read()
    try:
        return predictor.predict(file.filename or "unknown", content)
    except FeatureExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
