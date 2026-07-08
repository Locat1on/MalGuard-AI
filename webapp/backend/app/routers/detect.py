from fastapi import APIRouter, UploadFile

from app.predictor import predictor
from app.schemas import DetectionResult

router = APIRouter()


@router.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile) -> DetectionResult:
    content = await file.read()
    return predictor.predict(file.filename or "unknown", content)
