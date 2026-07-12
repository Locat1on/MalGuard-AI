import hashlib

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app import history
from app.predictor import FeatureExtractionError, predictor
from app.schemas import BatchDetectionResult, BatchItem, DetectionResult

router = APIRouter()

# predictor.predict() runs GPU inference plus a synchronous OpenRouter LLM call on every
# request (see src/llm/report.py) — well over what's reasonable to block the single-threaded
# asyncio event loop on, which would otherwise stall every other in-flight request (including
# unrelated /api/health or /api/metrics calls) for the duration of one detection.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# Cap on how many files a single batch request may carry, so one request can't tie up the
# worker threadpool indefinitely.
MAX_BATCH_FILES = 100


class FileTooLargeError(Exception):
    """An upload exceeded MAX_UPLOAD_BYTES. Single-file detect turns this into a 413; batch
    turns it into a per-file failure item so one oversized file doesn't sink the whole batch."""


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload, rejecting anything over MAX_UPLOAD_BYTES before it is fully buffered."""
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(f"文件过大（{file.size} 字节），超过 {MAX_UPLOAD_BYTES} 字节上限。")
    # file.size isn't always populated ahead of time, so reading one byte past the cap is the
    # actual backstop against buffering an unbounded upload into memory.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(f"文件过大，超过 {MAX_UPLOAD_BYTES} 字节上限。")
    return content


@router.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile) -> DetectionResult:
    try:
        content = await _read_capped(file)
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    filename = file.filename or "unknown"
    try:
        result = await run_in_threadpool(predictor.predict, filename, content)
    except FeatureExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    # Persist only genuine detections — the hash-based stub (models not loaded) is not recorded,
    # so history reflects real model output only.
    if predictor.models_loaded:
        sha256 = hashlib.sha256(content).hexdigest()
        result.historyId = await run_in_threadpool(history.record, result, "single", sha256)
    return result


def _error_item(filename: str, error: str) -> BatchItem:
    return BatchItem(
        filename=filename, ok=False, verdict=None, confidence=None, family=None,
        familyConfidence=None, lgbmScore=None, mlpScore=None, modelAgreement=None,
        historyId=None, error=error,
    )


@router.post("/detect/batch", response_model=BatchDetectionResult)
async def detect_batch(files: list[UploadFile]) -> BatchDetectionResult:
    """Scan multiple files with the two ML models only (no LLM/ATT&CK — see predict_ml_only).

    Per-file failures (a non-PE file, or one that's over the size cap) are reported as `ok=False`
    items rather than failing the whole request, so one bad file in a folder doesn't sink the
    rest of the batch.
    """
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件。")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(status_code=413, detail=f"单次批量最多 {MAX_BATCH_FILES} 个文件，本次为 {len(files)} 个。")

    items: list[BatchItem] = []
    # (index into `items`, DetectionResult, sha256) for successful rows, so they can all be
    # persisted in one transaction after the loop and have their history ids written back.
    to_record: list[tuple[int, DetectionResult, str]] = []
    for file in files:
        filename = file.filename or "unknown"
        try:
            content = await _read_capped(file)
        except FileTooLargeError as e:
            items.append(_error_item(filename, str(e)))
            continue
        try:
            result = await run_in_threadpool(predictor.predict_ml_only, filename, content)
        except FeatureExtractionError as e:
            items.append(_error_item(filename, str(e)))
            continue

        item = BatchItem(
            filename=filename, ok=True, verdict=result.verdict, confidence=result.confidence,
            family=result.family, familyConfidence=result.familyConfidence,
            lgbmScore=result.lgbmScore, mlpScore=result.mlpScore,
            modelAgreement=result.modelAgreement, historyId=None, error=None,
        )
        if predictor.models_loaded:
            to_record.append((len(items), result, hashlib.sha256(content).hexdigest()))
        items.append(item)

    if to_record:
        ids = await run_in_threadpool(
            history.record_many, [(r, sha) for _, r, sha in to_record], "batch"
        )
        for (idx, _, _), history_id in zip(to_record, ids):
            items[idx].historyId = history_id

    malicious = sum(1 for i in items if i.verdict == "malicious")
    benign = sum(1 for i in items if i.verdict == "benign")
    failed = sum(1 for i in items if not i.ok)
    return BatchDetectionResult(
        items=items, total=len(items), malicious=malicious, benign=benign, failed=failed
    )
