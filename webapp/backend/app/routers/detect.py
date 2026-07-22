import hashlib
import logging
import time

from fastapi import APIRouter, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app import history
from app.predictor import FeatureExtractionError, ModelUnavailableError, predictor
from app.schemas import BatchDetectionResult, BatchItem, DetectionResult
from app.upload_limits import MAX_BATCH_FILES, MAX_BATCH_PAYLOAD_BYTES, MAX_UPLOAD_BYTES

router = APIRouter()
BATCH_LOGGER = logging.getLogger("malguard.batch")

# predictor.predict() runs GPU inference plus a synchronous OpenRouter LLM call on every
# request (see src/llm/report.py) — well over what's reasonable to block the single-threaded
# asyncio event loop on, which would otherwise stall every other in-flight request (including
# unrelated /api/health or /api/metrics calls) for the duration of one detection.

class FileTooLargeError(Exception):
    """An upload exceeded MAX_UPLOAD_BYTES. Single-file detect turns this into a 413; batch
    turns it into a per-file failure item so one oversized file doesn't sink the whole batch."""


def _require_detector() -> None:
    if not predictor.models_loaded and not predictor.stub_enabled:
        raise HTTPException(
            status_code=503,
            detail=predictor.model_load_error or "检测模型当前不可用。",
        )


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
    _require_detector()
    try:
        content = await _read_capped(file)
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    filename = file.filename or "unknown"
    try:
        result = await run_in_threadpool(predictor.predict, filename, content)
    except FeatureExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    # Persist only genuine detections — the hash-based stub (models not loaded) is not recorded,
    # so history reflects real model output only.
    if predictor.models_loaded:
        sha256 = hashlib.sha256(content).hexdigest()
        result.historyId = await run_in_threadpool(history.record, result, "single", sha256)
    return result


def _error_item(filename: str, error: str) -> BatchItem:
    return BatchItem(
        filename=filename,
        ok=False,
        verdict=None,
        confidence=None,
        family=None,
        familyConfidence=None,
        lgbmScore=None,
        mlpScore=None,
        modelAgreement=None,
        historyId=None,
        error=error,
    )


def _success_item(result: DetectionResult) -> BatchItem:
    return BatchItem(
        filename=result.filename,
        ok=True,
        verdict=result.verdict,
        confidence=result.confidence,
        family=result.family,
        familyConfidence=result.familyConfidence,
        lgbmScore=result.lgbmScore,
        mlpScore=result.mlpScore,
        modelAgreement=result.modelAgreement,
        historyId=None,
        error=None,
    )


@router.post("/detect/batch", response_model=BatchDetectionResult)
async def detect_batch(files: list[UploadFile]) -> BatchDetectionResult:
    """Extract files independently, then run one vectorized ML inference batch."""
    _require_detector()
    request_started = time.perf_counter()
    extraction_seconds = 0.0
    inference_seconds = 0.0
    if not files:
        raise HTTPException(status_code=400, detail="未收到任何文件。")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"单次批量最多 {MAX_BATCH_FILES} 个文件，本次为 {len(files)} 个。",
        )
    known_payload_bytes = sum(file.size or 0 for file in files)
    if known_payload_bytes > MAX_BATCH_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"批量文件总计 {known_payload_bytes} 字节，超过 "
                f"{MAX_BATCH_PAYLOAD_BYTES} 字节上限。"
            ),
        )

    items: list[BatchItem | None] = [None] * len(files)
    valid_positions: list[int] = []
    valid_filenames: list[str] = []
    feature_rows: list = []
    file_hashes: list[str] = []
    total_upload_bytes = 0

    for position, file in enumerate(files):
        filename = file.filename or "unknown"
        try:
            content = await _read_capped(file)
        except FileTooLargeError as error:
            items[position] = _error_item(filename, str(error))
            continue

        total_upload_bytes += len(content)
        if total_upload_bytes > MAX_BATCH_PAYLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"批量文件总计超过 {MAX_BATCH_PAYLOAD_BYTES} 字节上限。"
                ),
            )

        if not predictor.models_loaded:
            result = await run_in_threadpool(
                predictor.predict_ml_only, filename, content
            )
            items[position] = _success_item(result)
            del content
            continue

        file_hash = hashlib.sha256(content).hexdigest()
        extraction_started = time.perf_counter()
        try:
            features = await run_in_threadpool(
                predictor.extract_feature_vector, content
            )
        except FeatureExtractionError as error:
            items[position] = _error_item(filename, str(error))
            extraction_seconds += time.perf_counter() - extraction_started
            del content
            continue
        extraction_seconds += time.perf_counter() - extraction_started
        del content

        valid_positions.append(position)
        valid_filenames.append(filename)
        feature_rows.append(features)
        file_hashes.append(file_hash)

    to_record: list[tuple[int, DetectionResult, str]] = []
    if feature_rows:
        inference_started = time.perf_counter()
        try:
            results = await run_in_threadpool(
                predictor.predict_features_ml_only,
                valid_filenames,
                feature_rows,
            )
            inference_seconds = time.perf_counter() - inference_started
        except ModelUnavailableError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        for position, result, file_hash in zip(
            valid_positions, results, file_hashes
        ):
            items[position] = _success_item(result)
            to_record.append((position, result, file_hash))

    if to_record:
        history_ids = await run_in_threadpool(
            history.record_many,
            [(result, file_hash) for _, result, file_hash in to_record],
            "batch",
        )
        for (position, _, _), history_id in zip(to_record, history_ids):
            item = items[position]
            if item is not None:
                item.historyId = history_id

    if any(item is None for item in items):
        raise RuntimeError("batch result alignment invariant failed")
    final_items = [item for item in items if item is not None]
    malicious = sum(1 for item in final_items if item.verdict == "malicious")
    benign = sum(1 for item in final_items if item.verdict == "benign")
    failed = sum(1 for item in final_items if not item.ok)
    BATCH_LOGGER.info(
        "files=%d valid=%d failed=%d extraction_ms=%.1f inference_ms=%.1f total_ms=%.1f",
        len(final_items),
        len(final_items) - failed,
        failed,
        extraction_seconds * 1000,
        inference_seconds * 1000,
        (time.perf_counter() - request_started) * 1000,
    )
    return BatchDetectionResult(
        items=final_items,
        total=len(final_items),
        malicious=malicious,
        benign=benign,
        failed=failed,
    )
