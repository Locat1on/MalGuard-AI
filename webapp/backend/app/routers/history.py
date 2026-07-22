import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from app import history
from app.schemas import HistoryRecord, HistoryStats

router = APIRouter()
HISTORY_LOGGER = logging.getLogger("malguard.history")


@router.get("/history", response_model=list[HistoryRecord])
async def list_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    return await run_in_threadpool(history.list_recent, limit, offset)


@router.get("/history/stats", response_model=HistoryStats)
async def get_history_stats() -> dict:
    return await run_in_threadpool(history.stats)


@router.get("/history/backup", response_class=FileResponse)
async def backup_history() -> FileResponse:
    """Download a consistent SQLite snapshot without stopping detection writes."""
    backup_path = history.DB_PATH.parent / f".history-backup-{uuid.uuid4().hex}.db"
    try:
        await run_in_threadpool(history.backup_to, backup_path)
    except Exception as error:
        backup_path.unlink(missing_ok=True)
        HISTORY_LOGGER.exception("history backup failed: %s", type(error).__name__)
        raise HTTPException(status_code=500, detail="历史数据库备份失败。") from error

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return FileResponse(
        path=backup_path,
        media_type="application/vnd.sqlite3",
        filename=f"malguard-history-{timestamp}.db",
        background=BackgroundTask(backup_path.unlink, missing_ok=True),
    )


@router.get("/history/{detection_id}", response_model=HistoryRecord)
async def get_history(detection_id: int) -> dict:
    rec = await run_in_threadpool(history.get, detection_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"检测记录 #{detection_id} 不存在。")
    return rec


@router.get("/history/{detection_id}/report", response_class=HTMLResponse)
async def get_report(detection_id: int) -> HTMLResponse:
    rec = await run_in_threadpool(history.get, detection_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"检测记录 #{detection_id} 不存在。")
    html = history.render_report_html(rec)
    # inline (not attachment) so the browser renders it and the user can print-to-PDF; the
    # filename hint still applies if they choose "save as".
    filename = f"report-{detection_id}.html"
    return HTMLResponse(content=html, headers={"Content-Disposition": f'inline; filename="{filename}"'})


@router.delete("/history/{detection_id}")
async def delete_history(detection_id: int) -> dict:
    deleted = await run_in_threadpool(history.delete, detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"检测记录 #{detection_id} 不存在。")
    return {"deleted": True}


@router.delete("/history")
async def clear_history() -> dict:
    count = await run_in_threadpool(history.clear)
    return {"deleted": count}
