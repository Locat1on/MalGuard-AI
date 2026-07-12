"""SQLite persistence for detection history + self-contained HTML report rendering.

This is runtime state, not a model checkpoint, so it lives under data/ (already gitignored)
rather than checkpoints/. Only real detections are recorded — the router guards recording on
`predictor.models_loaded`, so the hash-based stub never pollutes the history.

Uses stdlib sqlite3 (no new dependency) with a fresh connection per call: detection runs off
the event loop via run_in_threadpool, so a single shared connection would be touched from
multiple worker threads — a connection per call sidesteps that without a connection pool.
"""

import json
import sqlite3
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from app.predictor import PROJECT_ROOT
from app.schemas import DetectionResult

DB_PATH = PROJECT_ROOT / "data" / "history.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     TEXT    NOT NULL,
                filename       TEXT    NOT NULL,
                sha256         TEXT    NOT NULL,
                source         TEXT    NOT NULL,
                verdict        TEXT    NOT NULL,
                confidence     REAL    NOT NULL,
                family         TEXT,
                family_confidence REAL,
                lgbm_score     REAL    NOT NULL,
                mlp_score      REAL    NOT NULL,
                model_agreement TEXT   NOT NULL,
                llm_verdict    TEXT,
                llm_confidence REAL,
                llm_report     TEXT    NOT NULL,
                attck          TEXT    NOT NULL
            )
            """
        )
        # Migrate DBs created before family_confidence existed (the file is regenerable, but
        # an ALTER keeps any accumulated demo history intact rather than requiring a wipe).
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(detections)")}
        if "family_confidence" not in columns:
            conn.execute("ALTER TABLE detections ADD COLUMN family_confidence REAL")


_INSERT_SQL = """
    INSERT INTO detections (
        created_at, filename, sha256, source, verdict, confidence, family, family_confidence,
        lgbm_score, mlp_score, model_agreement, llm_verdict, llm_confidence,
        llm_report, attck
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _row_values(result: DetectionResult, source: str, sha256: str) -> tuple:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        result.filename,
        sha256,
        source,
        result.verdict,
        result.confidence,
        result.family,
        result.familyConfidence,
        result.lgbmScore,
        result.mlpScore,
        result.modelAgreement,
        result.llmVerdict,
        result.llmConfidence,
        result.llmReport,
        json.dumps([t.model_dump() for t in result.attck], ensure_ascii=False),
    )


def record(result: DetectionResult, source: str, sha256: str) -> int:
    """Persist one detection, returning its new row id."""
    with _connect() as conn:
        cur = conn.execute(_INSERT_SQL, _row_values(result, source, sha256))
        return int(cur.lastrowid)


def record_many(entries: list[tuple[DetectionResult, str]], source: str) -> list[int]:
    """Persist a batch of detections in one connection/transaction, returning row ids in order.

    Uses per-row execute (not executemany) inside a single transaction so each row's lastrowid
    can be captured — a batch scan needs every item's history id back to build its response.
    """
    ids: list[int] = []
    with _connect() as conn:
        for result, sha256 in entries:
            cur = conn.execute(_INSERT_SQL, _row_values(result, source, sha256))
            ids.append(int(cur.lastrowid))
    return ids


def _row_to_dict(row: sqlite3.Row) -> dict:
    """SQLite row -> the camelCase shape HistoryRecord expects."""
    return {
        "id": row["id"],
        "createdAt": row["created_at"],
        "filename": row["filename"],
        "sha256": row["sha256"],
        "source": row["source"],
        "verdict": row["verdict"],
        "confidence": row["confidence"],
        "family": row["family"],
        "familyConfidence": row["family_confidence"],
        "lgbmScore": row["lgbm_score"],
        "mlpScore": row["mlp_score"],
        "modelAgreement": row["model_agreement"],
        "llmVerdict": row["llm_verdict"],
        "llmConfidence": row["llm_confidence"],
        "llmReport": row["llm_report"],
        "attck": json.loads(row["attck"]),
    }


def list_recent(limit: int, offset: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get(detection_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    return _row_to_dict(row) if row else None


def delete(detection_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
        return cur.rowcount > 0


def clear() -> int:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM detections")
        return cur.rowcount


def render_report_html(rec: dict) -> str:
    """A self-contained HTML report for one detection — no external assets, so the browser can
    open it directly and print to PDF. Kept intentionally plain (inline CSS, no JS)."""
    verdict_cn = "恶意" if rec["verdict"] == "malicious" else "良性"
    verdict_color = "#b91c1c" if rec["verdict"] == "malicious" else "#15803d"
    # Family attribution is a probabilistic guess ("most-resembled known family"), not a forensic
    # identification — present it as a suspicion with its confidence, never a bald claim.
    if rec["family"] is None:
        family_line = "未知（不适用或置信度过低）"
    else:
        fam_conf = rec.get("familyConfidence")
        conf_suffix = f"（置信度 {fam_conf * 100:.0f}%）" if fam_conf is not None else ""
        family_line = f"疑似 {rec['family']}{conf_suffix}"

    if rec["llmVerdict"] is None:
        llm_line = "本次为批量检测，未运行 LLM 分析。"
    else:
        llm_verdict_cn = "恶意" if rec["llmVerdict"] == "malicious" else "良性"
        conf = f"{rec['llmConfidence'] * 100:.0f}%" if rec["llmConfidence"] is not None else "—"
        llm_line = f"独立判定：{llm_verdict_cn}（置信度 {conf}）"

    attck_rows = "".join(
        f"<tr><td>{escape(t['tactic'])}</td><td>{escape(t['technique'])}</td></tr>"
        for t in rec["attck"]
    )
    attck_table = (
        f"<table><thead><tr><th>战术 Tactic</th><th>技术 Technique</th></tr></thead>"
        f"<tbody>{attck_rows}</tbody></table>"
        if attck_rows
        else "<p class='muted'>无 ATT&amp;CK 标签。</p>"
    )
    report_text = escape(rec["llmReport"]) or "<span class='muted'>（无）</span>"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>检测报告 · {escape(rec['filename'])}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; color: #1f2937;
         max-width: 760px; margin: 40px auto; padding: 0 24px; line-height: 1.6; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .sub {{ color: #6b7280; font-size: 13px; margin-top: 0; }}
  .verdict {{ display: inline-block; padding: 4px 14px; border-radius: 999px; color: #fff;
             font-weight: 600; background: {verdict_color}; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
  th {{ background: #f9fafb; }}
  .grid {{ display: grid; grid-template-columns: 160px 1fr; row-gap: 6px; font-size: 14px; margin: 16px 0; }}
  .grid div:nth-child(odd) {{ color: #6b7280; }}
  .muted {{ color: #9ca3af; }}
  .report {{ white-space: pre-wrap; background: #f9fafb; border: 1px solid #e5e7eb;
            border-radius: 8px; padding: 16px; font-size: 14px; }}
  h2 {{ font-size: 16px; margin-top: 28px; }}
</style>
</head>
<body>
  <h1>MalGuard AI 检测报告</h1>
  <p class="sub">生成于 {escape(rec['createdAt'])} · 记录 #{rec['id']}</p>
  <p><span class="verdict">{verdict_cn}</span> &nbsp; 置信度 {rec['confidence'] * 100:.1f}%</p>
  <div class="grid">
    <div>文件名</div><div>{escape(rec['filename'])}</div>
    <div>SHA-256</div><div style="word-break: break-all;">{escape(rec['sha256'])}</div>
    <div>检测来源</div><div>{'单文件' if rec['source'] == 'single' else '批量'}</div>
    <div>疑似家族</div><div>{escape(family_line)}</div>
    <div>LightGBM 概率</div><div>{rec['lgbmScore'] * 100:.1f}%</div>
    <div>MLP 概率</div><div>{rec['mlpScore'] * 100:.1f}%</div>
    <div>模型一致性</div><div>{'一致' if rec['modelAgreement'] == 'agree' else '不一致'}</div>
    <div>LLM 判定</div><div>{escape(llm_line)}</div>
  </div>
  <h2>ATT&amp;CK 标签</h2>
  {attck_table}
  <h2>LLM 行为分析</h2>
  <div class="report">{report_text}</div>
</body>
</html>"""
