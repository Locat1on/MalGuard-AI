"""Generate a natural-language behavior analysis report via an LLM (OpenRouter).

Design constraints (see CLAUDE.local.md "Design intent"):
  - Never on the hot path for bulk detection — only called for flagged/uncertain samples.
  - Results are cached by file hash so a live demo never depends on a live network call
    for a file it has already analyzed.
  - The LLM only narrates facts already established deterministically by
    src/llm/feature_summary.py + attck_rules.py — it is not asked to invent ATT&CK IDs.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hashlib

import requests

from src.config import load_config
from src.llm.feature_summary import StructuralSummary

CACHE_DIR = Path(r"D:\study\Integrated_Design\checkpoints\llm_cache")

SYSTEM_PROMPT = (
    "你是一个恶意软件静态分析助手。用户会给你一份 PE 文件的结构化事实（导入表、节区熵、"
    "签名情况、已匹配的 ATT&CK 战术），这些事实都已经由确定性规则提取好，不需要你验证或修改。"
    "你的任务只是用简洁的中文写一段 3-5 句话的行为分析narrative，说明这份样本的结构特征"
    "像什么类型的行为模式。不要编造事实之外的具体信息（比如具体的恶意软件家族名、"
    "C2 服务器地址等），只基于给出的事实做合理的技术解读。"
)


def _cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.json"


def _load_cached(file_hash: str) -> str | None:
    path = _cache_path(file_hash)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["report"]
    return None


def _save_cache(file_hash: str, report: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(file_hash).write_text(json.dumps({"report": report}, ensure_ascii=False), encoding="utf-8")


def generate_report(file_bytes: bytes, summary: StructuralSummary, verdict: str, confidence: float) -> str:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    cached = _load_cached(file_hash)
    if cached is not None:
        return cached

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "[LLM 分析不可用] 未设置 OPENROUTER_API_KEY 环境变量。"

    config = load_config("llm")
    user_prompt = (
        f"检测判定：{'恶意' if verdict == 'malicious' else '良性'}（置信度 {confidence:.1%}）\n\n"
        f"{summary.to_prompt_text()}"
    )

    try:
        response = requests.post(
            config["provider_base_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": config["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
            },
            timeout=config["timeout_seconds"],
        )
        response.raise_for_status()
        report = response.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError) as e:
        return f"[LLM 分析失败] {e}"

    _save_cache(file_hash, report)
    return report
