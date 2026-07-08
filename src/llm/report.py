"""Generate an independent LLM verdict + behavior analysis report via OpenRouter.

Design constraints (see CLAUDE.local.md "Design intent"):
  - Never on the hot path for bulk detection — only called for flagged/uncertain samples.
  - Results are cached by file hash so a live demo never depends on a live network call
    for a file it has already analyzed.
  - The LLM is given the same deterministic facts as attck_rules.py (imports, entropy,
    signature) but NOT the two ML models' verdict, so its own malicious/benign judgment is a
    genuinely independent third opinion (shown for comparison only, never averaged into the
    final probability — see predictor.py). It is not asked to invent ATT&CK IDs or facts
    beyond what's given (e.g. specific malware family names, C2 addresses).
"""

import json
import os
import re
import sys
from dataclasses import dataclass
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
    "请你仅根据这些事实独立判断该样本是恶意还是良性，不要参考任何其他信息来源。"
    "不要编造事实之外的具体信息（比如具体的恶意软件家族名、C2 服务器地址等），"
    "只基于给出的事实做合理的技术解读。"
    "严格按以下 JSON 格式输出，不要输出任何 JSON 之外的文字：\n"
    '{"verdict": "malicious" 或 "benign", "confidence": 0-100 的整数（你对该判断的把握程度）, '
    '"narrative": "3-5 句话的中文行为分析"}'
)


@dataclass
class LLMAnalysis:
    verdict: str | None  # "malicious" | "benign" | None (analysis unavailable/failed)
    confidence: float | None  # 0-1, None if verdict is None
    narrative: str


def _cache_path(file_hash: str) -> Path:
    return CACHE_DIR / f"{file_hash}.json"


def _load_cached(file_hash: str) -> LLMAnalysis | None:
    path = _cache_path(file_hash)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "verdict" not in data:
        # Cache entry from the old (pre-independent-verdict) report format — treat as a
        # miss so it gets regenerated in the current schema rather than crashing.
        return None
    return LLMAnalysis(verdict=data["verdict"], confidence=data["confidence"], narrative=data["narrative"])


def _save_cache(file_hash: str, analysis: LLMAnalysis) -> None:
    # Only cache successful analyses — a transient network failure shouldn't be pinned
    # forever as this file's "result".
    if analysis.verdict is None:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(file_hash).write_text(
        json.dumps(
            {"verdict": analysis.verdict, "confidence": analysis.confidence, "narrative": analysis.narrative},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _parse_response(raw: str) -> LLMAnalysis:
    # Reasoning models sometimes wrap JSON in a ```json fence despite the "no extra text"
    # instruction — strip that before parsing rather than failing the whole analysis.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    payload = json.loads(match.group(0) if match else raw)
    verdict = payload["verdict"]
    if verdict not in ("malicious", "benign"):
        raise ValueError(f"unexpected verdict value: {verdict!r}")
    return LLMAnalysis(
        verdict=verdict,
        confidence=max(0.0, min(1.0, float(payload["confidence"]) / 100)),
        narrative=str(payload["narrative"]).strip(),
    )


def generate_report(file_bytes: bytes, summary: StructuralSummary) -> LLMAnalysis:
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    cached = _load_cached(file_hash)
    if cached is not None:
        return cached

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return LLMAnalysis(verdict=None, confidence=None, narrative="[LLM 分析不可用] 未设置 OPENROUTER_API_KEY 环境变量。")

    config = load_config("llm")

    try:
        response = requests.post(
            config["provider_base_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": config["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": summary.to_prompt_text()},
                ],
                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
            },
            timeout=config["timeout_seconds"],
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if content is None:
            # Reasoning models (e.g. glm-5.2) can burn the whole max_tokens budget on the
            # "reasoning" field and return content=None — raise max_tokens in configs/llm.yaml
            # if this recurs (see CLAUDE.local.md).
            raise ValueError("模型返回了空 content（可能是 max_tokens 预算被 reasoning 字段耗尽）")
        analysis = _parse_response(content.strip())
    except (requests.RequestException, KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        return LLMAnalysis(verdict=None, confidence=None, narrative=f"[LLM 分析失败] {e}")

    _save_cache(file_hash, analysis)
    return analysis
