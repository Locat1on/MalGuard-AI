"""Validated runtime settings for backend deployment."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_DB = PROJECT_ROOT / "data" / "history.db"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def parse_cors_origins(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated HTTP origins and reject ambiguous CORS settings."""
    if value is None or not value.strip():
        return DEFAULT_CORS_ORIGINS

    origins: list[str] = []
    for raw_origin in value.split(","):
        origin = raw_origin.strip().rstrip("/")
        parsed = urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
            or any(character.isspace() for character in origin)
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
        ):
            raise ValueError(
                "MALGUARD_CORS_ORIGINS 只能包含逗号分隔的 http(s) 来源，"
                "例如 http://192.168.1.10:5173。"
            )
        if origin not in origins:
            origins.append(origin)
    if not origins:
        raise ValueError("MALGUARD_CORS_ORIGINS 至少需要一个有效来源。")
    return tuple(origins)


def parse_inference_concurrency(value: str | None) -> int:
    """Keep concurrency bounded because all requests share one loaded model set."""
    if value is None or not value.strip():
        return 1
    try:
        concurrency = int(value)
    except ValueError as error:
        raise ValueError("MALGUARD_INFERENCE_CONCURRENCY 必须是 1 到 8 的整数。") from error
    if not 1 <= concurrency <= 8:
        raise ValueError("MALGUARD_INFERENCE_CONCURRENCY 必须是 1 到 8 的整数。")
    return concurrency


def parse_detection_concurrency(value: str | None) -> int:
    """Bound expensive upload/extraction/LLM request concurrency per process."""
    if value is None or not value.strip():
        return 2
    try:
        concurrency = int(value)
    except ValueError as error:
        raise ValueError(
            "MALGUARD_DETECTION_CONCURRENCY 必须是 1 到 32 的整数。"
        ) from error
    if not 1 <= concurrency <= 32:
        raise ValueError("MALGUARD_DETECTION_CONCURRENCY 必须是 1 到 32 的整数。")
    return concurrency


def parse_history_db_path(value: str | None) -> Path:
    """Resolve an optional history DB path without tying storage to model imports."""
    if value is None or not value.strip():
        return DEFAULT_HISTORY_DB
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_api_key(value: str | None) -> str | None:
    """Return an optional secret, rejecting keys too short for network use."""
    if value is None or value == "":
        return None
    if len(value) < 16:
        raise ValueError("MALGUARD_API_KEY 启用时至少需要 16 个字符。")
    if not value.isascii():
        raise ValueError("MALGUARD_API_KEY 只能使用 ASCII 字符。")
    return value


@dataclass(frozen=True)
class Settings:
    cors_origins: tuple[str, ...]
    inference_concurrency: int
    detection_concurrency: int
    history_db_path: Path
    api_key: str | None = field(repr=False)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if environ is None else environ
        return cls(
            cors_origins=parse_cors_origins(source.get("MALGUARD_CORS_ORIGINS")),
            inference_concurrency=parse_inference_concurrency(
                source.get("MALGUARD_INFERENCE_CONCURRENCY")
            ),
            detection_concurrency=parse_detection_concurrency(
                source.get("MALGUARD_DETECTION_CONCURRENCY")
            ),
            history_db_path=parse_history_db_path(
                source.get("MALGUARD_HISTORY_DB")
            ),
            api_key=parse_api_key(source.get("MALGUARD_API_KEY")),
        )


settings = Settings.from_environ()
