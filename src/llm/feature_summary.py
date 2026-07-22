"""Extract bounded, human-readable PE facts for the optional LLM analysis."""

import ipaddress
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import lief

from src.llm.attck_rules import AttckTag, tag_for_entropy, tags_for_imports

MAX_INDICATORS_PER_KIND = 8
STANDARD_SECTION_NAMES = {
    ".text", ".data", ".rdata", ".bss", ".idata", ".edata", ".pdata",
    ".rsrc", ".reloc", ".tls", ".crt", ".debug", ".gfids", ".00cfg",
}
SUSPICIOUS_STRING_MARKERS = (
    "powershell", "cmd.exe", "rundll32", "regsvr32", "certutil",
    "mshta", "schtasks", "vssadmin", "bcdedit", "mimikatz",
)


@dataclass
class StructuralSummary:
    file_size: int
    is_signed: bool
    suspicious_imports: list[str]
    max_section_entropy: float
    high_entropy_sections: list[str]
    attck_tags: list[AttckTag] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    ip_addresses: list[str] = field(default_factory=list)
    registry_paths: list[str] = field(default_factory=list)
    suspicious_strings: list[str] = field(default_factory=list)
    nonstandard_sections: list[str] = field(default_factory=list)
    compile_time_utc: str | None = None
    compile_time_anomaly: str | None = None
    export_count: int = 0
    has_version_info: bool = False

    def to_prompt_text(self) -> str:
        lines = [
            f"文件大小：{self.file_size} 字节",
            f"数字签名：{'有' if self.is_signed else '无'}",
            "版本信息：" + ("有" if self.has_version_info else "无"),
            "命中的敏感 API 导入："
            + ("、".join(self.suspicious_imports) if self.suspicious_imports else "无"),
            f"节区最大熵值：{self.max_section_entropy:.2f}（理论最大值 8.0）",
        ]
        if self.high_entropy_sections:
            lines.append("疑似加壳/高熵节区：" + "、".join(self.high_entropy_sections))
        if self.nonstandard_sections:
            lines.append("非标准节区名（仅作为异常线索）：" + "、".join(self.nonstandard_sections))
        if self.compile_time_utc:
            lines.append("PE 头编译时间（可被篡改）：" + self.compile_time_utc)
        if self.compile_time_anomaly:
            lines.append("编译时间异常：" + self.compile_time_anomaly)
        lines.append(f"导出函数数量：{self.export_count}")
        if self.urls:
            lines.append("提取到的 URL：" + "、".join(self.urls))
        if self.ip_addresses:
            lines.append("提取到的 IP 地址：" + "、".join(self.ip_addresses))
        if self.registry_paths:
            lines.append("提取到的注册表路径：" + "、".join(self.registry_paths))
        if self.suspicious_strings:
            lines.append("命中的命令/工具字符串：" + "、".join(self.suspicious_strings))
        if self.attck_tags:
            lines.append(
                "规则关联的 ATT&CK 技术："
                + "；".join(f"{t.tactic} ({t.technique})" for t in self.attck_tags)
            )
        return "\n".join(lines)


def _printable_strings(file_bytes: bytes) -> list[str]:
    """Return bounded ASCII/UTF-16LE strings without decoding the whole file as text."""
    ascii_strings = [
        match.decode("ascii", errors="ignore")
        for match in re.findall(rb"[ -~]{6,200}", file_bytes)
    ]
    wide_matches = re.findall(rb"(?:[ -~]\x00){6,200}", file_bytes)
    wide_strings = [match.decode("utf-16le", errors="ignore") for match in wide_matches]
    return ascii_strings + wide_strings


def _unique_limited(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))[:MAX_INDICATORS_PER_KIND]


def _extract_string_indicators(
    file_bytes: bytes,
) -> tuple[list[str], list[str], list[str], list[str]]:
    strings = _printable_strings(file_bytes)
    urls: list[str] = []
    ips: list[str] = []
    registry_paths: list[str] = []
    suspicious_strings: list[str] = []

    for value in strings:
        urls.extend(re.findall(r"https?://[^\s\"'<>]{4,180}", value, flags=re.IGNORECASE))
        registry_paths.extend(
            re.findall(
                r"(?:HKEY_(?:LOCAL_MACHINE|CURRENT_USER|CLASSES_ROOT|USERS)|HKLM|HKCU)\\[^\s\"']{3,160}",
                value,
                flags=re.IGNORECASE,
            )
        )
        for candidate in re.findall(
            r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", value
        ):
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue
            context = value.lower()
            if value.strip() == candidate or any(
                marker in context
                for marker in ("http", "connect", "socket", "server", "host", "c2")
            ):
                ips.append(candidate)
        lowered = value.lower()
        if any(marker in lowered for marker in SUSPICIOUS_STRING_MARKERS):
            suspicious_strings.append(value[:180])

    return tuple(
        _unique_limited(values)
        for values in (urls, ips, registry_paths, suspicious_strings)
    )


def _compile_time(timestamp: int) -> tuple[str | None, str | None]:
    if timestamp <= 0:
        return None, "时间戳为空或无效"
    try:
        value = datetime.fromtimestamp(timestamp, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None, "时间戳无法解析"

    anomaly = None
    now = datetime.now(timezone.utc)
    if value < datetime(1995, 1, 1, tzinfo=timezone.utc):
        anomaly = "时间早于 1995 年，可能被篡改"
    elif value > now + timedelta(days=1):
        anomaly = "时间位于未来，可能被篡改"
    return value.isoformat(timespec="seconds"), anomaly


def summarize(file_bytes: bytes) -> StructuralSummary:
    binary = lief.PE.parse(file_bytes)
    if binary is None:
        return StructuralSummary(
            file_size=len(file_bytes),
            is_signed=False,
            suspicious_imports=[],
            max_section_entropy=0.0,
            high_entropy_sections=[],
        )

    import_names = {
        entry.name for imp in binary.imports for entry in imp.entries if entry.name
    }
    sections = list(binary.sections)
    high_entropy_sections = [section.name for section in sections if section.entropy >= 7.0]
    max_entropy = max((section.entropy for section in sections), default=0.0)
    nonstandard_sections = _unique_limited(
        [
            section.name
            for section in sections
            if section.name.lower() not in STANDARD_SECTION_NAMES
        ]
    )

    attck_tags = tags_for_imports(import_names)
    entropy_tag = tag_for_entropy(max_entropy)
    if entropy_tag and entropy_tag not in attck_tags:
        attck_tags.append(entropy_tag)

    suspicious_api_names = {
        "RegCreateKeyExA", "RegSetValueExA", "OpenSCManagerA", "CreateServiceA",
        "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
        "InternetOpenA", "InternetConnectA", "HttpSendRequestA",
        "CryptAcquireContextA", "CryptEncrypt",
    }
    suspicious_imports = sorted(import_names & suspicious_api_names)
    urls, ips, registry_paths, suspicious_strings = _extract_string_indicators(file_bytes)
    compile_time_utc, compile_time_anomaly = _compile_time(
        binary.header.time_date_stamps
    )

    has_version_info = False
    if binary.has_resources:
        try:
            has_version_info = binary.resources_manager.has_version
        except (lief.lief_errors, RuntimeError):
            pass

    return StructuralSummary(
        file_size=len(file_bytes),
        is_signed=binary.has_signatures,
        suspicious_imports=suspicious_imports,
        max_section_entropy=max_entropy,
        high_entropy_sections=high_entropy_sections,
        attck_tags=attck_tags,
        urls=urls,
        ip_addresses=ips,
        registry_paths=registry_paths,
        suspicious_strings=suspicious_strings,
        nonstandard_sections=nonstandard_sections,
        compile_time_utc=compile_time_utc,
        compile_time_anomaly=compile_time_anomaly,
        export_count=len(binary.exported_functions),
        has_version_info=has_version_info,
    )
