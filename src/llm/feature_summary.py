"""Extract a human-readable structural summary from a PE file's raw bytes.

Separate from src/features/extract.py's numeric EMBER vector (used by the classifiers) —
this produces the interpretable facts (which suspicious imports are present, section
entropy, signature presence) that feed the LLM prompt and the deterministic ATT&CK tags.
"""

from dataclasses import dataclass, field

import lief

from src.llm.attck_rules import AttckTag, tag_for_entropy, tags_for_imports


@dataclass
class StructuralSummary:
    file_size: int
    is_signed: bool
    suspicious_imports: list[str]
    max_section_entropy: float
    high_entropy_sections: list[str]
    attck_tags: list[AttckTag] = field(default_factory=list)

    def to_prompt_text(self) -> str:
        lines = [
            f"文件大小：{self.file_size} 字节",
            f"数字签名：{'有' if self.is_signed else '无'}",
        ]
        if self.suspicious_imports:
            lines.append("命中的敏感 API 导入：" + "、".join(self.suspicious_imports))
        else:
            lines.append("命中的敏感 API 导入：无")
        lines.append(f"节区最大熵值：{self.max_section_entropy:.2f}（理论最大值 8.0）")
        if self.high_entropy_sections:
            lines.append("疑似加壳/高熵节区：" + "、".join(self.high_entropy_sections))
        if self.attck_tags:
            lines.append(
                "已匹配的 ATT&CK 战术："
                + "；".join(f"{t.tactic} ({t.technique})" for t in self.attck_tags)
            )
        return "\n".join(lines)


def summarize(file_bytes: bytes) -> StructuralSummary:
    # Pass bytes straight to lief — `list(file_bytes)` would expand the whole file into a
    # Python list of ints (O(filesize) objects) on every detection. (lief 0.17 accepts bytes
    # but not bytearray.)
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

    high_entropy_sections = [s.name for s in binary.sections if s.entropy >= 7.0]
    max_entropy = max((s.entropy for s in binary.sections), default=0.0)

    attck_tags = tags_for_imports(import_names)
    entropy_tag = tag_for_entropy(max_entropy)
    if entropy_tag and entropy_tag not in attck_tags:
        attck_tags.append(entropy_tag)

    suspicious_present = [name for name in import_names if name in {
        "RegCreateKeyExA", "RegSetValueExA", "OpenSCManagerA", "CreateServiceA",
        "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
        "IsDebuggerPresent", "CheckRemoteDebuggerPresent",
        "InternetOpenA", "InternetConnectA", "HttpSendRequestA",
        "CryptAcquireContextA", "CryptEncrypt",
    }]

    return StructuralSummary(
        file_size=len(file_bytes),
        is_signed=binary.has_signatures,
        suspicious_imports=suspicious_present,
        max_section_entropy=max_entropy,
        high_entropy_sections=high_entropy_sections,
        attck_tags=attck_tags,
    )
