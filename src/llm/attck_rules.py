"""Deterministic API-name / heuristic -> ATT&CK tactic-technique mapping.

Kept separate from the LLM call on purpose: which ATT&CK techniques apply is a factual
lookup, not something an LLM should be asked to invent (avoids hallucinated technique IDs).
The LLM's job (see report.py) is only to narrate what these facts already established.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttckTag:
    tactic: str
    technique: str


# Same suspicious-API set used to build demo_samples/suspicious_demo.exe, so the detector's
# explanation and the demo construction are symmetric.
API_ATTCK_MAP: dict[str, AttckTag] = {
    "RegCreateKeyExA": AttckTag("Persistence", "T1547 Boot or Logon Autostart Execution"),
    "RegSetValueExA": AttckTag("Persistence", "T1547 Boot or Logon Autostart Execution"),
    "OpenSCManagerA": AttckTag("Persistence", "T1543.003 Create or Modify System Process: Windows Service"),
    "CreateServiceA": AttckTag("Persistence", "T1543.003 Create or Modify System Process: Windows Service"),
    "VirtualAllocEx": AttckTag("Defense Evasion", "T1055 Process Injection"),
    "WriteProcessMemory": AttckTag("Defense Evasion", "T1055 Process Injection"),
    "CreateRemoteThread": AttckTag("Defense Evasion", "T1055 Process Injection"),
    "IsDebuggerPresent": AttckTag("Defense Evasion", "T1622 Debugger Evasion"),
    "CheckRemoteDebuggerPresent": AttckTag("Defense Evasion", "T1622 Debugger Evasion"),
    "InternetOpenA": AttckTag("Command and Control", "T1071.001 Application Layer Protocol: Web Protocols"),
    "InternetConnectA": AttckTag("Command and Control", "T1071.001 Application Layer Protocol: Web Protocols"),
    "HttpSendRequestA": AttckTag("Command and Control", "T1071.001 Application Layer Protocol: Web Protocols"),
    "CryptAcquireContextA": AttckTag("Impact", "T1486 Data Encrypted for Impact"),
    "CryptEncrypt": AttckTag("Impact", "T1486 Data Encrypted for Impact"),
}

HIGH_ENTROPY_THRESHOLD = 7.0
PACKING_TAG = AttckTag("Defense Evasion", "T1027 Obfuscated Files or Information")


def tags_for_imports(import_names: set[str]) -> list[AttckTag]:
    seen: list[AttckTag] = []
    for name in import_names:
        tag = API_ATTCK_MAP.get(name)
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def tag_for_entropy(max_section_entropy: float) -> AttckTag | None:
    return PACKING_TAG if max_section_entropy >= HIGH_ENTROPY_THRESHOLD else None
