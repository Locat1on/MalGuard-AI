"""Build a demo PE whose STATIC feature profile looks structurally like malware, for testing
this project's detector. NOT malware: starts from a real, unmodified system binary
(attrib.exe) and only:
  - adds extra import-table entries (persistence / process-injection / C2 / anti-debug /
    crypto APIs) that are never actually called by any code in the binary
  - adds a high-entropy RWX section (a classic packer/shellcode indicator) filled with random
    bytes that are never executed — the original entry point and code path are untouched
  - ensures no Authenticode signature is present

No existing behavior is changed: if run, it does exactly what the unmodified attrib.exe does.
This is standard test-vector construction for validating a defensive ML classifier, not
malware — nothing here performs any actually harmful action.

Run: .venv\\Scripts\\python.exe demo_samples/build_suspicious_pe.py
"""

import os
from pathlib import Path

import lief

SOURCE = Path(r"C:\Windows\System32\attrib.exe")
OUTPUT = Path(r"D:\study\Integrated_Design\demo_samples\suspicious_demo.exe")

EXTRA_IMPORTS: dict[str, list[str]] = {
    "advapi32.dll": [
        "RegCreateKeyExA",
        "RegSetValueExA",
        "RegDeleteKeyA",
        "OpenSCManagerA",
        "CreateServiceA",
        "CryptAcquireContextA",
        "CryptEncrypt",
    ],
    "kernel32.dll": [
        "VirtualAllocEx",
        "WriteProcessMemory",
        "CreateRemoteThread",
        "IsDebuggerPresent",
        "CheckRemoteDebuggerPresent",
    ],
    "wininet.dll": [
        "InternetOpenA",
        "InternetConnectA",
        "HttpSendRequestA",
    ],
}


def main() -> None:
    binary = lief.PE.parse(str(SOURCE))
    if binary is None:
        raise RuntimeError(f"lief failed to parse {SOURCE}")

    # 1. Strip any Authenticode signature (defensive; attrib.exe already has none).
    cert_dir = binary.data_directory(lief.PE.DataDirectory.TYPES.CERTIFICATE_TABLE)
    cert_dir.rva = 0
    cert_dir.size = 0

    # 2. Graft suspicious import-table entries (never called by any code here).
    existing_libs = {imp.name.lower(): imp for imp in binary.imports}
    for dll_name, functions in EXTRA_IMPORTS.items():
        imp = existing_libs.get(dll_name.lower())
        if imp is None:
            imp = binary.add_import(dll_name)
        for func in functions:
            imp.add_entry(func)

    # 3. Add high-entropy RWX sections (packer-like signal), large relative to the original
    #    file, never executed by the (unchanged) entry point.
    for i, size in enumerate((131072, 65536, 65536)):
        packed_section = lief.PE.Section(f".ex{i}")
        packed_section.content = list(os.urandom(size))
        packed_section.characteristics = (
            lief.PE.Section.CHARACTERISTICS.MEM_EXECUTE
            | lief.PE.Section.CHARACTERISTICS.MEM_READ
            | lief.PE.Section.CHARACTERISTICS.MEM_WRITE
            | lief.PE.Section.CHARACTERISTICS.CNT_INITIALIZED_DATA
        )
        binary.add_section(packed_section)

    config = lief.PE.Builder.config_t()
    config.imports = True
    builder = lief.PE.Builder(binary, config)
    builder.build()
    builder.write(str(OUTPUT))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
