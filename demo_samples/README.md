# demo_samples/

`suspicious_demo.exe` is **not malware**. It exists only to give the live demo a file that
this project's detector flags as malicious, without needing to source or handle a real
malicious binary.

## How it was built

`build_suspicious_pe.py` starts from a real, unmodified Windows system binary
(`C:\Windows\System32\attrib.exe`) and, using `lief`, makes three structural changes:

1. **Grafts import-table entries** for APIs commonly associated with persistence, process
   injection, C2 communication, and anti-debugging (`RegSetValueExA`, `CreateServiceA`,
   `VirtualAllocEx`, `WriteProcessMemory`, `CreateRemoteThread`, `InternetOpenA`, etc.).
2. **Adds three RWX sections filled with random bytes** (entropy 8.0 — the theoretical
   maximum, a classic packer/shellcode indicator).
3. **Ensures no Authenticode signature** is present.

## Why this is safe

- **The entry point is byte-for-byte unchanged** from the original `attrib.exe`
  (verified: both point to RVA `0x13b0`). If run, this program does exactly what the real
  `attrib.exe` does — nothing else.
- **None of the grafted import entries are ever called.** They exist only as import-table
  entries; no code path in the binary references them.
- **The random-byte sections are never executed, read, or written to** by any code path.
  They exist purely to shift the file's statistical/entropy profile.
- No file is encrypted or deleted, no registry persistence is installed, no other process is
  touched, and no network connection is ever made.

## Result

Against this project's trained models (`checkpoints/lightgbm.txt` + `checkpoints/mlp.pt`):

| File | LightGBM p(malicious) | MLP p(malicious) | Verdict |
|---|---|---|---|
| `attrib.exe` (original, unmodified) | — | — | benign, 99.15% confidence |
| imports-only graft (first attempt) | — | — | benign, 97.93% — barely moved |
| imports + 8KB packed section | 66.2% | 33.2% | benign, 50.28% — **models disagree** |
| imports + 262KB packed sections (final) | 96.2% | 99.9% | **malicious, 98.04% — both models agree** |

The middle row is worth keeping in the write-up: it's a genuine, unforced case of the two
models disagreeing, which is exactly the "flag for review" scenario the dual-model design is
built around — not something manufactured for the demo.

## Regenerating

```powershell
.venv\Scripts\python.exe demo_samples\build_suspicious_pe.py
```
