"""Extract the 'family' label from the raw EMBER2024 JSONL files, without re-running feature
extraction. Reuses thrember's own file gathering/iteration so row order matches X_train.dat/
X_test.dat exactly (both were built from the same sorted file list, read in the same order).

Writes plain JSON lists (family name string or null per row) — kept separate from thrember's
own y_train.dat/y_test.dat (binary label) so nothing existing gets overwritten.

Run: .venv\\Scripts\\python.exe src/data/extract_family_labels.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thrember.model import gather_feature_paths, raw_feature_iterator

DATA_DIR = Path(
    os.environ.get(
        "EMBER2024_DATA_DIR",
        Path(__file__).resolve().parents[2] / "data" / "raw" / "ember2024",
    )
)


def extract_families(subset: str) -> list[str | None]:
    feature_paths = gather_feature_paths(DATA_DIR, subset)
    families: list[str | None] = []
    for raw_line in raw_feature_iterator(feature_paths):
        record = json.loads(raw_line)
        families.append(record.get("family"))
    return families


def main() -> None:
    for subset in ("train", "test"):
        print(f"Extracting family labels for {subset} set")
        families = extract_families(subset)
        out_path = DATA_DIR / f"family_{subset}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(families, f)
        non_null = sum(1 for f in families if f)
        print(f"  {len(families)} rows, {non_null} with a family label -> {out_path}")


if __name__ == "__main__":
    main()
