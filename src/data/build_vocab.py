"""Build API-name vocabulary from the raw train/test CSVs.

Streams both CSVs line-by-line so the 2-3GB files never need to fit in memory.
Output: data/processed/vocab.json  -> {"api_name": id, ...}
  id 0 is reserved for PAD, id 1 for UNK.
"""
import csv
import json
from collections import Counter
from pathlib import Path

RAW_DIR = Path(r"D:\study\Integrated_Design\data\raw")
OUT_PATH = Path(r"D:\study\Integrated_Design\data\processed\vocab.json")


def count_apis(csv_path: Path, api_col: int, counts: Counter) -> None:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            counts[row[api_col]] += 1


def main() -> None:
    counts: Counter = Counter()
    print("scanning security_train.csv ...")
    count_apis(RAW_DIR / "security_train.csv", api_col=2, counts=counts)
    print("scanning security_test.csv ...")
    count_apis(RAW_DIR / "security_test.csv", api_col=1, counts=counts)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for api, _ in counts.most_common():
        vocab[api] = len(vocab)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"vocab size (incl. PAD/UNK): {len(vocab)}")
    print(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
