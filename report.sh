#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-outputs/beerschool/dataset}"
TRAIN_CSV="$DATASET_DIR/metadata_train.csv"
EVAL_CSV="$DATASET_DIR/metadata_eval.csv"
TRAIN_SCORED_CSV="$DATASET_DIR/metadata_train_scored.csv"
EVAL_SCORED_CSV="$DATASET_DIR/metadata_eval_scored.csv"
TRAIN_AUTO_CSV="$DATASET_DIR/metadata_train_auto.csv"
EVAL_AUTO_CSV="$DATASET_DIR/metadata_eval_auto.csv"

if [ ! -f "$TRAIN_SCORED_CSV" ] && [ -f "$DATASET_DIR/metadata_train_review.csv" ]; then
  TRAIN_SCORED_CSV="$DATASET_DIR/metadata_train_review.csv"
fi
if [ ! -f "$EVAL_SCORED_CSV" ] && [ -f "$DATASET_DIR/metadata_eval_review.csv" ]; then
  EVAL_SCORED_CSV="$DATASET_DIR/metadata_eval_review.csv"
fi
if [ ! -f "$TRAIN_AUTO_CSV" ] && [ -f "$DATASET_DIR/metadata_train_curated.csv" ]; then
  TRAIN_AUTO_CSV="$DATASET_DIR/metadata_train_curated.csv"
fi
if [ ! -f "$EVAL_AUTO_CSV" ] && [ -f "$DATASET_DIR/metadata_eval_curated.csv" ]; then
  EVAL_AUTO_CSV="$DATASET_DIR/metadata_eval_curated.csv"
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "error: .venv/bin/python not found. Activate or install the project venv first." >&2
  exit 1
fi

for file in \
  "$TRAIN_CSV" \
  "$EVAL_CSV" \
  "$TRAIN_SCORED_CSV" \
  "$TRAIN_AUTO_CSV"
do
  if [ ! -f "$file" ]; then
    echo "error: missing dataset artifact: $file" >&2
    echo "usage: ./report.sh [DATASET_DIR]" >&2
    exit 1
  fi
done

.venv/bin/python - "$DATASET_DIR" "$TRAIN_SCORED_CSV" "$EVAL_SCORED_CSV" "$TRAIN_AUTO_CSV" "$EVAL_AUTO_CSV" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path


def read_rows(path: Path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="|"))


def print_header(title: str):
    print()
    print(title)
    print("=" * len(title))


dataset_dir = Path(sys.argv[1])
train_csv = dataset_dir / "metadata_train.csv"
eval_csv = dataset_dir / "metadata_eval.csv"
train_scored_csv = Path(sys.argv[2])
eval_scored_csv = Path(sys.argv[3])
train_auto_csv = Path(sys.argv[4])
eval_auto_csv = Path(sys.argv[5])

train_rows = read_rows(train_csv)
eval_rows = read_rows(eval_csv)
train_scored_rows = read_rows(train_scored_csv)
eval_scored_rows = read_rows(eval_scored_csv) if eval_scored_csv.exists() else []
train_auto_rows = read_rows(train_auto_csv)
eval_auto_rows = read_rows(eval_auto_csv) if eval_auto_csv.exists() else []

print_header("Dataset Summary")
print(f"dataset_dir: {dataset_dir}")
print(f"train_rows: {len(train_rows)}")
print(f"eval_rows: {len(eval_rows)}")
print(f"train_scored_rows: {len(train_scored_rows)}")
print(f"eval_scored_rows: {len(eval_scored_rows)}")
print(f"train_auto_rows: {len(train_auto_rows)}")
print(f"eval_auto_rows: {len(eval_auto_rows)}")

print_header("Auto Status Distribution")
train_auto_status = Counter(row.get("auto_status", row.get("review_status", "missing")) for row in train_scored_rows)
eval_auto_status = Counter(row.get("auto_status", row.get("review_status", "missing")) for row in eval_scored_rows)
print(f"train_auto_status: {dict(train_auto_status)}")
print(f"eval_auto_status: {dict(eval_auto_status)}")

print_header("Top Train Flags")
train_flags = Counter(
    flag
    for row in train_scored_rows
    for flag in row.get("flags", "").split(",")
    if flag
)
for flag, count in train_flags.most_common(12):
    print(f"{flag}: {count}")
if not train_flags:
    print("none")

print_header("Top Eval Flags")
eval_flags = Counter(
    flag
    for row in eval_scored_rows
    for flag in row.get("flags", "").split(",")
    if flag
)
for flag, count in eval_flags.most_common(12):
    print(f"{flag}: {count}")
if not eval_flags:
    print("none")

print_header("Train Score Summary")
train_scores = [int(row.get("score", 0)) for row in train_scored_rows if row.get("score")]
if train_scores:
    print(f"min: {min(train_scores)}")
    print(f"max: {max(train_scores)}")
    print(f"avg: {sum(train_scores) / len(train_scores):.2f}")
else:
    print("no scored train rows")

print_header("Sample Auto Texts")
for index, row in enumerate(train_auto_rows[:15], start=1):
    print(f"{index}. {row['text']}")
if not train_auto_rows:
    print("no auto-kept train rows")

print_header("Sample Scored Rows")
for row in train_scored_rows[:15]:
    print(
        f"{row.get('auto_status', row.get('review_status', 'missing'))} "
        f"score={row.get('score', 'n/a')} flags={row.get('flags', '')} -> {row['text']}"
    )

print_header("Sample Audio Paths")
for row in train_auto_rows[:10]:
    print(dataset_dir / row["audio_file"])
if not train_auto_rows:
    print("no auto-kept train rows")
PY
