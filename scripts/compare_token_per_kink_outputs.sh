#!/bin/bash
#
# Compare the legacy token_per_kink output with the new script output.
# Usage:
#   scripts/compare_token_per_kink_outputs.sh <legacy_csv> <new_csv>
#
# Example:
#   scripts/compare_token_per_kink_outputs.sh \
#       output_aligned/legacy_row_old_sorted/AIME24_token_per_kink_rank_accuracy.csv \
#       output_aligned/new_fast/AIME24_rank_accuracies.csv
#

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <legacy_csv> <new_csv>" >&2
  exit 1
fi

LEGACY="$1"
NEW="$2"

if [[ ! -f "$LEGACY" ]]; then
  echo "Legacy CSV not found: $LEGACY" >&2
  exit 1
fi
if [[ ! -f "$NEW" ]]; then
  echo "New CSV not found: $NEW" >&2
  exit 1
fi

python3 <<PY
import csv, numpy as np, sys

legacy_path = r"$LEGACY"
new_path = r"$NEW"

with open(legacy_path, "r", encoding="utf-8") as f:
    legacy_rows = list(csv.DictReader(f))
with open(new_path, "r", encoding="utf-8") as f:
    new_rows = list(csv.DictReader(f))

if not legacy_rows or not new_rows:
    print("CSV(s) are empty.", file=sys.stderr)
    sys.exit(1)

if len(legacy_rows) != len(new_rows):
    print(f"Rank count mismatch: legacy {len(legacy_rows)} vs new {len(new_rows)}", file=sys.stderr)
    sys.exit(2)

def get_series(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=np.float64)

legacy_tpk = get_series(legacy_rows, "avg_token_per_kink")
new_tpk = get_series(new_rows, "avg_token_per_kink")
legacy_acc = get_series(legacy_rows, "accuracy")
new_acc = get_series(new_rows, "accuracy")

tp_diff = np.abs(legacy_tpk - new_tpk)
acc_diff = np.abs(legacy_acc - new_acc)

stats = {
    "rank_count": len(legacy_rows),
    "tpk_max_abs_diff": float(tp_diff.max()),
    "tpk_mean_abs_diff": float(tp_diff.mean()),
    "acc_max_abs_diff": float(acc_diff.max()),
    "acc_mean_abs_diff": float(acc_diff.mean()),
    "tpk_equal_ranks": int(np.sum(tp_diff < 1e-9)),
    "acc_equal_ranks": int(np.sum(acc_diff < 1e-9)),
}

print("comparison:")
for k, v in stats.items():
    print(f"  {k}: {v}")

if stats["acc_equal_ranks"] != stats["rank_count"] or stats["tpk_max_abs_diff"] > 1e-9:
    print("❌ Outputs are not perfectly aligned.", file=sys.stderr)
    sys.exit(3)

print("✅ Outputs are perfectly aligned.")
PY
