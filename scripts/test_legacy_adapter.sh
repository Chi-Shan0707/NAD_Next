#!/bin/bash
#
# Sanity check for nad.compat.legacy_cache_adapter.
# Verifies that legacy-style APIs can read a NAD_Next cache and return
# structures compatible with the old efficient cache utilities.
#
# Usage:
#   scripts/test_legacy_adapter.sh
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load config if available
if [ -f "${ROOT_DIR}/config.sh" ]; then
    source "${ROOT_DIR}/config.sh"
elif [ -f "${ROOT_DIR}/config_template.sh" ]; then
    source "${ROOT_DIR}/config_template.sh"
fi

# Use environment variable or default to ./MUI_public structure
MUI_PUBLIC_NEURON="${MUI_PUBLIC_NEURON:-./MUI_public/neuron}"
LEGACY_DIR="${LEGACY_MUI_PATH:-${MUI_PUBLIC_NEURON}/DeepSeek-R1-0528-Qwen3-8B/aime24/neuron_output_1_act_no_rms_20250902_025610}"
NEXT_CACHE="${ROOT_DIR}/cache_aime24"

if [[ ! -d "${NEXT_CACHE}" || ! -f "${NEXT_CACHE}/manifest.json" ]]; then
  echo "Missing NAD_Next cache at ${NEXT_CACHE}. Please place cache_aime24/ here."
  exit 1
fi

export NAD_NEXT_CACHE_ROOT="${NEXT_CACHE}"
export LEGACY_DIR="${LEGACY_DIR}"

python3 <<'PY'
import os
from pathlib import Path

from nad.compat.legacy_cache_adapter import (
    load_efficient_cache,
    load_neuron_data_parallel,
)

legacy_dir = os.environ.get("LEGACY_DIR", "./MUI_public/neuron/DeepSeek-R1-0528-Qwen3-8B/aime24/neuron_output_1_act_no_rms_20250902_025610")

correctness_map, meta, eval_file, topk_stub = load_efficient_cache(legacy_dir)

assert len(correctness_map) == len(meta.get("samples", [])) == 1920, "Unexpected sample count"
assert Path(eval_file).name in {"evaluation_report_compact.json", "evaluation_report.json"}
assert set(topk_stub.keys()) == {"metadata", "method1", "method2"}

neuron_data, token_metrics = load_neuron_data_parallel(legacy_dir)

assert len(neuron_data) == 1920, "Unexpected #runs"
assert token_metrics is None

first_sample = neuron_data[0]
assert len(first_sample) == 64, "Expected 64 slices in first sample"

entries = first_sample[min(first_sample.keys())][:5]
expected = [
    (16, 5998, 0.828125),
    (20, 10448, 0.78125),
    (23, 1551, 0.734375),
    (23, 1572, 1.078125),
    (23, 2118, 0.77734375),
]
assert entries == expected, "Slice entries mismatch"

print("legacy_cache_adapter smoke test passed ✔")
PY
