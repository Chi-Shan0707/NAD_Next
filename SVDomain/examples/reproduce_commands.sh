#!/usr/bin/env bash
set -euo pipefail

# SVDomain paper reproduction commands
# Run from repository root.

echo "[1] Create or activate environment"
# python3 -m venv .venv
source .venv/bin/activate
pip install -r SVDomain/env/requirements-paper.txt
python SVDomain/env/verify_imports.py

echo "[2] Optional repository-level checks"
bash cookbook/00_setup/verify.sh || true

echo "[3] Train canonical r1 bundles"
python3 SVDomain/train_es_svd_ms_rr_r1.py

echo "[4] Train coding branch (negative-result branch)"
python3 SVDomain/train_es_svd_coding_rr_r1.py

echo "[5] Export interpretability artifacts (smoke)"
python3 scripts/export_svd_explanations.py --max-problems 1

echo "[6] Export interpretability artifacts (full)"
# python3 scripts/export_svd_explanations.py

echo "[7] Start viewer"
# python3 cot_viewer/app.py

echo "[8] Suggested smoke checks"
python3 -m py_compile nad/explain/svd_explain.py scripts/export_svd_explanations.py cot_viewer/app.py
node --check cot_viewer/static/app.js

echo "Done. Review outputs in:"
echo "  - results/scans/earlystop/"
echo "  - results/interpretability/"
echo "  - submission/EarlyStop/"
