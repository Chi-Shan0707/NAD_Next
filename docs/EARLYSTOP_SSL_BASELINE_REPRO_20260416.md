# EarlyStop SSL Round1 — Baseline Reproduction Notes (2026-04-16)

## Scope

This note records the concrete baseline commands run before new SSL / semi-supervised work.

## Commands Actually Run

### 1) Canonical `es_svd_ms_rr_r1` smoke attempt (`cap2`) — failed as expected

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python SVDomain/train_es_svd_ms_rr_r1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --holdout-split 0.15 \
  --split-seed 42 \
  --n-splits 5 \
  --random-state 42 \
  --feature-workers 8 \
  --fit-workers 4 \
  --feature-chunk-problems 8 \
  --feature-cache-dir results/cache/earlystop_ssl_baseline/es_svd_ms_rr_r1_cap2 \
  --max-problems-per-cache 2 \
  --out-math-model models/ml_selectors/earlystop_ssl_baselines/es_svd_math_rr_r1_cap2.pkl \
  --out-science-model models/ml_selectors/earlystop_ssl_baselines/es_svd_science_rr_r1_cap2.pkl \
  --out-combined-model models/ml_selectors/earlystop_ssl_baselines/es_svd_ms_rr_r1_cap2.pkl \
  --out-summary results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_cap2_summary.json \
  --out-eval results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_cap2_eval.json \
  --out-doc docs/EARLYSTOP_SSL_BASELINE_ES_SVD_MS_RR_R1_CAP2_20260416.md
```

Observed failure:

- `RuntimeError: math_splitfit@10% found no valid SVD candidate`
- Interpretation: `cap2` is too small for stable route search at early anchors, so later repros switched back to the canonical cached full-store path.

### 2) Canonical `es_svd_ms_rr_r1` repro from cached feature store — succeeded

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python SVDomain/train_es_svd_ms_rr_r1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --holdout-split 0.15 \
  --split-seed 42 \
  --n-splits 5 \
  --random-state 42 \
  --feature-workers 4 \
  --fit-workers 4 \
  --feature-chunk-problems 24 \
  --feature-cache-dir results/cache/es_svd_ms_rr_r1 \
  --out-math-model models/ml_selectors/earlystop_ssl_baselines/es_svd_math_rr_r1_repro.pkl \
  --out-science-model models/ml_selectors/earlystop_ssl_baselines/es_svd_science_rr_r1_repro.pkl \
  --out-combined-model models/ml_selectors/earlystop_ssl_baselines/es_svd_ms_rr_r1_repro.pkl \
  --out-summary results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_repro_summary.json \
  --out-eval results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_repro_eval.json \
  --out-doc docs/EARLYSTOP_SSL_BASELINE_ES_SVD_MS_RR_R1_REPRO_20260416.md
```

Key reproduced numbers (`combined_noncoding`):

- `es_svd_ms_rr_r1`: `AUC-AUROC=0.9207`, `AUC-SelAcc=0.9955`, `AUROC@100=0.9474`, `Stop@100=0.7352`
- `tok_conf_prefix_mean_v1`: `0.7127 / 0.9469 / 0.7221 / 0.7389`
- `earlystop_svd_lowrank_lr_v1`: `0.7758 / 0.9582 / 0.8801 / 0.7259`
- `earlystop_prefix10_svd_round1`: `0.8988 / 0.9951 / 0.9264 / 0.7741`

### 3) Strong-feature cap8 repro — succeeded but winner drifted

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/run_earlystop_strongfeat_round1.py \
  --main-cache-root MUI_HUB/cache \
  --extra-cache-root /home/jovyan/public-ro/MUI_HUB/cache_train \
  --holdout-split 0.20 \
  --split-seed 42 \
  --n-splits 5 \
  --random-state 42 \
  --workers 4 \
  --feature-chunk-problems 24 \
  --feature-cache-dir results/cache/earlystop_strongfeat_round1 \
  --max-problems-per-cache 8 \
  --out-model models/ml_selectors/earlystop_ssl_baselines/earlystop_strongfeat_round1_cap8_repro.pkl \
  --out-summary results/scans/earlystop_ssl/baseline_strongfeat_cap8_repro_summary.json \
  --out-eval results/scans/earlystop_ssl/baseline_strongfeat_cap8_repro_eval.json \
  --out-doc docs/EARLYSTOP_SSL_BASELINE_STRONGFEAT_CAP8_REPRO_20260416.md
```

Key holdout candidates:

- `strongfeat_noncoding_anchor4_coding_v1_ref020`:
  - `AUC-AUROC=0.9419`
  - `AUC-SelAcc=0.9521`
  - `AUROC@100=0.9486`
  - `Stop@100=0.7222`
- `strongfeat_noncoding_anchor4_coding_v1_ref030`:
  - `AUC-AUROC=0.9113`
  - `AUC-SelAcc=0.9590`
  - `AUROC@100=0.9203`
  - `Stop@100=0.6667`

Observed rerun difference:

- Historical doc winner: `ref020`
- This rerun’s summary-selected winner: `ref030`
- Summary reason: it beat `round1b_cap8` and `tok_conf`, but **did not** conservatively beat `earlystop_svd_lowrank_lr_v1` under the script’s export guardrail.
- Interpretation: this cap8 lane is useful as a strong reference, but not stable enough to be the canonical model-selection protocol for new SSL claims.

### 4) Cross-anchor baseline repro — succeeded

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python SVDomain/experiments/run_cross_anchor_transfer.py \
  --bundle-path models/ml_selectors/es_svd_ms_rr_r1.pkl \
  --cache-root MUI_HUB/cache \
  --domains math,science \
  --source-anchors 10,40,70,100 \
  --target-anchors 10,40,70,100 \
  --n-splits 5 \
  --random-state 42 \
  --feature-workers 4 \
  --suite-workers 4 \
  --feature-chunk-problems 24 \
  --feature-cache-dir results/cache/cross_anchor_transfer \
  --out-matrix results/tables/earlystop_ssl/baseline_cross_anchor_transfer_matrix.csv \
  --out-deltas results/tables/earlystop_ssl/baseline_cross_anchor_transfer_deltas.csv \
  --out-summary results/tables/earlystop_ssl/baseline_cross_anchor_transfer_summary.csv \
  --out-note docs/EARLYSTOP_SSL_BASELINE_CROSS_ANCHOR_REPRO_20260416.md
```

Key AUROC summary rows:

- `math diagonal_mean`: frozen `0.9402`, task-specific `0.9413`, no-svd `0.9425`
- `math offdiag_focus_mean`: frozen `0.9556`, task-specific `0.9589`, no-svd `0.9601`
- `science diagonal_mean`: frozen `0.7334`, task-specific `0.7411`, no-svd `0.7452`
- `science offdiag_focus_mean`: frozen `0.7438`, task-specific `0.7635`, no-svd `0.7659`

## Artifacts Written

- `notes/EARLYSTOP_SSL_RECON_20260416.md`
- `results/tables/earlystop_ssl/baseline_summary.csv`
- `results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_repro_summary.json`
- `results/scans/earlystop_ssl/baseline_es_svd_ms_rr_r1_repro_eval.json`
- `docs/EARLYSTOP_SSL_BASELINE_ES_SVD_MS_RR_R1_REPRO_20260416.md`
- `results/scans/earlystop_ssl/baseline_strongfeat_cap8_repro_summary.json`
- `results/scans/earlystop_ssl/baseline_strongfeat_cap8_repro_eval.json`
- `docs/EARLYSTOP_SSL_BASELINE_STRONGFEAT_CAP8_REPRO_20260416.md`
- `results/tables/earlystop_ssl/baseline_cross_anchor_transfer_matrix.csv`
- `results/tables/earlystop_ssl/baseline_cross_anchor_transfer_deltas.csv`
- `results/tables/earlystop_ssl/baseline_cross_anchor_transfer_summary.csv`
- `docs/EARLYSTOP_SSL_BASELINE_CROSS_ANCHOR_REPRO_20260416.md`

## Working Conclusion Before New SSL Runs

- Use canonical `85/15` grouped protocol as the primary lane for new claims.
- Keep strongfeat `cap8` as a cheap reference / stress test, not as the main selection protocol.
- Treat `tok_conf_prefix_mean_v1`, `es_svd_ms_rr_r1`, and cross-anchor `frozen/task-specific/no_svd` as the mandatory baselines for the next stage.
