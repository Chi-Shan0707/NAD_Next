# EarlyStop SSL / Semi-Supervised Round1 (2026-04-16)

## Scope

This round follows the canonical EarlyStop / SVDomain protocol and asks:

1. Can a CPU-first linear SSL basis help EarlyStop under strict grouped holdout?
2. Does a shared/frozen basis stay close to task-specific fitting?
3. Does pseudo-labeling help in low-label regimes?
4. Are gains domain-specific, or do they generalize across math / science / coding?

## Protocol

- **Repo root**: `/home/jovyan/work/NAD_Next`
- **Data roots**: cached canonical feature store from `results/cache/es_svd_ms_rr_r1`
- **Split**: grouped `85/15` holdout by `dataset + problem_id`, `split_seed=42`
- **Domains**: `math`, `science`; `coding` only cheap confirmation
- **Metrics**:
  - `AUC of AUROC`
  - `AUC of SelAcc`
  - `Earliest > 0.6`
  - `AUROC@100%`
  - `Stop Acc@100%`
- **Safety**:
  - prefix-safe fixed feature family only
  - no leaderboard / blind-test selection
  - same-run views are default positives
  - same-problem positives only as explicit ablation

## New Code

- `nad/ops/earlystop_ssl.py`
  - split-safe problem-key sampling/subsetting
  - linear `ridge CCA`, `cross-SVD / reduced-rank`, and `denoise SVD`
  - lightweight LR head fitting
  - high-confidence pseudo-label helpers
- `scripts/train_earlystop_ssl_basis.py`
  - shared vs task-specific basis study
  - same-run vs same-problem positive ablation
  - raw / rank / raw+rank / token-only baselines
- `scripts/train_earlystop_semisup.py`
  - hidden-label simulation
  - budgets `5 / 10 / 25 / 50 / 100%`
  - frozen SSL, task-specific SSL, pseudo-label, agreement-filtered pseudo-label

## Views and Methods

### Views

- `raw ↔ rank`
- `token_only ↔ token_plus_traj`
- `adjacent anchors (10/40, 40/70, 70/100)` as one shared adjacent-anchor basis
- masked / denoising reconstruction on full `raw+rank`

### Methods

- `cca_ridge`
- `rrr_cross_svd`
- `denoise_svd`

## Basis Results

See:

- `results/tables/earlystop_ssl_summary.csv`
- `results/tables/earlystop_ssl_ablation.csv`
- `results/scans/earlystop_ssl/ssl_basis_detail.json`

### Math

- **Best simple baseline**: `raw+rank LR`
  - `AUC-AUROC = 0.9593`
  - `AUC-SelAcc = 0.9973`
  - `AUROC@100 = 0.9837`
- **Best SSL family**: `denoise_full @ rank 16`
  - shared basis: `0.9580`
  - task-specific basis: `0.9584`
- **Interpretation**:
  - denoising low-rank basis nearly matches the strongest simple supervised baseline
  - shared vs task-specific gap is tiny at `rank=16`, suggesting the basis itself is reusable
  - but SSL does **not** beat the raw supervised `raw+rank LR` baseline

### Science

- **Best simple baseline**: `token_only LR`
  - `AUC-AUROC = 0.8356`
- **Best shared SSL family**: `token_only ↔ token_plus_traj`, `rrr`, `rank 16`
  - `AUC-AUROC = 0.8240`
- **Same-problem ablation**:
  - `adjacent_full / cca / same_run / rank 4 = 0.7478`
  - `adjacent_full / cca / same_problem / rank 4 = 0.8211`
- **Interpretation**:
  - same-problem positives can partially “recover” science AUC, but still fail to beat the simplest supervised baselines
  - this makes same-problem positives look tempting while remaining protocol-risky and unstable

### Coding (cheap confirmation only)

- See:
  - `results/tables/earlystop_ssl/coding_ssl_basis_summary.csv`
  - `results/scans/earlystop_ssl/coding_ssl_basis_detail.json`
- Best cheap coding numbers stayed near random:
  - `rank_only LR = 0.4958`
  - `adjacent_cca same_problem rank16 = 0.4944`
  - `raw+rank LR = 0.4747`
- **Interpretation**: coding remains a fallback / confirmation lane, not a promising SSL target in this feature space.

## Label-Efficiency / Semi-Supervised Results

See:

- `results/tables/earlystop_ssl_label_efficiency.csv`
- `results/tables/earlystop_ssl_label_efficiency_summary.csv`
- `results/scans/earlystop_ssl/ssl_semisup_detail.json`

### Math

- **5% labels**
  - `raw+rank LR = 0.9065 ± 0.0465`
  - `frozen SVD = 0.8828 ± 0.0589`
  - best SSL variant (`task denoise rank16`) = `0.8811 ± 0.0605`
- **10% labels**
  - `raw+rank LR = 0.9366 ± 0.0132`
  - `frozen SVD = 0.9342 ± 0.0090`
  - best SSL variant (`task denoise rank16`) = `0.9305 ± 0.0135`
- **25% labels**
  - `raw+rank LR = 0.9570 ± 0.0004`
  - `task denoise rank16 = 0.9562 ± 0.0005`
  - `frozen SVD = 0.9560 ± 0.0004`
- **Conclusion**:
  - math does **not** benefit from the new SSL / semisup routes
  - even when pseudo-label precision is high, the simple supervised baseline stays strongest

### Science

- **5% labels**
  - `raw+rank LR = 0.7895 ± 0.0092`
  - `frozen tokenpair_rrr rank16 = 0.8090 ± 0.0151`
  - `agreement-filtered denoise rank16 = 0.8128 ± 0.0094`
- **10% labels**
  - `raw+rank LR = 0.7985 ± 0.0098`
  - `frozen tokenpair_rrr rank16 = 0.8136 ± 0.0129`
  - `agreement-filtered tokenpair_rrr rank16 = 0.8135 ± 0.0114`
- **25% labels**
  - `raw+rank LR = 0.8258 ± 0.0029`
  - `frozen tokenpair_rrr rank16 = 0.8290 ± 0.0073`
  - `agreement-filtered tokenpair_rrr rank16 = 0.8296 ± 0.0078`
- **50% labels**
  - `raw+rank LR = 0.8378 ± 0.0051`
  - best SSL/semisup remains below (`tokenpair pseudo rank16 = 0.8270 ± 0.0021`)
- **100% labels**
  - `raw+rank LR = 0.8349`
  - `frozen tokenpair_rrr rank16 = 0.8240`
- **Conclusion**:
  - science is the only domain where SSL / semisup shows a reproducible low-label gain
  - the gain is concentrated at `5% / 10% / 25%`
  - once labels are plentiful (`50% / 100%`), simple supervised baselines retake the lead

## Pseudo-Label Threshold Sweep

See:

- `results/tables/earlystop_ssl/ssl_semisup_threshold_sweep_summary.csv`
- `results/scans/earlystop_ssl/ssl_semisup_threshold_sweep_detail.json`

Threshold sweep was run on `tokenpair_rrr @ rank 16` with thresholds `0.90` and `0.95`.

### Math

- At `10%` labels:
  - pseudo `0.90`: `AUC=0.8747`, `n_pseudo≈2394`, `precision≈0.8700`
  - pseudo `0.95`: `AUC=0.8775`, `n_pseudo≈1866`, `precision≈0.9035`
- **Takeaway**: stricter `0.95` helps math by improving pseudo quality enough to offset the smaller pool.

### Science

- At `5%` labels:
  - agreement `0.90`: `AUC=0.8107`, `precision≈0.7787`
  - agreement `0.95`: `AUC=0.8113`, `precision≈0.8159`
- At `10%` labels:
  - pseudo `0.90`: `AUC=0.8109`, `precision≈0.7832`
  - pseudo `0.95`: `AUC=0.8132`, `precision≈0.8580`
- At `25%` labels:
  - agreement `0.90`: `AUC=0.8311`, `precision≈0.8097`
  - agreement `0.95`: `AUC=0.8296`, `precision≈0.8595`
- **Takeaway**:
  - `0.95` is safer and usually better in the very low-label science regime
  - `0.90` can slightly win once labels are less scarce and the teacher is already stronger

## Main Findings

1. **Science-only low-label win exists**  
   Shared SSL basis + light head is genuinely useful for `science` at `5% / 10% / 25%` labels, especially with `token_only ↔ token_plus_traj` and `denoise_full`.

2. **Math prefers simple supervised signals**  
   Math does not reward the new SSL / semisup routes. Even strong pseudo-label precision does not beat plain `raw+rank LR`.

3. **Shared basis can be nearly reusable without being universally best**  
   On math, `denoise_full rank16` shared basis (`0.9580`) is nearly tied with task-specific (`0.9584`), which supports the “reusable basis + cheap head” idea even though it does not exceed the best supervised baseline.

## Important Failure Modes

1. **Raw↔rank alone is not a good SSL target**  
   `raw_rank_cca` and related local-view variants underperform badly on both math and science.

2. **Same-problem positives are inconsistent and risky**  
   They hurt math, and while they improve one science ablation, they still do not beat the simplest supervised baselines. This looks more like identity smoothing than a robust general signal.

3. **Coding mismatch persists**  
   All cheap coding confirmations stayed around chance, so this SSL route should not absorb more compute right now.

## What Looks Worth Continuing

- **Continue**: `science`-focused, shared-basis label-efficiency work
  - primary candidates:
    - `token_only ↔ token_plus_traj` with `rrr`, `rank 16`
    - `denoise_full`, `rank 16`
  - semisup recipe:
    - frozen shared basis
    - light LR head
    - high-confidence pseudo-labels
    - agreement filtering
    - threshold near `0.95` for the smallest budgets

## What Looks Not Worth Continuing

- **Stop**: pushing `raw ↔ rank` CCA as the main SSL direction
- **Stop**: spending major compute on coding in this feature family
- **Stop**: treating same-problem positives as a default positive definition

## Incomplete / Deferred

These were not completed in this round:

- separate `10↔40`, `40↔70`, `70↔100` pairwise reports as independent final tables
- fresh reruns of `centered_raw` / `centered_raw+rank` under the new SSL harness
- unified noncoding semisup vs math-only vs science-only direct comparison in one table

The current evidence is already sufficient for a strong Round1 conclusion:

- **positive**: low-label science can benefit
- **negative**: math and coding do not justify broader SSL/semisup investment in this form

## Submission Artifacts

Two EarlyStop submission files were generated from the round1 outputs.

### Full shared-basis experimental hybrid

- `submission/EarlyStop/earlystop_ssl_round1_shared_ms_experimental.json`
- manifest: `submission/EarlyStop/earlystop_ssl_round1_shared_ms_experimental.manifest.json`
- model: `models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_shared_ms.pkl`

Domain mapping:

- `math`: `denoise_full`, shared basis, `rank=16`
- `science`: `tokenpair_rrr`, shared basis, `rank=16`
- `coding`: copied from `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`

### Recommended science-only patch

- `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.json`
- manifest: `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.manifest.json`
- model: `models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_science_tokenpair16.pkl`

This patch only overrides the science (`gpqa`) caches and leaves math/coding on the prior base submission. That is the recommended export from this round because only science showed positive protocol-safe evidence.

## Commands Actually Run

### Baseline recon / repro

See:

- `docs/EARLYSTOP_SSL_BASELINE_REPRO_20260416.md`

### Basis smoke

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/train_earlystop_ssl_basis.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --domains math science \
  --ssl-ranks 4 \
  --seeds 42 \
  --smoke \
  --out-summary-csv results/tables/earlystop_ssl/ssl_basis_summary.smoke.csv \
  --out-ablation-csv results/tables/earlystop_ssl/ssl_basis_ablation.smoke.csv \
  --out-detail-json results/scans/earlystop_ssl/ssl_basis_detail.smoke.json
```

### Full basis sweep

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/train_earlystop_ssl_basis.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --domains math science \
  --configs raw_rank_cca tokenpair_rrr adjacent_cca adjacent_rrr denoise_full \
  --ssl-ranks 4 8 16 \
  --seeds 42 43 44 \
  --out-summary-csv results/tables/earlystop_ssl_summary.csv \
  --out-ablation-csv results/tables/earlystop_ssl_ablation.csv \
  --out-detail-json results/scans/earlystop_ssl/ssl_basis_detail.json
```

### Coding cheap confirmation

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/train_earlystop_ssl_basis.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --domains coding \
  --configs adjacent_cca tokenpair_rrr denoise_full \
  --ssl-ranks 16 \
  --seeds 42 \
  --out-summary-csv results/tables/earlystop_ssl/coding_ssl_basis_summary.csv \
  --out-ablation-csv results/tables/earlystop_ssl/coding_ssl_basis_ablation.csv \
  --out-detail-json results/scans/earlystop_ssl/coding_ssl_basis_detail.json
```

### Full label-efficiency / semisup sweep

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/train_earlystop_semisup.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --supervised-bundle models/ml_selectors/es_svd_ms_rr_r1.pkl \
  --domains math science \
  --configs adjacent_cca tokenpair_rrr denoise_full \
  --ssl-ranks 4 16 \
  --label-fractions 0.05 0.10 0.25 0.50 1.0 \
  --seeds 42 43 44 \
  --pseudo-thresholds 0.95 \
  --out-csv results/tables/earlystop_ssl_label_efficiency.csv \
  --out-summary-csv results/tables/earlystop_ssl_label_efficiency_summary.csv \
  --out-detail-json results/scans/earlystop_ssl/ssl_semisup_detail.json
```

### Threshold sweep

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/train_earlystop_semisup.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --supervised-bundle models/ml_selectors/es_svd_ms_rr_r1.pkl \
  --domains math science \
  --configs tokenpair_rrr \
  --ssl-ranks 16 \
  --label-fractions 0.05 0.10 0.25 \
  --seeds 42 43 44 \
  --pseudo-thresholds 0.90 0.95 \
  --out-csv results/tables/earlystop_ssl/ssl_semisup_threshold_sweep.csv \
  --out-summary-csv results/tables/earlystop_ssl/ssl_semisup_threshold_sweep_summary.csv \
  --out-detail-json results/scans/earlystop_ssl/ssl_semisup_threshold_sweep_detail.json
```

### Submission export

```bash
.venv/bin/python scripts/export_earlystop_ssl_submission.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --cache-root /home/jovyan/public-ro/MUI_HUB/cache_test \
  --blind-feature-store-pkl results/cache/export_earlystop_svd_submission_round1c_override/feature_store_all_ref030_3849580f4da8ce1c.pkl \
  --domain-spec math=denoise_full:16 science=tokenpair_rrr:16 \
  --base-submission submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json \
  --method-name earlystop_ssl_round1_shared_ms_experimental \
  --filename earlystop_ssl_round1_shared_ms_experimental.json \
  --out-dir submission/EarlyStop \
  --out-model models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_shared_ms.pkl
```

```bash
.venv/bin/python scripts/export_earlystop_ssl_submission.py \
  --prebuilt-cache-dir results/cache/es_svd_ms_rr_r1 \
  --cache-root /home/jovyan/public-ro/MUI_HUB/cache_test \
  --blind-feature-store-pkl results/cache/export_earlystop_svd_submission_round1c_override/feature_store_all_ref030_3849580f4da8ce1c.pkl \
  --domain-spec science=tokenpair_rrr:16 \
  --base-submission submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json \
  --method-name earlystop_ssl_round1_science_tokenpair16_patch \
  --filename earlystop_ssl_round1_science_tokenpair16_patch.json \
  --out-dir submission/EarlyStop \
  --out-model models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_science_tokenpair16.pkl
```
