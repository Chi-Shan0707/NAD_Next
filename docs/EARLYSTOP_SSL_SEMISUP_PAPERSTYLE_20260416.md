# EarlyStop with Shared Low-Rank SSL Basis and Lightweight Heads

**Round1 paper-style report — 2026-04-16**

## Abstract

We study a CPU-first alternative to task-specific EarlyStop training in `NAD_Next`: first learn a shared low-rank basis from prefix-safe multi-view structure, then fit a lightweight discriminative head with limited labels. The goal is not to “bolt on SSL”, but to test whether the existing EarlyStop / SVDomain trajectory admits a useful decomposition into:

1. **shared structure learning** from same-run multi-view agreement, and
2. **small-head supervision** for correctness ranking.

Under the canonical grouped `85/15` holdout protocol, the main result is asymmetric:

- **science** shows a real low-label opportunity;
- **math** does not reward the new SSL route over simple supervised baselines;
- **coding** remains effectively random in this feature family.

The strongest positive result is not a universal leaderboard gain, but a **domain- and budget-specific finding**: for science, shared SSL bases can outperform simple supervised baselines in the `5% / 10% / 25%` label regime. The strongest negative result is equally clear: **raw↔rank-only SSL and coding-focused transfer are not worth further investment in the current feature space.**

## Contributions

This round makes four concrete contributions to the repository’s EarlyStop line.

1. It adds a **CPU-first reusable SSL basis toolkit** for grouped-protocol EarlyStop experiments.
2. It shows that **shared low-rank bases are structurally reusable** even when they are not universally stronger than supervised baselines.
3. It identifies a **narrow but real low-label science regime** where SSL/semisupervision helps.
4. It records several **strong negative results** clearly enough to guide future compute allocation.

## 1. Research Questions

This round was organized around five concrete questions.

### Q1. Low-label benefit

Can same-run self-supervised multiview training improve EarlyStop non-coding performance in low-label regimes?

### Q2. Shared-basis reuse

Can a frozen shared basis remain competitive with task-specific fitting, especially across anchors?

### Q3. View design

Which views are most useful?

- `raw ↔ rank`
- `token_only ↔ token_plus_traj`
- adjacent anchors (`10↔40`, `40↔70`, `70↔100`) as a shared anchor-consistency signal
- masked / denoising low-rank reconstruction

### Q4. Semi-supervision

Does high-confidence pseudo-labeling help, and if so, in which domains and under which thresholds?

### Q5. Failure characterization

Which routes fail for principled reasons rather than mere under-tuning?

## 2. Protocol

### Data and Split

- **Repo root**: `/home/jovyan/work/NAD_Next`
- **Labeled training roots**: canonical cached feature store under `results/cache/es_svd_ms_rr_r1`
- **Evaluation split**: grouped `85/15` holdout by `dataset + problem_id`
- **Split seed**: `42`
- **Domains prioritized**: `math`, `science`
- **Coding policy**: cheap confirmation only

### Metrics

All claims use the repository’s existing EarlyStop metric family:

- `AUC of AUROC`
- `AUC of SelAcc`
- `Earliest > 0.6`
- `AUROC@100%`
- `Stop Acc@100%`

### Prefix Safety

Only prefix-safe fixed features were used:

- token confidence / gini / entropy / self-cert / logprob prefix features
- trajectory features
- availability flags
- raw / within-problem rank transforms at the same anchor

No future-anchor or known leakage-prone feature was mixed into early-prefix training.

## 3. Methods

### 3.1 Representation family

The study stayed within the existing `token_plus_traj_fixed` backbone rather than introducing new heavy features.

### 3.2 SSL basis families

Three linear / near-linear basis families were implemented:

1. **ridge CCA**
2. **cross-covariance SVD / reduced-rank regression**
3. **denoising SVD**

All methods are CPU-first and sklearn / numpy based.

### 3.3 Heads

Each basis was evaluated with a lightweight logistic head:

- **shared / frozen basis + head**
- **task-specific basis + head**
- **no-basis baseline**

### 3.4 Semi-supervised setting

When no truly unlabeled pool is protocol-safe, we use **hidden-label simulation**:

- label budgets: `5 / 10 / 25 / 50 / 100%`
- remaining training problems are treated as unlabeled
- pseudo-labels come from a high-confidence teacher
- optional agreement filtering uses a masked perturbation view

## 4. Baselines

The study was anchored to the already-reproduced baselines:

- canonical `es_svd_ms_rr_r1`
- simple `tok_conf`-style reference
- no-basis `raw+rank LR`
- cross-anchor `frozen / task-specific / no_svd` comparisons

See:

- `docs/EARLYSTOP_SSL_BASELINE_REPRO_20260416.md`
- `results/tables/earlystop_ssl/baseline_summary.csv`

## 5. Main Results

## 5.1 Basis learning: math

From `results/tables/earlystop_ssl_summary.csv`:

| Domain | Model | AUC-AUROC | AUC-SelAcc | AUROC@100 |
|---|---|---:|---:|---:|
| math | `raw+rank LR` | `0.9593` | `0.9973` | `0.9837` |
| math | `denoise_full shared rank16` | `0.9580` | `0.9973` | `0.9811` |
| math | `denoise_full task rank16` | `0.9584` | `0.9976` | `0.9818` |
| math | `token_pair rrr shared rank16` | `0.9378` | `0.9978` | `0.9722` |

### Interpretation

- The best SSL family on math is **denoising low-rank reconstruction**.
- The **shared-vs-task-specific gap is negligible** at rank `16`.
- But the strongest simple supervised baseline still wins by a small margin.

This is a useful structural result: shared basis reuse looks plausible, but **not yet necessary** when labels are abundant.

## 5.2 Basis learning: science

| Domain | Model | AUC-AUROC | AUC-SelAcc | AUROC@100 |
|---|---|---:|---:|---:|
| science | `token_only LR` | `0.8356` | `0.9682` | `0.8676` |
| science | `raw+rank LR` | `0.8349` | `0.9846` | `0.8767` |
| science | `token_pair rrr shared rank16` | `0.8240` | `0.9914` | `0.8504` |
| science | `token_pair rrr task rank16` | `0.8236` | `0.9859` | `0.8600` |
| science | `adjacent_cca same_problem rank4` | `0.8211` | `0.9845` | `0.8468` |

### Interpretation

- The strongest full-label SSL family on science is **`token_only ↔ token_plus_traj` with reduced-rank coupling**.
- It still trails the simplest supervised full-label baselines.
- However, it becomes useful in low-label semisupervised settings.

## 5.3 Label-efficiency: math

From `results/tables/earlystop_ssl_label_efficiency_summary.csv`:

| Label Budget | Best Simple Supervised | Best SSL/Semisup | Main Reading |
|---|---:|---:|---|
| `5%` | `0.9065 ± 0.0465` (`raw+rank LR`) | `0.8811 ± 0.0605` (`task denoise@16`) | simple supervised wins |
| `10%` | `0.9366 ± 0.0132` (`raw+rank LR`) | `0.9305 ± 0.0135` (`task denoise@16`) | simple supervised still wins |
| `25%` | `0.9570 ± 0.0004` (`raw+rank LR`) | `0.9562 ± 0.0005` (`task denoise@16`) | effectively tied, but still no win |
| `50%` | `0.9586 ± 0.0008` (`raw+rank LR`) | `0.9582 ± 0.0012` (`task denoise@16`) | no practical gain |
| `100%` | `0.9593` (`raw+rank LR`) | `0.9584` (`task denoise@16`) | no gain |

### Interpretation

Math does **not** provide a positive case for continuing this SSL route. The best story here is not “better accuracy”, but “shared basis can stay close without much degradation”.

## 5.4 Label-efficiency: science

| Label Budget | Best Simple Supervised | Strongest SSL/Semisup | AUC-AUROC |
|---|---:|---|---:|
| `5%` | `0.7895 ± 0.0092` (`raw+rank LR`) | `agreement denoise shared@16, thr=0.95` | `0.8128 ± 0.0094` |
| `10%` | `0.7985 ± 0.0098` (`raw+rank LR`) | `frozen token_pair rrr shared@16` | `0.8136 ± 0.0129` |
| `25%` | `0.8258 ± 0.0029` (`raw+rank LR`) | `agreement token_pair rrr shared@16, thr=0.95` | `0.8296 ± 0.0078` |
| `50%` | `0.8378 ± 0.0051` (`raw+rank LR`) | best SSL below baseline | no gain |
| `100%` | `0.8349` (`raw+rank LR`) | best SSL below baseline | no gain |

### Interpretation

This is the clearest positive result of the round:

- SSL/semisup is **not broadly useful**
- but it **is** useful for **science under small label budgets**

The signal is strongest when:

1. the basis is shared rather than aggressively task-specific,
2. the views couple `token_only` and `token_plus_traj`, or use denoising,
3. pseudo-labels are kept high-confidence.

## 6. Pseudo-Label Quality and Thresholds

From `results/tables/earlystop_ssl/ssl_semisup_threshold_sweep_summary.csv`:

### Science

- `5%` labels:
  - agreement, `thr=0.90`: `AUC=0.8107`, pseudo precision `0.7787`
  - agreement, `thr=0.95`: `AUC=0.8113`, pseudo precision `0.8159`
- `10%` labels:
  - pseudo, `thr=0.90`: `AUC=0.8109`, precision `0.7832`
  - pseudo, `thr=0.95`: `AUC=0.8132`, precision `0.8580`
- `25%` labels:
  - agreement, `thr=0.90`: `AUC=0.8311`, precision `0.8097`
  - agreement, `thr=0.95`: `AUC=0.8296`, precision `0.8595`

### Reading

- In the smallest budgets, **`0.95` is the safer threshold**.
- Once the teacher is stronger, `0.90` can sometimes recover a small gain by adding more pseudo-data.
- The pseudo-label story is therefore **teacher-quality dependent**, not a universal monotonic function of more pseudo-data.

## 7. Ablations That Matter

### 7.1 Same-run vs same-problem positives

The default protocol used same-run positives only.

Same-problem positives were tested only as ablation and showed:

- **math**: usually harmful
- **science**: can lift one adjacent-anchor CCA ablation from `0.7478` to `0.8211`

This is exactly the kind of seductive ablation that can mislead future work:

- it may look strong,
- it is not consistently strong,
- and it risks learning problem identity / template similarity rather than robust correctness structure.

### 7.2 Raw↔rank-only SSL

`raw_rank_cca` underperformed on both math and science and never emerged as a practical candidate. This route should be treated as **investigated and deprioritized**.

## 8. Negative Results Worth Keeping

The round produced several strong negative results that are worth preserving:

1. **No broad non-coding win**  
   SSL does not dominate existing supervised EarlyStop lines.

2. **Math does not need this SSL route**  
   Even when shared bases are decent, the simple supervised baseline remains as good or better.

3. **Coding remains a dead end here**  
   Cheap confirmation stayed at chance:
   - `rank_only LR = 0.4958`
   - `adjacent_cca same_problem@16 = 0.4944`
   - `raw+rank LR = 0.4747`

These are not weak failures; they are strong enough to guide resource allocation.

## 9. What Is Actually Worth Continuing

The best continuation is narrow and specific:

### Continue

- **Domain**: science
- **Settings**: `5% / 10% / 25%` labels
- **Basis candidates**:
  - `token_only ↔ token_plus_traj` + `rrr` + `rank16`
  - `denoise_full` + `rank16`
- **Semisup recipe**:
  - frozen shared basis
  - logistic head
  - high-confidence pseudo-labels
  - agreement filtering
  - `0.95` threshold as the default low-budget starting point

### Stop

- raw↔rank-only SSL as the main route
- coding-focused continuation in this feature space
- same-problem positives as a default SSL positive definition

## 10. Submission Artifacts

Two experimental EarlyStop submission files were generated from this round.

### 10.1 Full shared-basis hybrid

- `submission/EarlyStop/earlystop_ssl_round1_shared_ms_experimental.json`
- model: `models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_shared_ms.pkl`
- manifest: `submission/EarlyStop/earlystop_ssl_round1_shared_ms_experimental.manifest.json`

This export uses:

- **math**: `denoise_full shared rank16`
- **science**: `token_pair rrr shared rank16`
- **coding**: copied from `submission/EarlyStop/es_svd_ms_rr_r1__coding_from_round1c.json`

This file is the most faithful “shared low-rank basis + lightweight head” export from the round.

### 10.2 Conservative science-only patch

- `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.json`
- model: `models/ml_selectors/earlystop_ssl/earlystop_ssl_round1_science_tokenpair16.pkl`
- manifest: `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.manifest.json`

This export replaces only the two `gpqa` blind caches and leaves all math / coding caches from the base submission untouched.

### Recommendation

If an experimental submission must be chosen from this round, the **science-only patch** is the more defensible one, because:

1. the only positive evidence from the round is science-specific,
2. math did not beat its strongest supervised baseline,
3. coding was intentionally left untouched.

So the recommended experimental file is:

- `submission/EarlyStop/earlystop_ssl_round1_science_tokenpair16_patch.json`

## 11. Threats to Validity

Three caveats matter for interpreting the results correctly.

### 11.1 Full-label science does not improve

The strongest science findings are **label-efficiency** findings, not full-label replacement results. The best full-label science supervised baseline remains stronger than the tested SSL families.

### 11.2 Same-problem positives can overstate progress

The science adjacent-anchor ablation with same-problem positives looks deceptively attractive, but it is not stable enough to justify adopting it as the default SSL relation. It is better interpreted as an analysis probe than as a production recipe.

### 11.3 Coding conclusions are feature-space specific

The negative coding result applies to the current prefix-safe token/trajectory feature family and lightweight linear heads. It does not prove that coding EarlyStop is hopeless in general; it only shows that this route is a poor investment within the present constraints.

## 12. Reproducibility

All main scripts and outputs are checked into the workspace:

- basis study: `scripts/train_earlystop_ssl_basis.py`
- semisup study: `scripts/train_earlystop_semisup.py`
- export: `scripts/export_earlystop_ssl_submission.py`

Main output tables:

- `results/tables/earlystop_ssl_summary.csv`
- `results/tables/earlystop_ssl_ablation.csv`
- `results/tables/earlystop_ssl_label_efficiency.csv`
- `results/tables/earlystop_ssl_label_efficiency_summary.csv`
- `results/tables/earlystop_ssl/ssl_semisup_threshold_sweep_summary.csv`

## 13. Bottom Line

This round does **not** justify replacing the repository’s main EarlyStop line with a new SSL backbone.

It **does** justify a narrower conclusion:

> Shared low-rank basis learning is a viable reusable abstraction, but its practical value is concentrated in **science under low-label regimes**, not in math, and not in coding.

That is already a worthwhile research result:

- it validates one promising direction,
- sharply rejects two misleading ones,
- and provides a clean experimental basis for a smaller, better-targeted Round2.
