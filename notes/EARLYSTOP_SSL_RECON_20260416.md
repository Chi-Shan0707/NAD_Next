# EarlyStop SSL / Semisup Recon Memo (2026-04-16)

## Repo root and scope

- Resolved repo root: `/home/jovyan/work/NAD_Next`
- Reason: it is the first matching root that simultaneously contains `README.md`, `docs/README.md`, and `scripts/README.md`.
- Main code/doc threads read before experimentation:
  - `README.md`
  - `docs/README.md`
  - `scripts/README.md`
  - `models/README.md`
  - `submission/README.md`
  - `docs/EARLYSTOP_SVD_IMPLEMENTATION_20260408.md`
  - `docs/EARLYSTOP_PREFIX10_SVD_ROUND1_PLAN_20260408.md`
  - `docs/EARLYSTOP_STRONG_FEATURES_ROUND1_20260409.md`
  - `docs/EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_RESULTS_20260409.md`
  - `docs/11_CROSS_ANCHOR_TRANSFER.md`
  - `docs/ES_SVD_MS_RR_R1.md`
  - `docs/ES_SVD_MS_RR_R2_REPORT.md`
  - `docs/ES_SVD_CODING_RR_R1.md`
  - `docs/19_SEMISUP_SVDOMAIN_CONCLUSIONS.md`
  - `docs/SEMISUP_SVDOMAIN.md`
  - `docs/DOMAIN_SPECIFIC_SSL.md`
  - `nad/ops/earlystop_svd.py`
  - `scripts/train_earlystop_svd.py`
  - `scripts/run_earlystop_prefix10_svd_round1.py`
  - `scripts/run_earlystop_prefix10_svd_round1b.py`
  - `scripts/run_earlystop_strongfeat_round1.py`
  - `scripts/run_earlystop_svd_problem_centered_round1.py`
  - `scripts/semi_supervised/train_semisup_svdomain.py`
  - `scripts/semi_supervised/train_domain_specific_ssl.py`
  - `SVDomain/train_es_svd_ms_rr_r1.py`
  - `SVDomain/experiments/run_cross_anchor_transfer.py`

## 1. 当前 earlystop 的最强 supervised 线是什么？

需要分三层说，避免把不同协议混成一个结论：

- **当前主线上线 / exported 非 coding 主线**：
  - `es_svd_ms_rr_r1`
  - 文档：`docs/ES_SVD_MS_RR_R1.md`
  - offline combined noncoding holdout:
    - `AUC of AUROC = 92.26%`
    - `AUC of SelAcc = 99.52%`
    - `Earliest >0.6 = 10%`
    - `AUROC@100% = 95.05%`
    - `Stop Acc@100% = 72.78%`
  - online note in the doc: `es_svd_ms_rr_r1__coding_from_round1c` is marked `CURRENT BEST`.

- **当前 canonical full-cache offline 非 coding SVD 更强版本**：
  - `es_svd_ms_rr_r2`
  - 文档：`docs/ES_SVD_MS_RR_R2_REPORT.md`
  - same grouped `85/15` protocol, but `10` anchors instead of `4`
  - combined noncoding holdout:
    - `AUC of AUROC = 93.66%`
    - `AUC of SelAcc = 99.51%`
    - `AUROC@100% = 94.88%`
    - `Stop Acc@100% = 74.26%`
  - Interpretation: `r2` is the stronger **offline AUROC-oriented canonical SVD** baseline; `r1` remains the cleaner anchor-transfer reference and current exported backbone.

- **当前 small-slice strongest noncoding research winner**：
  - `strongfeat_noncoding_anchor4_coding_v1_ref020`
  - 文档：`docs/EARLYSTOP_STRONG_FEATURES_ROUND1_20260409.md`
  - protocol is different: `cache + cache_train`, grouped `80/20` holdout, `max_problems_per_cache=8`
  - holdout:
    - `AUC of AUROC = 94.84%`
    - `AUC of SelAcc = 95.73%`
    - `Earliest >0.6 = 10%`
    - `AUROC@100% = 95.39%`
    - `Stop Acc@100% = 83.33%`
  - This is the strongest **cap8 / 80-20 strong-feature research winner**, but not the same protocol as canonical SVDomain.

**Working decision for new SSL/semisup study**:
- Treat `es_svd_ms_rr_r1` + `es_svd_ms_rr_r2` + `no_svd_lr` + `frozen_basis` as the canonical baseline family.
- Treat `strongfeat_noncoding_anchor4_coding_v1_ref020` as a useful strong-feature side reference, not the default selection protocol for the new study.

## 2. 当前 SVD 线有哪些已知优点和已知失败？

### 已知优点

- **math 上 cross-anchor frozen basis 很强**：
  - `docs/11_CROSS_ANCHOR_TRANSFER.md`
  - diagonal mean: frozen `94.02%` vs task-specific `94.13%`
  - off-diagonal focus mean: frozen `95.56%` vs task-specific `95.89%`
  - many math anchor pairs are effectively tie/win.

- **science 上 late-anchor frozen basis 仍可迁移**：
  - `100 -> 40`: only `-0.07 pts`
  - `100 -> 70`: only `-0.09 pts`

- **shared low-rank basis is reusable enough to justify frozen-basis tests**：
  - This is already supported by `cross_anchor_transfer_{matrix,deltas,summary}.csv`.

- **r2 improves canonical offline AUROC**：
  - `docs/ES_SVD_MS_RR_R2_REPORT.md`
  - stronger than `r1` on combined noncoding `AUC of AUROC`.

- **SVD remains interpretable and route-level analyzable**：
  - repository has `svd_explain.py`, `svd_introspection.py`, failure-mode docs, and exact route bundles.

### 已知失败 / negative results

- **coding is still near-random under token/trajectory feature space**：
  - `docs/ES_SVD_CODING_RR_R1.md`
  - holdout `AUC of AUROC = 43.42%`, `AUC of SelAcc = 24.00%`
  - coding branch does not justify major compute investment right now.

- **shared SSL basis from masked reconstruction failed badly**：
  - `docs/19_SEMISUP_SVDOMAIN_CONCLUSIONS.md`
  - `no_svd_lr` dominates nearly everywhere
  - `semisup r=8` collapses to exact `0.5000`.

- **same broad linear SSL basis is not better than supervised/frozen baselines**：
  - `docs/SEMISUP_SVDOMAIN.md`
  - particularly on math/science full-label regimes.

- **even same-run contrastive/domain-specific SSL is not a free win**：
  - `docs/DOMAIN_SPECIFIC_SSL.md`
  - improves over old shared SSL, but still often trails `no_svd_lr` or `frozen_svd`.

- **problem-centered representations did not beat the strong-feature raw+rank winner**：
  - `docs/EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_RESULTS_20260409.md`
  - best representation there was still `raw+rank`, not centered alternatives.

## 3. 现有 protocol 的 split / holdout / metrics 是什么？

Again there are two relevant protocols; new study should default to the canonical one.

### Canonical SVD / SVDomain / semisup protocol

- Training roots:
  - main: `MUI_HUB/cache`
  - extra: `MUI_HUB/cache_train`
- Holdout:
  - grouped `85/15`
  - unit: `dataset + problem_id`
  - seed: `42`
  - same problem is kept on the same side across roots.
- Internal CV / route search:
  - `GroupKFold`
  - grouped by problem-group keys, with cross-root-safe grouping in the SVDomain scripts.
- Anchors:
  - `r1`: `10 / 40 / 70 / 100`
  - `r2`: `10 / 20 / ... / 100`
- Core metrics:
  - `AUC of AUROC`
  - `AUC of SelAcc`
  - `Earliest > 0.6`
  - `AUROC@100%`
  - `Stop Acc@100%`
- Additional for bridge/export diagnostics:
  - `Hit@1`
  - `Hit@3`
  - `SelAcc@10`
  - `Pairwise`

### Strong-feature / problem-centered research protocol

- Training roots: same `cache + cache_train`
- Holdout:
  - grouped `80/20`
  - unit: `dataset + problem_id`
- Often capped:
  - `max_problems_per_cache = 8`
- This is useful as a cheap exploratory lane, but it is not the cleanest canonical protocol for the new SSL study.

## 4. 哪些特征必须视为 prefix-safe，哪些可能带后验泄漏？

### Must be treated as prefix-safe

These are explicitly constructed from prefix-bounded arrays in `nad/ops/earlystop_svd.py` or `scripts/run_earlystop_prefix10_svd_round1.py`:

- Token prefix / recency statistics:
  - `tok_conf_prefix`
  - `tok_conf_recency`
  - `tok_gini_prefix`
  - `tok_gini_tail`
  - `tok_gini_slope`
  - `tok_neg_entropy_prefix`
  - `tok_neg_entropy_recency`
  - `tok_selfcert_prefix`
  - `tok_selfcert_recency`
  - `tok_logprob_prefix`
  - `tok_logprob_recency`
- Prefix trajectory stats computed from prefix slices:
  - `traj_continuity`
  - `traj_reflection_count`
  - `traj_novelty`
  - `traj_max_reflection`
  - `traj_late_convergence`
- Prefix-local stats:
  - `tail_q10`
  - `head_tail_gap`
  - `tail_variance`
  - `last_event_tail_conf`
  - `event_pre_post_delta`
- Prefix-safe meta stats kept in prefix runners:
  - `nc_mean`
  - `nc_slope`
- Raw/rank transforms at a given anchor
  - `raw`
  - `rank`
  - `raw+rank`

### Must be treated as leakage-prone or protocol-risky

- `self_similarity`
  - explicitly called out in round1b docs as leaking later information into prefix view.
- Any feature that directly uses full-sequence / future-anchor information inside an early-prefix supervised head.
- Any split that lets the same `dataset + problem_id` appear on both train and holdout across `cache` and `cache_train`.
- Slot100 / bridge / leaderboard feedback
  - valid for secondary export analysis, not valid for model selection in the new study.
- Same-problem positives across different runs
  - not automatically leakage, but protocol-risky because they can flatten the correctness boundary and can accidentally smuggle problem identity rather than stable within-run structure.
  - should be ablation-only, not default positive construction.

## 5. 哪些 domain 应优先，哪些 domain 只做廉价验证？

### Priority order

1. **math**
   - best evidence for reusable frozen/shared low-rank basis
   - strongest cross-anchor transfer stability
   - likely best place to test label-efficiency gains cleanly

2. **science**
   - more challenging but still promising
   - especially useful for testing whether late-anchor shared basis helps more than early-anchor basis
   - important negative-control domain for pseudo-label drift and anchor asymmetry

3. **coding**
   - cheap fallback / confirmation only
   - existing evidence says current feature family is fundamentally weak here
   - do only small confirmation sweeps unless a new noncoding result suggests a clearly transferable gain

## Decision for the new study

- Default protocol: grouped `85/15`, seed `42`, `dataset + problem_id`, no leaderboard-driven selection.
- Default domains: `math`, `science`.
- Default representations to compare first:
  - `raw`
  - `rank`
  - `raw+rank`
  - `token_only`
  - `token_plus_traj`
- Default label budgets:
  - `5 / 10 / 25 / 50 / 100%`
- Default view positives:
  - same-run multi-view only
- same-problem different-run positives:
  - ablation only
