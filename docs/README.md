# Docs Layout

## Active analysis docs
- `HYPOTHESIS_ANALYSIS_20260405.md`: top-level hypothesis and feature-family readout.
- `EXTREME12_TEST_ANALYSIS.md`: frozen generic/math-side proxy baseline.
- `EXTREME12_V2_EXPERIMENT.md`: extreme12 follow-up experiment notes.
- `EXTREME_SELACC_GRID_20260406.md`: exhaustive `SelAcc@k%` grid for the extreme line.

## SVD line
- `ES_SVD_MS_RR_R1.md`: canonical `r1` multi-domain EarlyStop SVD report.
- `ES_SVD_MS_RR_R2_REPORT.md`: 10-anchor `r2` EarlyStop SVD training report.
- `ES_SVD_CODING_RR_R1.md`: coding-only EarlyStop SVD report and negative-result context.
- `ES_SVD_MATH_RL_CHECKPOINT_RANKING.md`: RL checkpoint ranking export based on SVD models.
- `SVD_PERF_PLAN_20260411.md`: coding `slot100` SVDomain plan.
- `SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`: coding `slot100` SVDomain implementation and smoke results.
- `SVD_INTERPRETABILITY_R1_20260411.md`: explainability/viewer integration notes.
- `SVD_INTROSPECTION_RESULTS_es_svd_ms_rr_r1.md`: exact weight/back-projection introspection summary for `es_svd_ms_rr_r1`.
- `SVD_FAILURE_MODES_es_svd_ms_rr_r1.md`: wrong-top1/failure-mode analysis summary for `es_svd_ms_rr_r1`.
- `11_CROSS_ANCHOR_TRANSFER.md`: full `10/40/70/100` anchor-to-anchor transfer evidence for the shared low-rank basis.
- `12_CODING_DIAGNOSIS.md`: coding negative-result diagnosis separating feature mismatch, instability, and low-rank compactness.
- `16_DENSE_ANCHOR_EARLYSTOP.md`: `10/20/.../100` dense-anchor EarlyStop timing note for `math / science / coding`, including onset/plateau tables and neuron-vs-legacy comparison.
- `17_DENSE_CROSS_ANCHOR_TRANSFER.md`: all-to-all `10/20/.../100` dense cross-anchor transfer note for `math / science`, including gap-by-distance and source-anchor rankings.
- `18_SVD_FEATURE_COMPLEXITY.md`: three-layer feature-count / noise-control protocol for showing when SVD becomes more valuable than no-SVD.
- `BESTOFN_ES_SVD_MS_RR_R1_SLOT100_20260411.md`: current slot100 bridge result snapshot.

## Coding line
- `CODE_SELECTOR_VALIDATION_20260405.md`
- `CODE_BASELINE_V1_PHASE2_20260405.md`
- `CODE_V2_CANDIDATE_20260405.md`
- `CODE_V2_EXHAUSTIVE_20260406.md`
- `CODE_DEEPSETS_ROUND1_PLAN_20260408.md`
- `CODE_DEEPSETS_ROUND1_RESULTS_20260408.md`
- `CODE_RNS_ROUND1_RESULTS_20260409.md`

## Science / GPQA line
- `SCIENCE_BASELINE_V1_ROUND1_20260405.md`
- `GPQA_PAIRWISE_ROUND1_RESULTS_20260406.md`
- `GPQA_PAIRWISE_ROUND2_RESULTS_20260406.md`
- `SCIENCE_HYBRID_ROUND3_PLAN_20260406.md`
- `SCIENCE_HYBRID_ROUND3_RESULTS_20260406.md`
- `GPQA_GROUP_MODEL_PLAN_20260406.md`
- `GPQA_DEEPSETS_ROUND1_PLAN_20260407.md`
- `GPQA_DEEPSETS_ROUND1_RESULTS_20260407.md`

## Math line
- `MATH_DEEPSETS_ROUND1_PLAN_20260408.md`
- `MATH_DEEPSETS_ROUND1_RESULTS_20260408.md`
- `BESTOFN_SCORE_RECOVERY_20260408.md`
- `BESTOFN_MODEL_CONTINUATION_20260408.md`

## Status snapshots
- `GPQA_RANKSVM_MATH_SVM_STATUS_20260408.md`
- `WORK_SUMMARY_0408_06.md`
- `EARLYSTOP_PREFIX10_SVD_ROUND1B_CAP8_RESULTS_20260409.md`
- `EARLYSTOP_STRONG_FEATURES_ROUND1_20260409.md`
- `EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_PLAN_20260409.md`
- `EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_RESULTS_20260409.md`
- `SVD_PERF_PLAN_20260411.md`
- `SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`

## Handoffs
- `handoffs/2026-04-05/`: dated handoff sequence for the coding-selector work.

## Reference
- `reference/FEATURES.md`: repository feature inventory.
- `reference/LOCAL_CODEX_NOTES_20260403.md`: runtime and environment notes.

## Ideas and archive
- `ideas/IDEA.md`: exploratory design notes.
- `archive/raw_sessions/`: archived raw session text dumps.

## 当前分析文档
- `HYPOTHESIS_ANALYSIS_20260405.md`：顶层假设与特征家族梳理。
- `EXTREME12_TEST_ANALYSIS.md`：冻结的 generic/math 侧 proxy 基线。
- `EXTREME12_V2_EXPERIMENT.md`：extreme12 后续实验记录。
- `EXTREME_SELACC_GRID_20260406.md`：extreme 线的 `SelAcc@k%` 穷举表。

## SVD 主线
- `ES_SVD_MS_RR_R1.md`：canonical `r1` 多域 EarlyStop SVD 报告。
- `ES_SVD_MS_RR_R2_REPORT.md`：10-anchor `r2` EarlyStop SVD 训练报告。
- `ES_SVD_CODING_RR_R1.md`：coding-only EarlyStop SVD 结果与负结果语境。
- `ES_SVD_MATH_RL_CHECKPOINT_RANKING.md`：基于 SVD 模型导出的 RL checkpoint ranking 结果。
- `SVD_PERF_PLAN_20260411.md`：coding `slot100` SVDomain 实验计划。
- `SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`：coding `slot100` SVDomain 实现与 smoke 结果。
- `SVD_INTERPRETABILITY_R1_20260411.md`：解释性与 viewer 接入说明。
- `SVD_INTROSPECTION_RESULTS_es_svd_ms_rr_r1.md`：`es_svd_ms_rr_r1` 的精确权重回投与稳定性内省摘要。
- `SVD_FAILURE_MODES_es_svd_ms_rr_r1.md`：`es_svd_ms_rr_r1` 的 wrong-top1 / failure-mode 摘要。
- `11_CROSS_ANCHOR_TRANSFER.md`：共享 low-rank basis 在 `10/40/70/100` 全 trajectory 上的 anchor-to-anchor transfer 证据。
- `12_CODING_DIAGNOSIS.md`：coding 负结果诊断，区分 feature mismatch、评估不稳定性与低秩紧致性。
- `16_DENSE_ANCHOR_EARLYSTOP.md`：`math / science / coding` 的 `10/20/.../100` dense-anchor EarlyStop timing 结果，包含 onset / plateau 表与 neuron-vs-legacy 对照。
- `17_DENSE_CROSS_ANCHOR_TRANSFER.md`：`math / science` 在全 `10/20/.../100` trajectory 上的 all-to-all dense cross-anchor transfer 结果，包含 distance-decay 与 source-anchor ranking。
- `18_SVD_FEATURE_COMPLEXITY.md`：三层 feature-count / noise-control 实验协议，用来证明 SVD 在更宽、更脏的 feature bank 上何时开始明显优于 no-SVD。
- `BESTOFN_ES_SVD_MS_RR_R1_SLOT100_20260411.md`：当前 slot100 bridge 结果快照。

## Coding 主线
- `CODE_SELECTOR_VALIDATION_20260405.md`
- `CODE_BASELINE_V1_PHASE2_20260405.md`
- `CODE_V2_CANDIDATE_20260405.md`
- `CODE_V2_EXHAUSTIVE_20260406.md`
- `CODE_DEEPSETS_ROUND1_PLAN_20260408.md`
- `CODE_DEEPSETS_ROUND1_RESULTS_20260408.md`
- `CODE_RNS_ROUND1_RESULTS_20260409.md`

## Science / GPQA 主线
- `SCIENCE_BASELINE_V1_ROUND1_20260405.md`
- `GPQA_PAIRWISE_ROUND1_RESULTS_20260406.md`
- `GPQA_PAIRWISE_ROUND2_RESULTS_20260406.md`
- `SCIENCE_HYBRID_ROUND3_PLAN_20260406.md`
- `SCIENCE_HYBRID_ROUND3_RESULTS_20260406.md`
- `GPQA_GROUP_MODEL_PLAN_20260406.md`
- `GPQA_DEEPSETS_ROUND1_PLAN_20260407.md`
- `GPQA_DEEPSETS_ROUND1_RESULTS_20260407.md`

## Math 主线
- `MATH_DEEPSETS_ROUND1_PLAN_20260408.md`
- `MATH_DEEPSETS_ROUND1_RESULTS_20260408.md`
- `BESTOFN_SCORE_RECOVERY_20260408.md`
- `BESTOFN_MODEL_CONTINUATION_20260408.md`

## 状态快照
- `GPQA_RANKSVM_MATH_SVM_STATUS_20260408.md`
- `WORK_SUMMARY_0408_06.md`
- `EARLYSTOP_PREFIX10_SVD_ROUND1B_CAP8_RESULTS_20260409.md`
- `EARLYSTOP_STRONG_FEATURES_ROUND1_20260409.md`
- `EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_PLAN_20260409.md`
- `EARLYSTOP_SVD_PROBLEM_CENTERED_ROUND1_RESULTS_20260409.md`
- `SVD_PERF_PLAN_20260411.md`
- `SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`

## 交接文档
- `handoffs/2026-04-05/`：coding-selector 相关的日期化交接记录。

## 参考资料
- `reference/FEATURES.md`：仓库功能与特征清单。
- `reference/LOCAL_CODEX_NOTES_20260403.md`：运行环境与本地注意事项。

## 想法与归档
- `ideas/IDEA.md`：探索性设计记录。
- `archive/raw_sessions/`：历史 raw session 文本归档。
