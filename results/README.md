# Results Layout

## Curated experiment families
- `extreme8_experiments/`: now organized with timestamped snapshots.
- `extreme12_band_experiments/`
- `extreme12_v2_experiments/`
- `reflection_dynamics/`
- `selector_comparison/`
- `trajectory_experiments/`
- `selector_rankings/`: root-level ranking JSON/CSV snapshots moved here for easier browsing.
- `scans/`: small standalone scan outputs grouped by topic.

## Rule of thumb
- Put comparison-ready artifacts in `results/`.
- Put temporary local run outputs in `result/`.
- If a normally-local `result/` run becomes part of a documented study, keep the bundle small and force-add only the minimal payloads needed for reproducibility.

## Current documented exception
- `result/gpqa_deepsets_round1_20260407_run1/`
- `result/gpqa_deepsets_round1_20260407_run2/`
- `result/gpqa_deepsets_round1_20260408_run3/`
- `result/code_deepsets_round1_20260408_run2/`
- `result/math_deepsets_round1_main_20260408_run1/`

These `result/` bundles are intentionally tracked because the corresponding
DeepSets result reports cite them as the official minimal reproducibility payloads
for the documented GPQA / coding / math studies.

---

# Results 布局

## 已整理的实验家族
- `extreme8_experiments/`：按时间戳整理的快照结果。
- `extreme12_band_experiments/`
- `extreme12_v2_experiments/`
- `reflection_dynamics/`
- `selector_comparison/`
- `trajectory_experiments/`
- `selector_rankings/`：根目录散落的 ranking JSON/CSV 已迁到这里，便于浏览。
- `scans/`：按主题归类的小型扫描输出。

## 经验规则
- 可直接比较、适合沉淀的产物放在 `results/`。
- 临时、本地运行输出放在 `result/`。
- 如果一个原本本地的 `result/` 运行变成正式文档的一部分，只保留最小可复现 payload，并按需强制加入版本库。

## 当前已文档化的例外
- `result/gpqa_deepsets_round1_20260407_run1/`
- `result/gpqa_deepsets_round1_20260407_run2/`
- `result/gpqa_deepsets_round1_20260408_run3/`
- `result/code_deepsets_round1_20260408_run2/`
- `result/math_deepsets_round1_main_20260408_run1/`

这些 `result/` 目录之所以被跟踪，是因为对应的 DeepSets 结果文档把它们当作
GPQA / coding / math 三条研究线的最小可复现正式结果载荷。
