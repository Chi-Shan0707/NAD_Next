# Models Layout

This folder stores trained selector artifacts and related metadata.

## Main subtree
- `ml_selectors/`: sklearn/joblib-based selector models and stats.

## Current notable tracked artifacts
- `ml_selectors/gpqa_pairwise_round1.pkl`: frozen GPQA pairwise round-1 reference model.
- `ml_selectors/gpqa_deepsets_round1.pkl`: GPQA DeepSets round-1 reference artifact, currently pointed at the best round-1 candidate `max_pairaux0p25` even though the line is `NO-PROMOTE`.
- `ml_selectors/gpqa_deepsets_round1_mean.pkl`
- `ml_selectors/gpqa_deepsets_round1_max.pkl`
- `ml_selectors/gpqa_deepsets_round1_max_pairaux0p25.pkl`
- `ml_selectors/gpqa_deepsets_round1_max_pairaux0p50.pkl`
- `ml_selectors/code_deepsets_round1.pkl`: coding DeepSets round-1 reference artifact; line remains `NO-PROMOTE`.
- `ml_selectors/code_deepsets_round1_mean.pkl`
- `ml_selectors/code_deepsets_round1_max.pkl`
- `ml_selectors/code_deepsets_round1_max_pairaux0p25.pkl`
- `ml_selectors/math_deepsets_round1.pkl`: promoted math DeepSets round-1 reference artifact, pointed at `max_pairaux0p25`.
- `ml_selectors/math_deepsets_round1_mean.pkl`
- `ml_selectors/math_deepsets_round1_max.pkl`
- `ml_selectors/math_deepsets_round1_max_pairaux0p25.pkl`

## Organization rule
- Keep stable, code-referenced model files in predictable paths.
- Put local or domain-specific exploratory variants under dedicated subfolders instead of the root.

---

# Models 布局

这个目录保存训练好的 selector 模型产物及其相关元数据。

## 主要子树
- `ml_selectors/`：基于 sklearn / joblib 的 selector 模型与统计文件。

## 当前值得关注的已跟踪产物
- `ml_selectors/gpqa_pairwise_round1.pkl`：冻结的 GPQA pairwise round-1 参考模型。
- `ml_selectors/gpqa_deepsets_round1.pkl`：GPQA DeepSets round-1 参考产物，当前指向本轮最佳候选 `max_pairaux0p25`，但该主线结论仍是 `NO-PROMOTE`。
- `ml_selectors/gpqa_deepsets_round1_mean.pkl`
- `ml_selectors/gpqa_deepsets_round1_max.pkl`
- `ml_selectors/gpqa_deepsets_round1_max_pairaux0p25.pkl`
- `ml_selectors/gpqa_deepsets_round1_max_pairaux0p50.pkl`
- `ml_selectors/code_deepsets_round1.pkl`：coding DeepSets round-1 参考产物；该主线结论仍是 `NO-PROMOTE`。
- `ml_selectors/code_deepsets_round1_mean.pkl`
- `ml_selectors/code_deepsets_round1_max.pkl`
- `ml_selectors/code_deepsets_round1_max_pairaux0p25.pkl`
- `ml_selectors/math_deepsets_round1.pkl`：已 promote 的 math DeepSets round-1 参考产物，当前指向 `max_pairaux0p25`。
- `ml_selectors/math_deepsets_round1_mean.pkl`
- `ml_selectors/math_deepsets_round1_max.pkl`
- `ml_selectors/math_deepsets_round1_max_pairaux0p25.pkl`

## 组织规则
- 代码稳定引用的模型文件应保留在可预测路径中。
- 本地或领域特定的探索性变体，优先放到专门子目录，而不是根目录。
