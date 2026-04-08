# Scripts Layout

This folder keeps executable entrypoints at the top level so existing commands continue to work.

## Groups
- `train_*`: model training and selector fitting.
- `run_*`: experiment runners that produce timestamped outputs.
- `export_*`: submission/export helpers.
- `scan_*` / `analyze_*`: one-off analysis or signal scans.
- `plot_*`: visualization and figure generation.
- `rank_*` / `reorder_*` / `rescale_*`: post-processing utilities.
- `test_*.sh` / `compare_*.sh`: compatibility or comparison helpers.

## Recommended reading order
- Start with `run_*` for experiment orchestration.
- Use `train_*` when regenerating models.
- Use `export_*` after a run is frozen.
- Use `rank_*` and `plot_*` for summaries.

## Current notable `run_*` entrypoints
- `run_gpqa_pairwise_round1.py`: frozen GPQA pairwise round-1 reference run.
- `run_gpqa_pairwise_round2.py`: round-2 no-promote recency-family follow-up.
- `run_science_hybrid_round3.py`: current promoted science patch search and proxy evaluation.
- `run_gpqa_deepsets_round1.py`: minimal full-group contextual GPQA DeepSets study.
- `run_code_deepsets_round1.py`: minimal full-group contextual coding DeepSets study.
- `run_math_deepsets_round1.py`: minimal full-group contextual math DeepSets study.
- `patch_bestofn_submission_with_math_deepsets_round1.py`: export the promoted BestofN math patch on top of the current stack.

## Note
- New local-only experiments should prefer subfolders conceptually, but stable user-facing entry scripts remain in `scripts/` for backwards compatibility.

---

# Scripts 布局

这个目录把可执行入口保留在顶层，保证现有命令路径不变。

## 分组
- `train_*`：模型训练与 selector 拟合。
- `run_*`：生成时间戳输出目录的实验 runner。
- `export_*`：提交 / 导出辅助脚本。
- `scan_*` / `analyze_*`：一次性分析或信号扫描。
- `plot_*`：可视化与作图。
- `rank_*` / `reorder_*` / `rescale_*`：后处理工具。
- `test_*.sh` / `compare_*.sh`：兼容性或对比辅助脚本。

## 推荐阅读顺序
- 先看 `run_*`，理解实验编排。
- 需要重训模型时再看 `train_*`。
- 结果冻结后使用 `export_*`。
- 总结和汇总阶段再看 `rank_*` 与 `plot_*`。

## 当前值得关注的 `run_*` 入口
- `run_gpqa_pairwise_round1.py`：冻结的 GPQA pairwise round-1 参考运行。
- `run_gpqa_pairwise_round2.py`：round-2、无 promote 的 recency 家族后续实验。
- `run_science_hybrid_round3.py`：当前 promoted science patch 搜索与 proxy 评估。
- `run_gpqa_deepsets_round1.py`：最小 full-group contextual GPQA DeepSets 实验。
- `run_code_deepsets_round1.py`：最小 full-group contextual coding DeepSets 实验。
- `run_math_deepsets_round1.py`：最小 full-group contextual math DeepSets 实验。
- `patch_bestofn_submission_with_math_deepsets_round1.py`：在当前 stack 上导出 promoted BestofN 数学补丁。

## 说明
- 新的纯本地实验从概念上更适合放子目录，但稳定、面向用户的入口脚本仍保留在 `scripts/` 顶层，以兼容既有调用方式。
