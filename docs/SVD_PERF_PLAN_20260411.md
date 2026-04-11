# SVD Perf Plan (2026-04-11)

## Scope

- 本次先收窄到 `coding` 域。
- 只做 `slot100`，目标是改善 `Best-of-N`。
- 继续保持 `SVD family` 与 `raw+rank + low-rank linear` 主线。

## Coding Slot100 Plan

- 新训练入口：`SVDomain/train_slot100_svd_domain_r1.py`
- 固定搜索：
  - `family`：`svd_code_core` / `svd_code_dyn` / `svd_code_deriv` / `svd_code_messy_all`
  - `head`：`pointwise` 与 `pairwise`
  - `rank`：`4, 8, 12, 16, 24`
  - `whiten`：`False / True`
- 固定比较对象：
  - `code_v2`
  - 当前 `SVD slot100` 直抽线

## Feature Groups

- `code_v2 core`：
  - `prefix_best_window_quality`
  - `head_tail_gap`
  - `tail_variance`
  - `post_reflection_recovery`
  - `last_block_instability`
- `dynamic aux`：
  - `reflection_density`
  - `reflection_count`
  - `traj_continuity`
  - `traj_max_reflection`
  - `traj_late_convergence`
- `terminal aux`：
  - `nc_mean`
  - `nc_slope`
  - `self_similarity`
  - `tail_q10`
  - `last_event_tail_conf`
  - `event_pre_post_delta`
- `slice derivatives`：
  - `conf/gini/entropy` 的 `d1 tail mean`
  - `abs(d1) tail mean`
  - `abs(d2) tail mean`
  - `abs(d1) full minus tail`

## Outputs

- `results/scans/bestofn_bridge/slot100_svd_code_domain_r1_*`
- `submission/BestofN/extreme12/patches/slot100_svd_code_domain_r1__*.json`
- `docs/SVD_SLOT100_DOMAIN_R1_RESULTS_20260411.md`

## Decision Rule

- 优先看 `Hit@1`。
- 若 `Hit@1` 差距小于 `0.2pp`，再看 `Pairwise`。
- 若仍没有单一赢家，输出“双头建议”：
  - `Hit@1-first`
  - `Pairwise-first`
