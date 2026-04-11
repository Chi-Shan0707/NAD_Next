# SVDomain

本目录用于新的 SVD 工作面，不再继续扩散到旧的 `earlystop_prefix10_*` 命名。

## Current Canonical IDs

- `es_svd_math_rr_r1`
- `es_svd_science_rr_r1`
- `es_svd_ms_rr_r1`

## Naming

- `es_svd`：EarlyStop SVD family
- `math` / `science` / `ms`：覆盖域
- `rr`：`raw+rank`
- `r1`：当前分域重训第一版

说明：

- 名字里故意去掉 `prefix` / `p10`
- 但训练协议仍然使用 4 anchors：`10 / 40 / 70 / 100`
- `20 / 30 -> 10`，`50 / 60 -> 40`，`80 / 90 -> 70`

## Current Training Script

- `SVDomain/train_es_svd_ms_rr_r1.py`

## Current Protocol

- 数据：`MUI_HUB/cache` + `MUI_HUB/cache_train`
- 切分：`85 / 15` holdout，按 `dataset + problem_id` 跨 root 一致切分
- 域：`math` / `science` 分开训练
- 表示：只用 `raw+rank`
- 特征：固定为旧 `earlystop_prefix10_svd_round1` noncoding 主特征组
- 路由：不允许 baseline / single-feature route 进入最终模型
- coding：本轮故意不纳入最终 bundle，避免把旧 baseline routing 带进来
