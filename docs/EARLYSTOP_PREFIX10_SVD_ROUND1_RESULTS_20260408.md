# EARLYSTOP PREFIX10 / ANCHOR4 SVD ROUND1 RESULTS (2026-04-08)

## 开头确认

- repo 已经有 `earlystop_svd_lowrank_lr_v1` 的导出链路
- 这次任务只做 Early-Stop
- 目标是提高公开/综合 Early-Stop 分数，不是扩平台
- 本轮核心假设：prefix 前 10% 已包含足够强的早停信号，值得单独训练一个 prefix-10 专用 low-rank 模型

## 0. 协议更正

- round1 **没有**把 `cache_train` 纳入训练 / 验证协议；这一点后来被确认是方法学缺口。
- round1 表里的“离线自测提升”来自 `MUI_HUB/cache` 的训练侧 / 同源 direct-eval，不能视为独立 holdout 提升。
- 因此，round1 的独立证据其实只有 blind leaderboard；而 blind leaderboard 的综合排名并没有超过 `earlystop_svd_lowrank_lr_v1`。
- 后续 round1b 已改成：`MUI_HUB/cache` + `cache_train` 扩大训练池，并对 non-coding 做确定性 holdout split。

## 1. 我确认的当前 repo 状态

- 已读 `docs/README.md`、`docs/WORK_SUMMARY_0408_06.md`、`nad/ops/earlystop_svd.py`、`scripts/export_earlystop_svd_submission.py`。
- `scripts/export_earlystop_svd_submission.py` 已可直接加载 `models/ml_selectors/earlystop_svd_lowrank_lr_v1.pkl` 导出 Early-Stop submission。
- 当前 repo 内可直接复跑的 Early-Stop 导出线至少包括：`tok_conf_prefix_mean_v1`、`earlystop_svd_lowrank_lr_v1`、`earlystop_from_bestofn_svm_bridge_v1`。
- 从现有反馈文本看，repo 内已导出线里 `earlystop_svd_lowrank_lr_v1` 强于 `earlystop_from_bestofn_svm_bridge_v1`；`benchmark_early_stop_v1` 只在反馈文本中出现，不是当前仓库内可直接复跑的实现。

## 2. 训练 / 自测 / 榜单口径（round1 原始协议）

- `训练 / 搜索`：使用 `MUI_HUB/cache` 的 `6` 个 labeled cache，分别是 `DS-R1/aime24`、`DS-R1/aime25`、`DS-R1/brumo25`、`DS-R1/gpqa`、`DS-R1/hmmt25`、`DS-R1/lcb_v5`。
- `离线自测`：仍然在同一个 `MUI_HUB/cache` 上做 direct-eval；路由选择用 grouped CV，但表中的提升是同一批 labeled cache 上的训练侧 / 自测侧提升，不是 blind leaderboard 提升。
- `盲榜 / leaderboard`：导出到 `/home/jovyan/public-ro/MUI_HUB/cache_test`，覆盖 `12` 个 test cache（`DS-R1` + `Qwen3-4B` 各 6 个 benchmark）。榜单回执见 `submission/resultofleaderboard/Early Stop — earlystop_prefix10_svd_round1.txt`。
- `离线自测相对 v1 的提升`：`AUC of AUROC +4.23pt`、`AUC of SelAcc +0.12pt`、`AUROC@100% +1.97pt`、`Stop Acc@100% +3.89pt`；但这些提升属于 **train-side / in-sample** 指标。
- `盲榜相对 v1 的结果`：`Avg Rank 4.0000 vs 2.5625`（更差），`AUC of SelAcc 0.8311 vs 0.8483`（更差），但 `AUROC@100% 0.8492 vs 0.8456`、`Stop Acc@100% 0.7504 vs 0.7351`（更好）。

## 3. prefix-10 特征定义

### 保留特征

- `token_only`: `tok_conf_*`、`tok_gini_*`、`tok_neg_entropy_*`、`tok_selfcert_*`、`tok_logprob_*` 与对应 token availability flags。
- `token_plus_traj`: 上述 token 特征 + `traj_continuity`、`traj_reflection_count`、`traj_novelty`、`traj_max_reflection`、`traj_late_convergence` + `has_rows_bank`。
- `all`: `token_plus_traj` + `nc_mean`、`nc_slope` + 全部 availability flags。

### 删除特征

- `self_similarity`：现有实现按全序列前半/后半计算，会把 50%/100% 后验信息泄露进前缀视角；本轮明确删除。

## 4. 5/10/15/20 checkpoint 对照

### 全域统一单 checkpoint

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | 76.38% | 89.19% | 72.03% |
| 10% | 77.06% | 88.19% | 71.82% |
| 15% | 77.76% | 88.22% | 70.27% |
| 20% | 78.78% | 89.57% | 71.43% |

### 非 coding 单 checkpoint

| Checkpoint | AUROC | SelAcc@10 | StopAcc |
|---|---:|---:|---:|
| 5% | 83.00% | 96.57% | 71.09% |
| 10% | 84.54% | 96.36% | 72.06% |
| 15% | 85.14% | 96.88% | 72.26% |
| 20% | 85.93% | 97.81% | 72.73% |

- 这些对照只用于验证早段 checkpoint 信号强弱，不直接作为最终十槽 submission。
- 离线 control 结果并不支持“10% 单点最优”：无论全域还是非 coding，`20%` 都优于 `10%`。
- 因此本轮最终胜出并不是靠“纯 10% 单 checkpoint”，而是靠 prefix-safe 训练 + 非 coding 专用 anchor4 + coding 回退到 v1。

## 5. 与 `earlystop_svd_lowrank_lr_v1` 的对比（离线自测 on `MUI_HUB/cache`）

### 整体对比

| Method | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| tok_conf_prefix_mean_v1 | 67.68% | 87.55% | 10% | 68.54% | 67.75% |
| earlystop_from_bestofn_svm_bridge_v1 | 75.32% | 91.33% | 10% | 82.07% | 71.49% |
| earlystop_svd_lowrank_lr_v1 | 77.10% | 91.76% | 10% | 82.31% | 71.83% |
| global_anchor4 | 79.42% | 90.26% | 10% | 81.50% | 70.49% |
| noncoding_anchor4_coding_v1 | 81.33% | 91.88% | 10% | 84.28% | 75.72% |

### noncoding_anchor4_coding_v1 per-cache

| Cache | AUC of AUROC | AUC of SelAcc | Earliest >0.6 | AUROC@100% | Stop Acc@100% |
|---|---:|---:|---:|---:|---:|
| DS-R1/aime24 | 89.56% | 99.74% | 10% | 93.49% | 90.00% |
| DS-R1/aime25 | 90.22% | 99.84% | 10% | 94.26% | 83.33% |
| DS-R1/brumo25 | 92.31% | 99.53% | 10% | 94.58% | 90.00% |
| DS-R1/gpqa | 74.78% | 93.69% | 10% | 77.18% | 67.17% |
| DS-R1/hmmt25 | 88.99% | 99.95% | 10% | 93.91% | 63.33% |
| DS-R1/lcb_v5 | 52.10% | 58.54% | N/A | 52.24% | 60.48% |

### Anchor 映射

- 正式 bundle 采用四个锚点模型：`10/40/70/100`。
- 十槽映射：`10/20/30→10`，`40/50/60→40`，`70/80/90→70`，`100→100`。
- `global_anchor4`：math/science/coding 全域统一训练。
- `noncoding_anchor4_coding_v1`：math+science 统一训练，coding 直接复用 `earlystop_svd_lowrank_lr_v1` 原生十槽路由。

## 6. 是否导出新 submission

- 结论：`YES`（这是 round1 当时基于 train-side direct-eval 作出的决定）。
- 判定理由：offline direct-eval strict dominance over earlystop_svd_lowrank_lr_v1。
- 事后复盘：这个导出决策的证据强度不够，因为它没有经过 `cache_train` 或其它独立 holdout 口径验证。

## 7. blind leaderboard 结果（2026-04-09）

- 提交文件：`submission/EarlyStop/earlystop_prefix10_svd_round1.json`
- leaderboard 方法名：`earlystop_prefix10_svd_round1`
- 榜单记录文件：`submission/resultofleaderboard/Early Stop — earlystop_prefix10_svd_round1.txt`
- 结果：`Rank 4`，`Avg Rank 4.0000`
- 与盲榜上的 `early-stop v1` 相比：`AUC of SelAcc` 下降（`0.8311 < 0.8483`），但 `AUROC@100%` 与 `Stop Acc@100%` 上升（`0.8492 > 0.8456`，`0.7504 > 0.7351`）。
- 结论：本轮方法在训练侧 / 离线自测上确实优于 `v1`，但 blind leaderboard 的综合排名没有超过 `v1`。

## 8. 如果没有胜出，失败原因是什么

- 事后复盘口径下，真正的问题是：`cache_train` 没被纳入协议，导致 train-side direct-eval 过于乐观，而 blind leaderboard 综合排名没有超过 `v1`。

## 9. 改了哪些文件

- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1_PLAN_20260408.md`
- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1_RESULTS_20260408.md`
- `scripts/run_earlystop_prefix10_svd_round1.py`
- `scripts/export_earlystop_svd_submission.py`
- `nad/ops/earlystop_svd.py`
- `nad/core/selectors/trajectory_impl.py`
- `submission/EarlyStop/earlystop_prefix10_svd_round1.json`

## 10. 如何复跑

```bash
bash cookbook/00_setup/verify.sh
python3 -m nad.cli --help
python3 scripts/run_earlystop_prefix10_svd_round1.py \
  --cache-root MUI_HUB/cache \
  --test-cache-root /home/jovyan/public-ro/MUI_HUB/cache_test
```

```bash
python3 scripts/export_earlystop_svd_submission.py \
  --cache-root /home/jovyan/public-ro/MUI_HUB/cache_test \
  --model-path models/ml_selectors/earlystop_prefix10_svd_round1.pkl \
  --method-name earlystop_prefix10_svd_round1 \
  --filename earlystop_prefix10_svd_round1.json
```
