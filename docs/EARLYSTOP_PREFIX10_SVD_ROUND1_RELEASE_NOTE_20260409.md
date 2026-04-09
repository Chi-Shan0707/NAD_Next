# EARLYSTOP PREFIX10 SVD ROUND1 RELEASE NOTE (2026-04-09)

## 摘要

- 本轮只做 Early-Stop，没有改 `Best-of-N` 主线，也没有碰 `code_v2` / `science_hybrid` / `math_deepsets`。
- 方法主体仍是 `low-rank SVD + LogisticRegression`，但特征被严格限制在 prefix-safe 视角。
- 最终导出的候选是 `noncoding_anchor4_coding_v1`：
  - non-coding（math + science）使用本轮 prefix-safe anchor4 模型；
  - coding 直接回退到 `earlystop_svd_lowrank_lr_v1`。

## 训练 / 自测 / 榜单分别是什么

- `训练 / 搜索`：
  - 数据来源：`MUI_HUB/cache`
  - 范围：`6` 个 labeled cache
  - 明细：`DS-R1/aime24`、`DS-R1/aime25`、`DS-R1/brumo25`、`DS-R1/gpqa`、`DS-R1/hmmt25`、`DS-R1/lcb_v5`
- `离线自测`：
  - 仍然使用同一个 `MUI_HUB/cache`
  - 路由搜索时使用 grouped CV
  - 但最终对比表中的提升，依然是训练侧 / 自测侧 direct-eval，不是 blind leaderboard 提升
- `blind leaderboard`：
  - 数据来源：`/home/jovyan/public-ro/MUI_HUB/cache_test`
  - 范围：`12` 个 test cache（`DS-R1` + `Qwen3-4B`）
  - 榜单回执：`submission/resultofleaderboard/Early Stop — earlystop_prefix10_svd_round1.txt`

## 离线自测提升（on `MUI_HUB/cache`）

- 对比对象：`earlystop_svd_lowrank_lr_v1`
- 候选方法：`noncoding_anchor4_coding_v1`
- 提升如下：
  - `AUC of AUROC`: `77.10% -> 81.33%`（`+4.23pt`）
  - `AUC of SelAcc`: `91.76% -> 91.88%`（`+0.12pt`）
  - `AUROC@100%`: `82.31% -> 84.28%`（`+1.97pt`）
  - `Stop Acc@100%`: `71.83% -> 75.72%`（`+3.89pt`）

## Blind Leaderboard 结果（on `cache_test`）

- 提交方法：`earlystop_prefix10_svd_round1`
- 提交文件：`submission/EarlyStop/earlystop_prefix10_svd_round1.json`
- leaderboard 结果：
  - `Avg Rank = 4.0000`
  - `AUC of SelAcc = 0.8311`
  - `Earliest >0.6 = 10%`
  - `AUROC@100% = 0.8492`
  - `Stop Acc@100% = 0.7504`
- 与 blind leaderboard 上的 `early-stop v1` 相比：
  - `Avg Rank`: `4.0000` vs `2.5625`，更差
  - `AUC of SelAcc`: `0.8311` vs `0.8483`，更差
  - `AUROC@100%`: `0.8492` vs `0.8456`，更好
  - `Stop Acc@100%`: `0.7504` vs `0.7351`，更好

## 解读

- 这轮方法在训练侧 / 离线自测上，确实比 `earlystop_svd_lowrank_lr_v1` 更强。
- 但 blind leaderboard 没有同步转化成更好的综合排名。
- 现象上更像是：
  - 非 coding 的 prefix-safe 路由在训练侧收益明显；
  - `100%` 末端 stop 指标也确实提高；
  - 但 blind test 上的整体 `AUC of SelAcc` 与若干中间位置排名指标不如 `v1`，导致 `Avg Rank` 反而下降。

## 相关文件

- `docs/EARLYSTOP_PREFIX10_SVD_ROUND1_RESULTS_20260408.md`
- `results/scans/earlystop/earlystop_prefix10_svd_round1_summary.json`
- `results/scans/earlystop/earlystop_prefix10_svd_round1_eval.json`
- `submission/EarlyStop/earlystop_prefix10_svd_round1.json`
- `submission/resultofleaderboard/Early Stop — earlystop_prefix10_svd_round1.txt`
