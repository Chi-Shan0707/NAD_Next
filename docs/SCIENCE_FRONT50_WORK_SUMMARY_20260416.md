# Science Front50 工作总结（2026-04-16）

## 1. 目标

这轮工作的目标不是重做整套 `earlystop`，而是更聚焦地解决用户最关心的问题：

- 在 `science` 域上继续深挖；
- 重点盯住前半段槽位，也就是 `10% ~ 50%`；
- 不破坏现有 aggressive coding 路线；
- 只替换 `science` 的目标槽位，后半段仍保留 base submission；
- 在 `work/NAD_Next` 内完成，可复现、可解释、可继续迭代。

## 2. 先验判断

在正式搜索前，先基于已有文档与当前代码状态做了三件关键判断：

1. `es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100` 不是一个真正意义上“每个槽位都单独训练后再 earlystop”的 science 路线。
2. 老的 `r2` science cache 仍是旧版 `30` 维特征，而当前代码中的 EarlyStop 特征银行已经扩展到 `47` 维。
3. `science` 的前半段并不是“纯 late signal”，而是早期已经有可用信号，只是需要更合适的特征与跨槽位迁移。

这直接决定了策略：**不要继续迷信旧 science bundle，而要围绕当前 47 维特征银行重建前半段搜索。**

## 3. 做了什么

### 3.1 重建并分析 science 特征

基于当前代码重建了 `science/gpqa` 的特征缓存，并确认新增的后 `17` 个特征在 science 上是活跃的，不是空特征。

随后做了前半段单特征扫描，结论是：

- `10%` 就已经出现强信号：`prefix_best_window_quality`、`tok_conf_prefix`、`tok_conf_recency`、`tail_q10`；
- `20% ~ 30%` 开始，`nc_mean`、`traj_continuity`、`last_block_instability` 明显抬头；
- `40% ~ 50%` 上，`wide46` 这类更宽的特征集合开始持续占优。

对应文档：

- `docs/SCIENCE_FRONT_FEATURE_SCAN_20260416.md`

### 3.2 写了两套搜索基础设施

#### A. 全量 dense-search 基础设施

文件：

- `scripts/baselines/run_science_dense_slot_search.py`

作用：

- 支持基于当前 47 维特征重建 labeled / blind store；
- 支持 science 槽位级搜索；
- 支持 partial-slot patching；
- 支持只替换目标 science 槽位，其余槽位沿用 base submission。

这套脚本偏“全功能”和“长期可扩展”。

#### B. 本轮实际落地的 front50 快速搜索

文件：

- `scripts/baselines/run_science_front50_holdout_search.py`

作用：

- 只搜索 `science` 的 `10,20,30,40,50` 槽位；
- 搜索空间刻意收窄到最有希望的前半段线性路线；
- 只改写 blind science 的 `10%~50%`，`60%~100%` 直接沿用 aggressive base；
- 更适合当前机器上还有其他高负载任务时快速产出可打分 submission。

## 4. 最终采用的策略

最终不是用树模型，也不是用大而全的慢速 dense CV，而是采用了更稳定的 **front50 direct-holdout linear search**。

最终选中的 science 前半段路线如下：

| Target | Source | Route | Family |
|---|---:|---|---|
| `10%` | `10%` | `lr_l1` | `fixed22` |
| `20%` | `30%` | `lr_l2` | `fixed22` |
| `30%` | `50%` | `lr_l2` | `science_front11` |
| `40%` | `50%` | `lr_l1` | `wide46` |
| `50%` | `50%` | `lr_l2` | `wide46` |

这组结果说明了三点：

1. 前两档（`10%`,`20%`）仍然以经典 `fixed22` 为主，说明最早期信号仍然偏“老核心特征”。
2. 到 `30%` 时，新增的前半段特征组合 `science_front11` 开始真正起效。
3. 到 `40%`,`50%`，更宽的 `wide46` 已经明显占优，说明新 47 维银行在 science 中后前段是有实质增益的。

## 5. 离线结果

在前半段 `10%~50%` 的 holdout 上：

- `science r2`: `auc_of_auroc = 0.7847`
- `dense task-specific`: `0.8207`
- `dense final`: `0.8267`

也就是：

- 相比 `r2`，提升约 `+0.0420`
- 相比单纯 task-specific，继续提升约 `+0.0060`

这说明：

- 前半段 science 的提升不是靠“同槽位重训”就能吃满；
- 跨槽位迁移在 `20% -> 30%`、`30% -> 50%` 这类设置上确实有效；
- 新特征并不是全程都赢，而是在 `30%+` 开始更稳定地产生收益。

对应评估文件：

- `results/scans/earlystop/science_front50_holdout_search_20260416_eval.json`
- `results/tables/science_front50_holdout_search_selected_20260416.csv`
- `results/tables/science_front50_holdout_search_manifest_20260416.json`

## 6. 产物

### 核心 submission

- `submission/EarlyStop/es_svd_ms_rr_r1__coding_rotation_adapter_aggressive_tokenonly_late70100__science_front50_holdout_search_20260416.json`

这个文件的行为是：

- `science` 的 `10%~50%` 使用本轮新搜索出的路线；
- `science` 的 `60%~100%` 保留 base aggressive submission；
- 非 science 部分保持原 submission 逻辑不动。

### 说明文档

- `docs/SCIENCE_FRONT_FEATURE_SCAN_20260416.md`
- `docs/SCIENCE_FRONT50_HOLDOUT_SEARCH_20260416.md`
- `docs/SCIENCE_FRONT50_WORK_SUMMARY_20260416.md`

## 7. 验证情况

### 成功项

- 新 submission payload 已成功通过内部结构校验；
- blind science 两个 cache 都完成了特征导出与 patch；
- 最终 payload 验证结果为：
  - `total_problems = 970`
  - `total_samples = 62080`

### 环境项

仓库的 `verify.sh` 跑过一次，但环境里缺少若干仓库级依赖：

- `pyroaring`
- `flask`
- `plotly`
- `hmmlearn`
- `tokenizers`
- `transformers`

所以 `verify.sh` 总体返回失败；这不是本次 science 前半段逻辑本身的错误，而是当前环境缺少可选/仓库全局依赖。

## 8. 本轮最重要的结论

如果只保留一句话，本轮最重要的结论是：

> `science` 前半段并不是没有信号，而是需要“分槽位 + 跨槽位迁移 + 新特征银行”的组合；其中 `30%` 左右开始，新增的前半段特征明显开始有价值。

换成更工程化的表述就是：

- `10%~20%`：老核心特征仍很强；
- `30%`：`science_front11` 开始进入最优解；
- `40%~50%`：`wide46` 接管；
- 因此 science 前半段最值得继续深挖的，不是盲目再做统一 bundle，而是继续做 **slot-aware / feature-aware** 路线。

## 9. 后续建议

如果继续提分，我建议按下面顺序推进：

1. 先实际打分 `science_front50_holdout_search_20260416.json`，确认线上收益是否跟离线方向一致；
2. 若线上有效，再把同样思路扩到 `60%~70%`，看看前中段能否继续吃收益；
3. 若还要继续深挖 `30%`，优先围绕 `science_front11` 做更细粒度 family 搜索，而不是直接回到大而慢的全空间暴搜；
4. 若想更进一步，可以在 blind `cache_test` 上做无监督结构特征补充，但前提是先确认本轮前半段 patch 在线上是否真的涨分。

## 10. 提交记录

本轮与脚本/文档相关的提交为：

- `a57030d` — `baselines: add science front-half holdout search`

这个提交包含：

- science dense-search / partial-slot patch 基础设施；
- science front-half holdout 快速搜索脚本；
- 前半段特征扫描文档；
- front50 搜索文档。
