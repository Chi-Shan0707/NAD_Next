# Reflection follow-up: correlations summary and executable TODO (2026-04-03)

## 1) 已观察到的相关性总结

### 1.1 阈值 sweep 的直接结论
- 当前 reflection event 的原始基线阈值是 `0.30`。
- 在单特征 `reflection_count_r` 的 LOO benchmark 上，最佳阈值是 `0.20`：
  - `0.20` → `71.7%`
  - `0.30` → `71.1%`
- 在 pooled mean 上，`0.25` 略高于 `0.20` / `0.30`：
  - `0.25` → `65.6%`
  - `0.30` → `65.2%`
  - `0.20` → `64.5%`
- 解释：
  - 如果目标是“单特征 reflection_count_r 的 LOO 均值”，当前最优阈值更像是 `0.20`。
  - 如果目标是“pooled 指标”，`0.25` 也值得保留为候选。
  - 但当前已经训练好的 Extreme8 模型仍然是用 `0.30` 训练出来的，因此 test-side 正式 submission 仍优先用 `0.30`，避免 train/test 特征定义漂移。

### 1.2 reflection 与平均统计量的稳定相关性
- reflection 与平均 `gini` **正相关**。
- reflection 与平均 `entropy` **负相关**。
- reflection 与平均 `confidence` 也有正相关，但整体强度弱于 `gini` / `entropy`。
- 这些相关性在 correct / incorrect 两类 run 上都存在，但强度随数据集变化。

### 1.3 最强相关性出现在哪些数据集
根据 `results/reflection_dynamics/summary.md` 中的 Top Correlations：
- `brumo25` 和 `hmmt25` 上最明显。
- `aime25`、`aime24` 也很稳定。
- 代表性数值：
  - `brumo25 / incorrect / gini / avg`: `+0.4911`
  - `brumo25 / incorrect / entropy / avg`: `-0.4895`
  - `hmmt25 / incorrect / gini / avg`: `+0.4572`
  - `hmmt25 / incorrect / entropy / avg`: `-0.4555`
  - `aime24 / correct / gini / avg`: `+0.4370`
  - `aime24 / correct / entropy / avg`: `-0.4267`

### 1.4 reflection event slice 与非 event slice 的差异
- reflection event slice 上，`confidence` 的 `abs_d1` / `abs_d2` gap 整体更低。
- 这说明 event slice 附近的置信度变化并不是“更尖锐地跳动”，反而经常表现为**更平滑 / 更收敛**的局部变化模式。
- 最明显的 gap 出现在：
  - `gpqa / correct / conf / abs_d2`: `-0.2654`
  - `brumo25 / correct / conf / abs_d2`: `-0.2355`
  - `aime24 / correct / conf / abs_d1`: `-0.2082`
  - `livecodebench_v5 / correct / conf / abs_d2`: `-0.1738`

### 1.5 现阶段最合理的解释
当前证据更支持下面这个解释：
- reflection 不是单纯“混乱”或“高波动”的信号。
- 它更像是与一种**特定的不确定性重整 / 再组织过程**相关：
  - 平均 `gini` 更高
  - 平均 `entropy` 更低
  - event 邻域的置信度离散变化更小
- 因此，下一步不该只问“reflection_count 有没有用”，而该问：
  - reflection 是否提供了 **独立于 entropy / gini / dc_* 的增量信息**？
  - 哪类 reflection 是“有助于恢复正确答案”的，哪类只是“无效反刍”？

---

## 2) 详细可执行 TODO

下面的 TODO 按优先级排列；每项都尽量给出目标、实现位置、产物和验收标准。

### P0 — 先验证 reflection 是否真的有独立增益

#### TODO P0.1：做“去相关 / 残差化”验证
**目标**
- 判断 `reflection_count_*` 是否只是 `avg_entropy` / `avg_gini` / `dc_*` 的替身。

**建议实现**
- 新增脚本：`scripts/analyze_reflection_incremental_value.py`
- 输入：现有 reflection-dynamics 的 per-run 明细表（若脚本当前未落明细表，则先在 `scripts/analyze_reflection_dynamics.py` 增加明细导出）
- 做三组比较：
  1. `entropy + gini`
  2. `entropy + gini + reflection_count`
  3. `entropy + gini + dc_* + reflection_count`
- 再做一版 residual test：
  - 先用 `entropy/gini/conf` 回归 `reflection_count`
  - 取残差 `reflection_residual`
  - 再测试 `reflection_residual` 是否仍能提升 LOO / pooled / pairwise 指标

**产物**
- `results/reflection_dynamics/incremental_value_summary.json`
- `results/reflection_dynamics/incremental_value_table.md`

**验收标准**
- 如果加入 reflection 后在 6 数据集均值上仍稳定提升（例如 `>= +0.3pp`），则保留 reflection 作为独立信息源。
- 如果提升只在少数数据集出现，说明应把 reflection 作为 dataset-conditional feature，而不是全局核心特征。

#### TODO P0.2：把 “event-local” 特征做出来，而不是只用 count
**目标**
- 从“reflection 总次数”升级到“reflection 发生时的结构特征”。

**建议新增特征**
- `reflection_event_ratio`：event slice 占比
- `first_reflection_pos_r`：第一次 reflection 的相对位置
- `last_reflection_pos_r`：最后一次 reflection 的相对位置
- `mean_gap_conf_abs_d1`：event vs non-event 的 `conf abs_d1` gap
- `mean_gap_conf_abs_d2`：event vs non-event 的 `conf abs_d2` gap
- `post_event_conf_recovery`：event 之后窗口的 conf 恢复幅度
- `post_event_entropy_drop`：event 之后窗口的 entropy 下降幅度
- `reflection_burst_count`：连续 reflection 段的段数
- `max_reflection_burst_len`：最长 burst 长度

**建议实现**
- 扩展：`scripts/analyze_reflection_dynamics.py`
- 新增导出：`results/reflection_dynamics/per_run_event_features.csv`

**产物**
- 一张 feature inventory 表：每个特征的定义、值域、缺失处理方式
- 每个特征的单特征 LOO / pooled 结果

**验收标准**
- 至少找出 2–4 个 event-local 特征，其表现不弱于 `reflection_count_r` 的 80–90%。
- 若有 event-local 特征超过 `reflection_count_r`，则下一轮 selector 训练应优先改用这些特征。

#### TODO P0.3：把阈值选择目标对齐到下游任务
**目标**
- 不再只用 “单特征 LOO 最优” 决定阈值，而是看下游 Best-of-N / selector 任务真正受益的阈值。

**建议比较的阈值**
- `0.20`
- `0.25`
- `0.30`

**建议实现**
- 在 `scripts/train_extreme8_selectors.py` 和 `scripts/run_extreme8_experiments.py` 增加统一参数 sweep 入口，或新建：
  - `scripts/compare_reflection_thresholds_downstream.py`
- 在 `cache_train` 上做带标签评估：
  - Hit@1
  - Hit@3
  - Pairwise accuracy
  - SelAcc@10%
  - best_only / mix 排名指标

**产物**
- `results/reflection_dynamics/downstream_threshold_comparison.json`
- `results/reflection_dynamics/downstream_threshold_comparison.md`

**验收标准**
- 若 `0.20` 在下游排序/选择任务上也稳定优于 `0.30`，才推动重训。
- 若 `0.20` 只提升单特征，不提升下游 selector，则当前模型仍继续用 `0.30`。

---

### P1 — 把 reflection 信号升级成 selector 特征

#### TODO P1.1：做一版 “reflection-augmented Extreme8” 候选特征集
**目标**
- 在保留当前 3 特征基线的同时，验证 reflection event-local 特征是否能提高 Best-of-N 排序。

**建议实现**
- 扩展 `nad/core/selectors/extreme8_impl.py`
- 保留 baseline 特征集：
  - `dc_z`
  - `dc_r`
  - `reflection_count_r`
- 新增候选特征开关，例如：
  - `reflection_event_ratio`
  - `first_reflection_pos_r`
  - `mean_gap_conf_abs_d2`
  - `post_event_conf_recovery`
- 训练脚本里增加 feature-set 选项：
  - `baseline3`
  - `baseline3+event2`
  - `baseline3+event4`

**产物**
- 新训练统计：`models/ml_selectors/extreme8_reflection_aug_stats.json`
- 多版本评估汇总：`results/extreme8_experiments/reflection_aug_summary.json`

**验收标准**
- 在 train-eval 上，新的 feature set 至少在 `best_only` 或 `mix` 中一个主指标上稳定提升。
- 如果提升只来自单一 dataset，则暂不替换 baseline，只保留为实验分支。

#### TODO P1.2：显式测试 interaction term，而不是只堆原始特征
**目标**
- 判断 reflection 是否主要通过与 `dc_*` / entropy / gini 的交互发挥作用。

**建议候选交互项**
- `dc_z * reflection_count_r`
- `dc_r * reflection_event_ratio`
- `avg_entropy * reflection_count_r`
- `avg_gini * post_event_conf_recovery`
- `first_reflection_pos_r * output_length_r`

**建议实现**
- 先在离线表格上做 logistic / linear probe 对比，再决定是否合入主 selector。

**产物**
- `results/reflection_dynamics/interaction_ablation.json`
- 一张交互项排序表

**验收标准**
- 若交互项 consistently 优于原始 reflection count，则之后的 selector 应优先保留交互项而不是 count 本身。

---

### P2 — 把 reflection 用在“预算分配 / 路由”上，而不仅是静态打分

#### TODO P2.1：定义 “有用 reflection” vs “无用 reflection”
**目标**
- 把所有 reflection 区分成至少两类：
  1. 触发后能恢复、最终更可信
  2. 触发后没有恢复、只是重复震荡

**建议标签构造**
- 先不做人为主观标注，先用 proxy：
  - event 后 conf 是否回升
  - event 后 entropy 是否下降
  - 最终答案是否正确
  - 最后 20% token 是否更稳定

**建议实现**
- 新建：`scripts/cluster_reflection_patterns.py` 或 `scripts/label_reflection_patterns.py`
- 先做 rule-based 分群，再看是否值得上简单聚类

**产物**
- `results/reflection_dynamics/reflection_pattern_summary.md`
- 若干典型 case 列表（每类挑 10–20 条）

**验收标准**
- 至少形成 2–3 类稳定可解释的 reflection pattern
- 每类 pattern 与最终正确率 / 排序价值存在明显差异

#### TODO P2.2：做一个“budget routing”原型
**目标**
- 不是只问哪个 run 分数高，而是决定哪些问题/轨迹值得分配更多计算预算。

**候选策略**
- 若 reflection pattern 显示“有恢复潜力”，增加 blind tuples 或保留更多候选
- 若 reflection pattern 显示“持续无恢复”，在 mix 里加大惩罚或直接降权

**建议实现**
- 新建：`scripts/simulate_reflection_budget_routing.py`
- 在 `cache_train` 上模拟：固定总预算下，路由策略是否能提升 Hit@1 / SelAcc@10%

**产物**
- `results/reflection_dynamics/budget_routing_summary.json`

**验收标准**
- 在固定预算下，路由版本应优于固定 uniform sampling
- 如果只提升少量但代价复杂，则先不接入主流程

---

## 3) 建议执行顺序（最务实版本）

### 第一轮：只做最便宜但信息量最大的实验
1. 做 `P0.1`：验证 reflection 是否有独立增益
2. 做 `P0.2`：把 event-local 特征导出来
3. 做 `P0.3`：在下游任务上对比 `0.20 / 0.25 / 0.30`

### 第二轮：如果第一轮证明 reflection 不是伪相关
4. 做 `P1.1`：训练 reflection-augmented Extreme8
5. 做 `P1.2`：筛 interaction term

### 第三轮：如果 selector 上已经看到收益
6. 做 `P2.1`：区分“有用/无用 reflection”
7. 做 `P2.2`：把它用于 budget routing

---

## 4) 立即可执行的具体下一步

如果只做一个最有价值、最不容易跑偏的下一步，建议是：

### Next action
- **先做 `P0.3`：把 `0.20 / 0.25 / 0.30` 三个阈值直接放到 downstream selector / Best-of-N 指标里比较。**

### 为什么先做这个
- 它直接回答：
  - 单特征里看起来更好的阈值，是否真的能改善最终选 best / 避免 worst？
- 如果答案是否定的，就没必要太早重训和改 feature set。
- 如果答案是肯定的，再去做更贵的 feature engineering 就更有把握。

### 最小交付物
- 一份 `downstream_threshold_comparison.json`
- 一份 `downstream_threshold_comparison.md`
- 明确写出：
  - 哪个阈值在单特征最好
  - 哪个阈值在 Best-of-N 下游最好
  - 当前 production / submission 默认应该用哪个阈值
