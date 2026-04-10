# CoT Viewer — Decision-First UX Design Notes (2026-04-09)

## 0) Why this refactor

旧版 `cot_viewer` 的核心价值在于可读链路与底层信号，但主视图是 **slice-first / signal-first**，用户很难在 10 秒内回答：

- 当前方法到底选了哪条 run？
- 为什么它是 top1？
- top1 比 top2 赢在什么关键特征？

本次改造目标是：把首页主视角从“指标浏览器”升级为“决策解释仪表盘”。

---

## 1) 已有能力（保留）

旧版 viewer 已具备并继续保留：

- dataset / problem 选择
- 多 run 并排查看
- token-level 指标：`conf / entropy / gini / self-cert / logprob`
- derivative 面板
- neuron activation 热力图
- Jaccard 相似度曲线

---

## 2) 旧版主要问题

- 信息层级不稳定：高密度低层图占主舞台，决策结论不够突出
- selector 语义不明确：用户看不出“为什么当前方法选了这条 run”
- 方法特征碎片化：SVD/slot100、code_v2、science hybrid 的关键特征没有被组织成方法透镜
- 组内定位弱：虽能看多 run，但“在 64 条里处于什么位置”不够直观

---

## 3) 新信息架构（5层）

### 第1层：Decision Summary（置顶）

- 方法下拉
- Top1 / Top2 / Top3 卡片（run id、✓/✗、score、margin）
- `Why selected?` 人话总结

### 第2层：Method Lens

按方法显示关键特征，不再默认展示无关图：

- `code_v2`：`prefix_best_window_quality`、`head_tail_gap`、`tail_variance`、`post_reflection_recovery`、`last_block_instability`
- `science_hybrid_round3`：baseline / pairwise / hybrid + shortlist/rerank 触发信息
- `slot100_verifier`：slot 轨迹 + 10/40/70/100 anchor 结构
- `extreme8_reflection`：`dc_z`、`dc_r`、`reflection_count_r`

### 第3层：Group Context

- 64-run 组内散点分布（correct vs incorrect）
- Top1/Top2/Top3 高亮
- Top1 vs Top2 / Top1 vs Median 差异条形图

### 第4层：Trajectory & Token Evidence

- Early-stop 轨迹（官方 slots + anchor 复用语义）
- Token 指标对照（默认 Top1 vs Top2）
- 点击特征后文本片段高亮 + token 细节表
- Derivative 作为 secondary 折叠面板

### 第5层：Advanced / Low-level

- neuron activation / jaccard 下沉为默认折叠
- 仅在深挖时展开

---

## 4) 方法覆盖优先级

当前已在 dashboard 中优先支持：

1. `svd_slot100`（`earlystop_prefix10_svd_round1_slot100` 主线，coding 自动 bridge）
2. `slot100_verifier`
3. `svm_bridge_lcb`
4. `code_v2`
5. `science_hybrid_round3`
6. `science_baseline_v1`
7. `extreme8_reflection`
8. `gpqa_pairwise_round1`（比较用）

---

## 5) 工程策略

- 继续使用 Flask + Plotly + Jinja（不引入 React/Vue）
- 优先在线重算 problem 级分数；复用已有模型与缓存产物
- 不重写 cache pipeline，不做大规模离线预计算
- 保留旧 API，同时新增 decision-first API：
  - `/api/method_catalog`
  - `/api/method_scores/<problem_id>`
  - `/api/method_lens/<problem_id>`
  - `/api/run_compare/<problem_id>`
  - `/api/token_evidence/<sample_id>`

---

## 6) 颜色语义（固定）

- 绿色：correct / positive
- 红色：incorrect / risky
- 蓝色：confidence / verifier
- 紫色：reflection / trajectory
- 橙色：instability / anomaly
- 灰色：inactive / fallback

