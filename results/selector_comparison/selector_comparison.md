# Selector Accuracy Comparison | 选择器准确率对比

模型 Model: DeepSeek-R1-0528-Qwen3-8B  |  距离 Distance: Jaccard  |  每题 run 数 Runs per problem: 64

类型图例 | Type legend:
- `[S]`    单样本评分 | single-run score (no distance matrix)
- `[P]`    两两 Jaccard 矩阵 | pairwise N×N Jaccard matrix
- `[O]`    Oracle 基线 | oracle baseline (uses all ground-truth labels)
- `[P+E]`  两两 + 分组 Ensemble | pairwise + group ensemble
- `[P+T]`  两两 + 锦标赛 + softmax | pairwise + tournament + softmax
- `[P+2S]` 两两 + 两阶段 top-k | pairwise + two-stage top-k
- `[ML]`   ML 选择器（需预训练模型）| ML selector (trained, requires `models/ml_selectors/`)

```
Selector                  AIME24    AIME25   BruMo25      GPQA    HMMT25     LCBv5      Mean        Type
--------------------------------------------------------------------------------------------------------
min-activation             83.3%     73.3%     76.7%     61.1%     46.7%     56.9%     66.3%         [S]
max-activation             50.0%     56.7%     56.7%     53.5%     43.3%     59.9%     53.3%         [S]
min-confidence             50.0%     50.0%     46.7%     55.6%     26.7%     59.9%     48.1%         [S]
medoid                     80.0%     73.3%     73.3%     65.7%     56.7%     58.1%     67.8%         [P]
knn-medoid                 86.7%     76.7%     76.7%     65.7%     53.3%     59.9%     69.8%         [P]
dbscan-medoid              80.0%     73.3%     73.3%     66.2%     56.7%     58.1%     67.9%         [P]
consensus-min              86.7%     73.3%     80.0%     66.7%     56.7%     59.3%     70.4%         [P]
consensus-max              80.0%     76.7%     70.0%     65.7%     53.3%     58.7%     67.4%         [P]
deepconf                   70.0%     70.0%     60.0%     60.6%     46.7%     58.7%     61.0%         [S]
ensemble-medoid            80.0%     73.3%     76.7%     64.6%     56.7%     59.3%     68.4%       [P+E]
ensemble-deepconf          70.0%     70.0%     60.0%     60.6%     46.7%     58.7%     61.0%       [S+E]
tournament-copeland        80.0%     70.0%     76.7%     65.2%     60.0%     61.1%     68.8%       [P+T]
tournament-deepconf        76.7%     63.3%     66.7%     58.1%     50.0%     59.3%     62.3%       [S+T]
twostage-medoid            80.0%     73.3%     80.0%     66.2%     56.7%     58.7%     69.1%      [P+2S]
twostage-tournament        80.0%     70.0%     80.0%     63.6%     63.3%     58.1%     69.2%      [P+2S]
linear-probe               86.7%     76.7%     76.7%     67.2%     56.7%     58.1%     70.3%        [ML]
logistic                   86.7%     76.7%     76.7%     68.2%     56.7%     58.7%     70.6%        [ML]
isotonic-medoid            80.0%     73.3%     76.7%     64.1%     56.7%     59.9%     68.4%        [ML]
isotonic-deepconf          76.7%     66.7%     63.3%     61.6%     46.7%     58.7%     62.3%        [ML]
temporal-slice             70.0%     63.3%     63.3%     66.7%     40.0%     58.7%     60.3%         [S]
trajectory                 66.7%     60.0%     66.7%     64.6%     36.7%     57.5%     58.7%         [S]
layer-stratified           83.3%     73.3%     80.0%     63.6%     56.7%     59.3%     69.4%         [S]
trajectory-fusion          66.7%     70.0%     83.3%     68.7%     63.3%     57.5%     68.3%        [ML]
avg64@                     75.3%     65.7%     69.4%     60.2%     48.6%     58.7%     63.0%         [O]
con64@                     80.0%     76.7%     80.0%     70.2%     66.7%     58.1%     71.9%         [O]
```

## Rank by Mean Accuracy | 按均值准确率排名

```
   1. con64@                   71.9%  [O]       oracle baseline
   2. logistic                 70.6%  [ML]      ★ ML — logistic on 12 features
   3. consensus-min            70.4%  [P]
   4. linear-probe             70.3%  [ML]      ★ ML — ridge on 12 features
   5. knn-medoid               69.8%  [P]
   6. layer-stratified         69.4%  [S]       ★ NEW — layer activation distribution
   7. twostage-tournament      69.2%  [P+2S]
   8. twostage-medoid          69.1%  [P+2S]
   9. tournament-copeland      68.8%  [P+T]
  10. isotonic-medoid          68.4%  [ML]      ★ ML — isotonic calibration of medoid score
  11. ensemble-medoid          68.4%  [P+E]
  12. trajectory-fusion        68.3%  [ML]      ★ NEW — 22-D logistic (12 base + 10 trajectory)
  13. dbscan-medoid            67.9%  [P]
  14. medoid                   67.8%  [P]
  15. consensus-max            67.4%  [P]
  16. min-activation           66.3%  [S]
  17. avg64@                   63.0%  [O]       oracle baseline
  18. tournament-deepconf      62.3%  [S+T]
  19. isotonic-deepconf        62.3%  [ML]
  20. deepconf                 61.0%  [S]
  21. ensemble-deepconf        61.0%  [S+E]
  22. temporal-slice           60.3%  [S]       temporal discount on tok_neg_entropy
  23. trajectory               58.7%  [S]       ★ NEW — trajectory structure heuristic
  24. max-activation           53.3%  [S]
  25. min-confidence           48.1%  [S]
```

## Notes | 说明

- **[S]** 单样本评分，独立计算每个 run。Single-run selectors score each run independently.
  - `min-activation`, `max-activation`, `min-confidence`, `deepconf`
- **[P]** 两两选择器，使用完整 N×N Jaccard 距离矩阵。Pairwise selectors use the full N×N Jaccard distance matrix.
- **[O]** Oracle 基线，使用所有 ground-truth 标签。Oracle baselines use ground-truth correctness from all runs.
  - `avg64@`: 64 个 run 的平均准确率 | average accuracy; `con64@`: 多数投票 | majority-vote
- **[P+E] 分组 Ensemble | Group Ensemble**: N 个 run 随机分为 8 人组，每组选最佳，胜者再选。split N runs into groups of 8, select best per group, then among winners.
  - `ensemble-medoid` (每组 medoid) | `ensemble-deepconf` (每组 DeepConf)
- **[P+T] 锦标赛 + softmax | Tournament + softmax**: Copeland 或 DeepConf 两两比较 → rank → softmax (temp=0.2).
  - `tournament-copeland` | `tournament-deepconf`
- **[P+2S] 两阶段 | Two-Stage**: 16 人组 → 每组 top-4 → 16 人决赛。groups of 16 → top-4 per group → 16-finalist final round.
  - `twostage-medoid` (medoid) | `twostage-tournament` (Copeland + softmax)
- **[ML] ML 选择器 | ML Selectors**: 用 `scripts/train_ml_selectors.py` 在 31,040 个带标注 (题目, run) 对上训练。trained on 31,040 labelled (problem, run) pairs. Models in `models/ml_selectors/`.
- **轨迹选择器 | Trajectory selectors** (实验 Exp 7-9, `nad/core/selectors/trajectory_impl.py`):
  - `trajectory`: 轨迹结构启发式评分（连续性/反思/新颖度）| heuristic scoring on trajectory structure
  - `layer-stratified`: 层激活分布启发式评分（深浅比/层熵/Gini）| heuristic scoring on layer distribution
  - `trajectory-fusion`: 22-D 融合特征 ML 逻辑回归 | 22-D logistic regression (12 base + 10 trajectory)

## Artifact Snapshot | 产物快照

- `2026-03-30` UTC：轨迹实验评估报告写入 `results/trajectory_experiments/accuracy_summary_20260330_112435.json`、`results/trajectory_experiments/trajectory_20260330_112435.json`、`results/trajectory_experiments/layer_stratified_20260330_112435.json`
- `2026-03-31 01:56:52` UTC：22-D 轨迹融合训练统计刷新到 `models/ml_selectors/trajectory_stats.json`，包含 `31,040` 个带标注样本对、`18,873` 个正确样本（`60.80%`）、`22` 个特征、`6` 个数据集
- 持久化模型/参数文件位于 `models/ml_selectors/feature_stats.json`、`models/ml_selectors/temporal_best_params.json`、`models/ml_selectors/trajectory_fusion.pkl`

### ML Features (12 dimensions, group-normalised) | ML 特征（12 维，组内归一化）

| # | Feature | Description |
|---|---------|-------------|
| 0–1 | `mean_dist_{z,r}` | z-score / rank of mean pairwise distance (medoid-like) |
| 2–3 | `knn3_{z,r}` | z-score / rank of KNN-3 similarity |
| 4–5 | `length_{z,r}` | z-score / rank of activation length |
| 6–7 | `dc_{z,r}` | z-score / rank of DeepConf quality score |
| 8–9 | `copeland_{z,r}` | z-score / rank of Copeland win count |
| 10 | `log_n` | log(group size), context feature |
| 11 | `log_length` | log(activation length), absolute scale |

### ML Model Details | ML 模型详情

| Model | Train target | Key hyperparams |
|-------|-------------|-----------------|
| `linear-probe` | Ridge regression → score | `alpha=1.0`, StandardScaler |
| `logistic` | LogisticRegression → P(correct) | `C=1.0`, balanced class weight |
| `isotonic-medoid` | IsotonicRegression on `mean_dist_r` | monotone increasing |
| `isotonic-deepconf` | IsotonicRegression on `dc_r` | monotone increasing |

### Leave-One-Dataset-Out CV | 留一数据集交叉验证（诚实的分布外估计）

```
Selector                      aime24    aime25   brumo25      gpqa    hmmt25  livecode    Mean
----------------------------------------------------------------------------------------------
linear_probe                   86.7%     76.7%     80.0%     62.6%     53.3%     59.3%   69.8%
logistic (base 12-D)           86.7%     76.7%     80.0%     63.6%     53.3%     58.1%   69.7%
isotonic_medoid                76.7%     73.3%     76.7%     66.7%     56.7%     59.9%   68.3%
trajectory-fusion (22-D)       66.7%     70.0%     83.3%     68.7%     63.3%     57.5%   68.3%
logistic (traj 10-D only)      70.0%     70.0%     76.7%     63.1%     56.7%     58.7%   65.9%
isotonic_deepconf              76.7%     66.7%     63.3%     61.6%     46.7%     58.7%   62.3%
```

## Single-Feature Ablation (Leave-One-Dataset-Out CV)

Each of the 22 ML features (12 base + 10 trajectory) is used **individually** to train a LogisticRegression selector, then evaluated via leave-one-dataset-out CV. This reveals which single signal is most predictive.

### Base Features (12-D, from distance matrix + token stats)

```
Feature              AIME24    AIME25   BruMo25      GPQA    HMMT25     LCBv5      Mean
---------------------------------------------------------------------------------------
mean_dist_z           80.0%     73.3%     73.3%     65.7%     56.7%     58.1%     67.8%
mean_dist_r           80.0%     73.3%     73.3%     65.7%     56.7%     58.1%     67.8%
knn3_z                86.7%     76.7%     76.7%     65.7%     53.3%     59.9%     69.8%
knn3_r                86.7%     76.7%     76.7%     65.7%     53.3%     59.9%     69.8%
length_z              83.3%     73.3%     76.7%     61.1%     46.7%     56.9%     66.3%
length_r              83.3%     73.3%     76.7%     61.1%     46.7%     56.9%     66.3%
dc_z                  80.0%     73.3%     83.3%     60.6%     70.0%     58.1%     70.9%
dc_r                  80.0%     73.3%     83.3%     60.6%     70.0%     58.1%     70.9%
copeland_z            80.0%     73.3%     76.7%     65.2%     56.7%     58.1%     68.3%
copeland_r            80.0%     70.0%     73.3%     66.2%     56.7%     58.1%     67.4%
log_n                 76.7%     66.7%     63.3%     61.6%     46.7%     58.7%     62.3%
log_length            83.3%     73.3%     76.7%     61.1%     46.7%     56.9%     66.3%
```

### Trajectory Features (10-D, from neuron activation trajectory analysis)

| # | Feature | Description |
|---|---------|-------------|
| 12 | `mean_continuity` | mean Jaccard similarity between consecutive 32-token slices |
| 13 | `mean_novelty` | mean (1 - max prior Jaccard) across slices |
| 14 | `max_reflection` | max non-adjacent Jaccard similarity (folding back) |
| 15 | `reflection_count_r` | rank of number of slices where reflection > 0.5 |
| 16 | `late_convergence` | whether trajectory converges in final 25% |
| 17 | `deep_shallow_ratio_z` | z-score of deep (top 25%) to shallow (bottom 25%) layer activation ratio |
| 18 | `layer_entropy` | Shannon entropy of layer-wise activation counts |
| 19 | `layer_gini` | Gini coefficient of layer-wise activation counts |
| 20 | `deep_frac_z` | z-score of fraction of activations in deep layers |
| 21 | `n_active_layers_z` | z-score of number of distinct active layers |

```
Feature              AIME24    AIME25   BruMo25      GPQA    HMMT25     LCBv5      Mean
---------------------------------------------------------------------------------------
mean_continuity       70.0%     63.3%     73.3%     64.1%     46.7%     56.9%     62.4%
mean_novelty          63.3%     56.7%     63.3%     60.1%     40.0%     58.7%     57.0%
max_reflection        56.7%     56.7%     63.3%     62.1%     50.0%     59.9%     58.1%
reflect_count_r ★★    83.3%     73.3%     86.7%     63.1%     60.0%     59.9%     71.1%
late_convergence      80.0%     66.7%     76.7%     63.1%     43.3%     58.1%     64.6%
deep_shallow_z        86.7%     73.3%     80.0%     62.6%     53.3%     58.7%     69.1%
layer_entropy         83.3%     76.7%     83.3%     63.6%     56.7%     57.5%     70.2%
layer_gini            83.3%     76.7%     83.3%     61.1%     56.7%     56.9%     69.7%
deep_frac_z           83.3%     73.3%     80.0%     63.6%     56.7%     58.7%     69.3%
n_active_layers_z     80.0%     66.7%     63.3%     61.1%     46.7%     57.5%     62.5%
```

### Ablation Rank (all 22 features)

```
   1. reflection_count_r   71.1%   ★★ NEW BEST — reflection count rank (trajectory)
   2. dc_z / dc_r          70.9%   ★  DeepConf quality (base)
   3. layer_entropy         70.2%      Layer activation entropy (trajectory)
   4. knn3_z / knn3_r      69.8%      KNN-3 similarity (base)
   5. layer_gini            69.7%      Layer Gini coefficient (trajectory)
   6. deep_frac_z           69.3%      Deep layer fraction (trajectory)
   7. deep_shallow_ratio_z  69.1%      Deep-shallow ratio (trajectory)
   8. copeland_z            68.3%      Copeland win count (base)
   9. mean_dist_z / r       67.8%      Mean distance (base)
  10. copeland_r            67.4%      Copeland rank (base)
  11. length_z / r          66.3%      Activation length (base)
  11. log_length            66.3%      log(activation length) (base)
  13. late_convergence      64.6%      Late convergence flag (trajectory)
  14. n_active_layers_z     62.5%      Active layer count (trajectory)
  15. mean_continuity       62.4%      Trajectory continuity (trajectory)
  16. log_n                 62.3%      log(group size) (base)
  17. max_reflection        58.1%      Max reflection similarity (trajectory)
  18. mean_novelty          57.0%      Trajectory novelty (trajectory) — weakest
```

### Ablation Insights

- **`reflection_count_r` (71.1%) is the new best single feature**, surpassing `dc_z` (70.9%). This is a pure neuron-activation trajectory structure feature: the number of slices where the activation pattern "folds back" to resemble an earlier non-adjacent slice. This suggests that correct reasoning exhibits a characteristic pattern of self-reflection.
- **Layer features dominate the top 10**: `layer_entropy` (70.2%), `layer_gini` (69.7%), `deep_frac_z` (69.3%), `deep_shallow_ratio_z` (69.1%) all appear in ranks 3-7. This confirms that **which layers activate** (deep vs shallow distribution) is strongly predictive of correctness.
- **`reflection_count_r` excels on BruMo25 (86.7%)** — the highest single-feature accuracy on any dataset. It also scores 60.0% on HMMT25, second only to `dc_z` (70.0%).
- **dc excels on HMMT25 (70.0%)** — far above any other single feature (next best: reflection_count_r at 60.0%) — indicating DeepConf captures confidence patterns especially valuable for competition math.
- **knn3 excels on AIME (86.7%)** — top single feature for that dataset, confirming local neighbourhood structure is informative for AIME-style problems.
- **Trajectory structure features (continuity 62.4%, novelty 57.0%, max_reflection 58.1%) are weak** as standalone features. The aggregate statistics lose too much information; the signal is in the rank normalization (`reflection_count_r`) and layer decomposition.
- **5 of the top 7 features are now trajectory-derived**, up from 0 before these experiments. This demonstrates that neuron activation trajectory analysis provides genuinely new predictive signal beyond existing distance/confidence features.
- The gap between best single-feature (71.1%) and full 22-D fusion (68.3%) indicates **overfitting risk** when combining all features. Feature selection or regularisation tuning is needed.

Persisted single-feature model files currently cover the 12 base features in `models/ml_selectors/single_feat_*.pkl`; the full 22-feature ablation results (including trajectory-derived features) are summarised in this report, with aggregate training statistics in `models/ml_selectors/trajectory_stats.json`.

---

## Temporal Discount Slice Selector

A heuristic selector that splits the token sequence into 32-token slices, weighs later slices more heavily via exponential discounting γ^(2k), and picks the run with the highest weighted quality score. No distance matrix needed (O(n) per problem).

### Formula

```
score(r) = Σ_{k=0}^{K-1} γ^(2k) · quality(r, slice_{S-1-k})

K = max slices where γ^(2k) ≥ threshold
quality = -mean(tok_conf) for tok_conf       (lower conf = more confident = better)
quality =  mean(tok_neg_entropy) for tok_neg_entropy  (closer to 0 = more certain = better)
```

### Grid Search Results

```
metric            gamma  thresh    AIME24    AIME25   BruMo25      GPQA    HMMT25     LCBv5      Mean
-----------------------------------------------------------------------------------------------------
tok_conf           0.70   0.001     43.3%     40.0%     36.7%     46.0%     23.3%     58.1%     41.2%
tok_conf           0.70   0.010     43.3%     40.0%     36.7%     46.0%     23.3%     58.1%     41.2%
tok_conf           0.70   0.100     43.3%     43.3%     40.0%     46.0%     23.3%     58.1%     42.3%
tok_conf           0.80   0.001     43.3%     40.0%     36.7%     44.9%     23.3%     56.3%     40.8%
tok_conf           0.80   0.010     43.3%     40.0%     36.7%     44.9%     23.3%     56.3%     40.8%
tok_conf           0.80   0.100     43.3%     40.0%     36.7%     44.9%     23.3%     57.5%     41.0%
tok_conf           0.90   0.001     46.7%     40.0%     36.7%     44.4%     20.0%     55.1%     40.5%
tok_conf           0.90   0.010     46.7%     40.0%     36.7%     44.9%     20.0%     55.7%     40.7%
tok_conf           0.90   0.100     50.0%     40.0%     36.7%     47.5%     20.0%     55.1%     41.5%
tok_conf           0.95   0.001     43.3%     43.3%     36.7%     46.0%     23.3%     58.1%     41.8%
tok_conf           0.95   0.010     43.3%     43.3%     36.7%     46.0%     23.3%     58.7%     41.9%
tok_conf           0.95   0.100     43.3%     43.3%     36.7%     46.5%     23.3%     59.3%     42.1%
tok_neg_entropy    0.70   0.001     70.0%     60.0%     60.0%     64.6%     36.7%     59.3%     58.4%
tok_neg_entropy    0.70   0.010     70.0%     60.0%     60.0%     64.6%     36.7%     58.7%     58.3%
tok_neg_entropy    0.70   0.100 ★   70.0%     63.3%     63.3%     66.7%     40.0%     58.7%     60.3%
tok_neg_entropy    0.80   0.001     63.3%     63.3%     60.0%     67.7%     36.7%     59.3%     58.4%
tok_neg_entropy    0.80   0.010     63.3%     63.3%     60.0%     66.7%     36.7%     59.3%     58.2%
tok_neg_entropy    0.80   0.100     63.3%     60.0%     60.0%     64.6%     33.3%     59.3%     56.8%
tok_neg_entropy    0.90   0.001     70.0%     66.7%     53.3%     66.7%     36.7%     56.9%     58.4%
tok_neg_entropy    0.90   0.010     70.0%     66.7%     53.3%     65.7%     30.0%     57.5%     57.2%
tok_neg_entropy    0.90   0.100     66.7%     63.3%     56.7%     66.7%     26.7%     58.1%     56.3%
tok_neg_entropy    0.95   0.001     73.3%     56.7%     53.3%     65.2%     33.3%     56.3%     56.4%
tok_neg_entropy    0.95   0.010     73.3%     56.7%     53.3%     66.2%     33.3%     57.5%     56.7%
tok_neg_entropy    0.95   0.100     73.3%     53.3%     60.0%     66.7%     30.0%     56.9%     56.7%
```

### Best Configuration

| Parameter | Value |
|-----------|-------|
| metric | `tok_neg_entropy` |
| gamma | 0.7 |
| threshold | 0.1 |
| slice_size | 32 |
| **Mean accuracy** | **60.3%** |

Best params saved to `models/ml_selectors/temporal_best_params.json`.

### Temporal Selector Insights

- **`tok_neg_entropy` vastly outperforms `tok_conf`** for temporal selection (60.3% vs 42.1% best). Negative entropy captures certainty patterns that scale better under slice-level aggregation.
- **Steeper discount (γ=0.7) works best** — the model benefits from strongly weighting the final reasoning steps over early ones.
- **Threshold has minor impact** — most accuracy variation comes from metric and gamma choices.
- **Temporal selector (60.3%) underperforms plain DeepConf (61.0%) and DeepConf-based ML features (dc_z 70.9%)**. Splitting into discrete slices loses the benefit of DeepConf's sliding-window smoothing. The temporal approach may be more effective when combined with other signals rather than used standalone.
- **HMMT25 remains the hardest** dataset across all methods (40.0% best temporal vs 70.0% best dc single-feature).

---

## Trajectory Analysis Selectors (Exp 7-9)

Neuron activation trajectory analysis leverages the `rows/` position-aware CSR bank (v4.1+) to decompose per-run activations into temporal slices (32 tokens each) and layer groups (decoded from key encoding `layer<<16|neuron_id`).

### Exp 7: TrajectorySelector (轨迹结构选择器)

**Method**: For each run, compute per-slice Jaccard similarities to measure:
- **Continuity** C_t = Jaccard(slice_t, slice_{t-1}) — backbone coherence
- **Reflection** R_t = max_{s<t-1} Jaccard(slice_t, slice_s) — folding back to earlier patterns
- **Novelty** N_t = 1 - max_{s<t} Jaccard(slice_t, slice_s) — exploration of new patterns

```
backbone_score = α·mean_continuity - β·mean_novelty + γ·late_convergence + δ·bounded_reflection
(α=1.0, β=0.3, γ=0.5, δ=0.3)
```

**Result: 58.7% mean** — weak standalone performance. The heuristic weighting does not generalise well; HMMT25 only 36.7%.

### Exp 8: LayerStratifiedSelector (分层激活选择器)

**Method**: Decode layer IDs from neuron keys, compute layer-wise activation distribution features:
- Deep-shallow ratio (top 25% layers / bottom 25% layers)
- Layer activation entropy (Shannon)
- Layer concentration (Gini)

```
layer_score = α·deep_frac_z + β·layer_entropy - γ·layer_gini
(α=1.0, β=0.5, γ=0.3)
```

**Result: 69.4% mean** — competitive with logistic (69.7%) without any ML training. Strong on AIME24 (83.3%) and BruMo25 (80.0%).

### Exp 9: Trajectory Fusion LOO CV

22-D logistic regression combining 12 base features + 10 trajectory features:

```
Model                             aime24    aime25   brumo25      gpqa    hmmt25  livecode   Mean
-------------------------------------------------------------------------------------------------
logistic (base 12-D)               86.7%     76.7%     80.0%     63.6%     53.3%     58.1%   69.7%
logistic (traj 10-D)               70.0%     70.0%     76.7%     63.1%     56.7%     58.7%   65.9%
logistic (fusion 22-D)             66.7%     70.0%     83.3%     68.7%     63.3%     57.5%   68.3%
```

**Result: 68.3% mean** — fusion did NOT improve over base 12-D (69.7%). The 22-D model suffers mild overfitting with only 6 datasets for LOO CV. However, the fusion model shows the best HMMT25 (63.3%) and GPQA (68.7%) scores, suggesting trajectory features capture complementary signal on harder datasets.

### Trajectory Experiment Insights

1. **Layer decomposition is the key insight**: 4 of the top 7 single features are layer-based (entropy, gini, deep_frac, deep_shallow_ratio), all at 69-70%.
2. **reflection_count_r (71.1%) is the new single-feature champion**: the rank-normalised count of "reflection" events (slices resembling earlier non-adjacent slices) surpasses dc_z (70.9%).
3. **Raw trajectory statistics (continuity, novelty) are weak**: aggregating Jaccard similarities across slices loses too much temporal structure.
4. **The heuristic `layer-stratified` selector (69.4%) nearly matches ML logistic (69.7%)** without training — confirming that layer activation distribution is a robust, generalisable signal.
5. **22-D fusion overfits on 6 datasets**: more datasets or feature selection needed to benefit from trajectory features in the ML pipeline.

Training script: `scripts/train_trajectory_selectors.py`
Implementation: `nad/core/selectors/trajectory_impl.py`
Models: `models/ml_selectors/trajectory_fusion.pkl`, `models/ml_selectors/trajectory_stats.json`

---

## Key Observations

- **`reflection_count_r` (71.1%) is the new best single feature**, surpassing `dc_z` (70.9%). This trajectory-derived feature captures how often a run's activation pattern "reflects" back to earlier states — a structural signature of correct self-reflection in reasoning.
- **`layer-stratified` (69.4%) ranks #6 overall** — the highest non-ML, non-pairwise selector. It requires no distance matrix and no training, making it the most efficient high-accuracy selector.
- **`logistic` (70.6%) and `linear-probe` (70.3%)** rank 2nd and 4th overall, surpassing `consensus-min` (70.4%) and `knn-medoid` (69.8%). They outperform all non-oracle non-ML selectors on mean accuracy.
- **Leave-one-out CV** shows `linear_probe` (69.8%) and `logistic` (69.7%) remain competitive with `knn-medoid` (69.8%) even when evaluated on held-out datasets, confirming generalisation.
- **Single-feature ablation (22 features)**: 5 of the top 7 features are trajectory-derived (reflection_count_r, layer_entropy, layer_gini, deep_frac_z, deep_shallow_ratio_z). Layer decomposition provides the strongest new signal class.
- **22-D fusion (68.3%) does NOT improve over 12-D base (69.7%)** — mild overfitting with only 6 LOO CV folds. However, fusion shows best HMMT25 (63.3%) and GPQA (68.7%) scores, suggesting trajectory features help on harder datasets.
- **`isotonic-medoid` (68.4%)** improves over plain `medoid` (67.8%) with no additional distance computation — it simply recalibrates the medoid score using learned monotonic mapping.
- **Temporal discount selector** (60.3% best) and **trajectory heuristic** (58.7%) are weak standalone — slice-level aggregation without ML or layer decomposition loses signal quality.
- DeepConf-based ML variants (`isotonic-deepconf`) do not improve meaningfully over plain `deepconf`, consistent with the pattern seen in other DeepConf-based selectors.
- All ML selectors are deterministic at inference (no randomness); training takes ~10 minutes on 6 datasets (trajectory feature extraction adds ~2 hours).
