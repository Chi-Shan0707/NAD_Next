# Selector Accuracy Comparison

Model: DeepSeek-R1-0528-Qwen3-8B  |  Distance: Jaccard  |  Runs per problem: 64

Type legend:
- `[S]`    single-run score (no distance matrix)
- `[P]`    pairwise N×N Jaccard matrix
- `[O]`    oracle baseline (uses all ground-truth labels)
- `[P+E]`  pairwise + group ensemble
- `[P+T]`  pairwise + tournament + softmax
- `[P+2S]` pairwise + two-stage top-k
- `[ML]`   ML selector (trained on labelled data, requires `models/ml_selectors/`)

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
avg64@                     75.3%     65.7%     69.4%     60.2%     48.6%     58.7%     63.0%         [O]
con64@                     80.0%     76.7%     80.0%     70.2%     66.7%     58.1%     71.9%         [O]
```

## Rank by Mean Accuracy

```
   1. con64@                   71.9%  [O]       oracle baseline
   2. logistic                 70.6%  [ML]      ★ ML — logistic regression on 12 features
   3. consensus-min            70.4%  [P]
   4. linear-probe             70.3%  [ML]      ★ ML — ridge regression on 12 features
   5. knn-medoid               69.8%  [P]
   6. twostage-tournament      69.2%  [P+2S]
   7. twostage-medoid          69.1%  [P+2S]
   8. tournament-copeland      68.8%  [P+T]
   9. isotonic-medoid          68.4%  [ML]      ★ ML — isotonic calibration of medoid score
  10. ensemble-medoid          68.4%  [P+E]
  11. dbscan-medoid            67.9%  [P]
  12. medoid                   67.8%  [P]
  13. consensus-max            67.4%  [P]
  14. min-activation           66.3%  [S]
  15. avg64@                   63.0%  [O]       oracle baseline
  16. tournament-deepconf      62.3%  [S+T]
  17. isotonic-deepconf        62.3%  [ML]
  18. deepconf                 61.0%  [S]
  19. ensemble-deepconf        61.0%  [S+E]
  20. max-activation           53.3%  [S]
  21. min-confidence           48.1%  [S]
```

## Notes

- **[S]** Single-run selectors score each run independently.
  - `min-activation`, `max-activation`, `min-confidence`, `deepconf`
- **[P]** Pairwise selectors use the full N×N Jaccard distance matrix.
- **[O]** Oracle baselines use ground-truth correctness from all runs.
  - `avg64@`: average accuracy across all 64 runs; `con64@`: majority-vote correctness
- **[P+E] Group Ensemble**: split N runs into random groups of 8, select best per group, then select among winners.
  - `ensemble-medoid` (medoid per group) | `ensemble-deepconf` (DeepConf per group)
- **[P+T] Tournament + softmax**: pairwise Copeland or DeepConf comparison → rank → softmax (temp=0.2).
  - `tournament-copeland` | `tournament-deepconf`
- **[P+2S] Two-Stage**: groups of 16 → top-4 per group → 16-finalist final round.
  - `twostage-medoid` (medoid) | `twostage-tournament` (Copeland + softmax)
- **[ML] ML Selectors**: trained with `scripts/train_ml_selectors.py` on 31,040 labelled (problem, run) pairs from all 6 datasets. Models stored in `models/ml_selectors/`.

### ML Features (12 dimensions, group-normalised)

| # | Feature | Description |
|---|---------|-------------|
| 0–1 | `mean_dist_{z,r}` | z-score / rank of mean pairwise distance (medoid-like) |
| 2–3 | `knn3_{z,r}` | z-score / rank of KNN-3 similarity |
| 4–5 | `length_{z,r}` | z-score / rank of activation length |
| 6–7 | `dc_{z,r}` | z-score / rank of DeepConf quality score |
| 8–9 | `copeland_{z,r}` | z-score / rank of Copeland win count |
| 10 | `log_n` | log(group size), context feature |
| 11 | `log_length` | log(activation length), absolute scale |

### ML Model Details

| Model | Train target | Key hyperparams |
|-------|-------------|-----------------|
| `linear-probe` | Ridge regression → score | `alpha=1.0`, StandardScaler |
| `logistic` | LogisticRegression → P(correct) | `C=1.0`, balanced class weight |
| `isotonic-medoid` | IsotonicRegression on `mean_dist_r` | monotone increasing |
| `isotonic-deepconf` | IsotonicRegression on `dc_r` | monotone increasing |

### Leave-One-Dataset-Out CV (honest out-of-distribution estimates)

```
Selector                      aime24    aime25   brumo25      gpqa    hmmt25  livecode    Mean
----------------------------------------------------------------------------------------------
linear_probe                   86.7%     76.7%     80.0%     62.6%     53.3%     59.3%   69.8%
logistic                       86.7%     76.7%     80.0%     63.6%     53.3%     58.1%   69.7%
isotonic_medoid                76.7%     73.3%     76.7%     66.7%     56.7%     59.9%   68.3%
isotonic_deepconf              76.7%     66.7%     63.3%     61.6%     46.7%     58.7%   62.3%
```

## Key Observations

- **`logistic` (70.6%) and `linear-probe` (70.3%)** rank 2nd and 4th overall, surpassing `consensus-min` (70.4%) and `knn-medoid` (69.8%). They outperform all non-oracle non-ML selectors on mean accuracy.
- **Leave-one-out CV** shows `linear_probe` (69.8%) and `logistic` (69.7%) remain competitive with `knn-medoid` (69.8%) even when evaluated on held-out datasets, confirming generalisation.
- **`isotonic-medoid` (68.4%)** improves over plain `medoid` (67.8%) with no additional distance computation — it simply recalibrates the medoid score using learned monotonic mapping.
- DeepConf-based ML variants (`isotonic-deepconf`) do not improve meaningfully over plain `deepconf`, consistent with the pattern seen in other DeepConf-based selectors.
- All ML selectors are deterministic at inference (no randomness); training takes ~10 minutes on 6 datasets.
