# Selector Accuracy Comparison

Model: DeepSeek-R1-0528-Qwen3-8B  |  Distance: Jaccard  |  Runs per problem: 64

Type legend:  [S] = single-run  |  [P] = pairwise (N×N Jaccard)  |  [O] = oracle baseline  |  [E] = group ensemble  |  [T] = tournament + softmax  |  [2S] = two-stage (group top-k → final)

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
avg64@                     75.3%     65.7%     69.4%     60.2%     48.6%     58.7%     63.0%         [O]
con64@                     80.0%     76.7%     80.0%     70.2%     66.7%     58.1%     71.9%         [O]
```

## Rank by Mean Accuracy

```
   1. con64@                   71.9%  [O]
   2. consensus-min            70.4%  [P]
   3. knn-medoid               69.8%  [P]
   4. twostage-tournament      69.2%  [P+2S]  ★NEW
   5. twostage-medoid          69.1%  [P+2S]  ★NEW
   6. tournament-copeland      68.8%  [P+T]
   7. ensemble-medoid          68.4%  [P+E]
   8. dbscan-medoid            67.9%  [P]
   9. medoid                   67.8%  [P]
  10. consensus-max            67.4%  [P]
  11. min-activation           66.3%  [S]
  12. avg64@                   63.0%  [O]
  13. tournament-deepconf      62.3%  [S+T]
  14. deepconf                 61.0%  [S]
  15. ensemble-deepconf        61.0%  [S+E]
  16. max-activation           53.3%  [S]
  17. min-confidence           48.1%  [S]
```

## Notes

- **[S] Single-run selectors** score each run independently — no N×N distance computation needed.
  - `min-activation`: fewest total neuron activations
  - `max-activation`: most total neuron activations
  - `min-confidence`: lowest mean tok_conf (= most confident token distribution)
  - `deepconf`: tok_conf with min-group sliding window reduction (window=20)
- **[P] Pairwise selectors** require the full N×N Jaccard distance matrix across all runs.
- **[O] Oracle baselines** use ground-truth correctness labels from all runs:
  - `avg64@`: average accuracy across all 64 runs per problem
  - `con64@`: majority-vote answer correctness
- **[P+E] Group Ensemble** (分组淘汰): randomly split N runs into groups, select best within each group, then select best among group winners.
  - `ensemble-medoid` (group=8): medoid within each group → medoid of winners
  - `ensemble-deepconf` (group=8): DeepConf quality within each group → best of winners
- **[P+T] / [S+T] Tournament** (两两比较 + softmax): pairwise comparisons to build a ranking, then softmax sampling (temperature=0.2) to select.
  - `tournament-copeland`: Copeland voting — for pair (i,j), count how many k are closer to i vs j
  - `tournament-deepconf`: pairwise DeepConf quality comparison
- **[P+2S] Two-Stage** (分组 Top-K → 决赛): randomly split into groups of 16, select top-4 from each group, then 16 finalists compete for the winner.
  - `twostage-medoid`: group top-4 by mean distance → final medoid
  - `twostage-tournament`: group top-4 by Copeland → final Copeland + softmax

## Key Observations

- **twostage-tournament** (69.2%) and **twostage-medoid** (69.1%) rank 4th/5th, very close to knn-medoid (69.8%). The two-stage 16→4→1 structure effectively filters outliers before the final round.
- **twostage-tournament** achieves the best HMMT25 (63.3%) among all non-oracle selectors, and ties for best BruMo25 (80.0%).
- All distance-based hierarchical methods (ensemble, tournament, twostage) outperform plain medoid (67.8%), showing that staged selection adds value.
- DeepConf-based variants consistently underperform distance-based ones, indicating the DeepConf signal is a weaker discriminator than pairwise Jaccard structure.
