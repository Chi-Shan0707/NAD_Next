# Selector Accuracy Comparison

Model: DeepSeek-R1-0528-Qwen3-8B  |  Distance: Jaccard  |  Runs per problem: 64

```
Selector                  AIME24    AIME25   BruMo25      GPQA    HMMT25     LCBv5      Mean
--------------------------------------------------------------------------------------------
min-activation             83.3%     73.3%     76.7%     61.1%     46.7%     56.9%     66.3%
max-activation             50.0%     56.7%     56.7%     53.5%     43.3%     59.9%     53.3%
medoid                     80.0%     73.3%     73.3%     65.7%     56.7%     58.1%     67.8%
knn-medoid                 86.7%     76.7%     76.7%     65.7%     53.3%     59.9%     69.8%
dbscan-medoid              80.0%     73.3%     73.3%     66.2%     56.7%     58.1%     67.9%
consensus-min              86.7%     73.3%     80.0%     66.7%     56.7%     59.3%     70.4%
consensus-max              80.0%     76.7%     70.0%     65.7%     53.3%     58.7%     67.4%
deepconf                   70.0%     70.0%     60.0%     60.6%     46.7%     58.7%     61.0%
avg64@                     75.3%     65.7%     69.4%     60.2%     48.6%     58.7%     63.0%
con64@                     80.0%     76.7%     80.0%     70.2%     66.7%     58.1%     71.9%
```

## Rank by Mean Accuracy

```
   1. con64@                   71.9%
   2. consensus-min            70.4%
   3. knn-medoid               69.8%
   4. dbscan-medoid            67.9%
   5. medoid                   67.8%
   6. consensus-max            67.4%
   7. min-activation           66.3%
   8. avg64@                   63.0%
   9. deepconf                 61.0%
  10. max-activation           53.3%
```

## Notes

- `avg64@` / `con64@` are oracle baselines using all 64 runs' correctness labels.
- `min-activation`, `max-activation`, `deepconf` score each run independently (no pairwise distance needed).
- Other selectors require the full N×N Jaccard distance matrix.
