# Hypothesis-Driven Structural Selector Analysis

Date: `2026-04-05`

## Scope

This note consolidates the finished work from:

- `result/all_model_20260404_155232/summary.json`
- `result/position_ablation_20260404_155548/position_ablation_summary.json`
- `results/selector_rankings/20260404/selector_rankings_20260404_155549.json`
- `results/selector_rankings/20260404/selector_rankings_20260404_160200.json`
- `result/structural_baselines/*/structural_accuracy.json`
- `results/extreme8_experiments/20260402_112323/summary_20260402_112323.json`
- `results/reflection_dynamics/threshold_sweep_summary.json`
- `results/scans/lcb/lcb_knn_scan.json`
- `docs/EXTREME12_TEST_ANALYSIS.md`
- `docs/EXTREME12_V2_EXPERIMENT.md`

I also generated a lightweight position-only ranking refresh at:

- `result/tmp_rank_positions/selector_rankings_20260405_023040.json`

Important status note:

- `result/cross_domain_eval/` exists but is still effectively incomplete; its dataset folders are present but empty.
- A background `nad.cli analyze` process for cross-domain evaluation was still running when this note was written, so the conclusions below do **not** rely on cross-domain matrix results.

## Executive Summary

The finished experiments already give a fairly clear direction:

1. **H1 is supported**: the best selector family is not stable across domains, especially once programming is isolated.
2. **H2 is supported**: “coding CoT is shorter” is not a strong primary feature by itself; structure and confidence shape matter more than raw length.
3. **H3 is only weakly supported**: coding is somewhat prefix-friendly, but the current evidence is not strong enough to claim a clean “code = prefix, math/science = full trajectory” split.
4. **H4 is partially supported**: dynamic / segmented confidence features look useful, but graph-topology features do **not** currently justify a full Extreme10 push.

The highest-ROI next step is **not** more visualization and **not** graph-heavy expansion. It is:

- keep a **math/science-oriented structural model**,
- treat **coding as a separate modeling problem**,
- prioritize **prefix / tail / reflection / local-stability features**,
- deprioritize **graph topology** until there is stronger evidence.

## Evidence by Hypothesis

### H1 — Math/science and coding should split

### Verdict

**Supported, with moderate confidence.**

### Evidence

- In `results/selector_rankings/20260404/selector_rankings_20260404_160200.json`, the best non-oracle selector differs by category:
  - `global`: `tournament-copeland` is the best implementable selector (`micro_accuracy ≈ 67.8%`)
  - `programming`: `min-confidence` ranks first (`micro_accuracy ≈ 59.9%`)
  - `math`: `tournament-copeland` ranks first among implementable selectors (`micro_accuracy ≈ 70.0%`)
  - `science`: `consensus-max` / `tournament-copeland` / `medoid` cluster near the top, not `min-confidence`
- Structural baselines also show different behavior:
  - `livecodebench_v5`: `knn-medoid` and `min-confidence` tie at `59.88%`
  - math-heavy datasets are usually led by `knn-medoid`, `medoid`, or `dbscan-medoid`
  - `graph-degree` only ties for first on `hmmt25`, and otherwise does not dominate
- Extreme8 → Extreme9 training artifacts suggest domain-specific feature value:
  - `extreme8_selacc_global_stats.json`: inner-tune baseline AUROC `67.65%`
  - `extreme9_selacc_global_stats.json`: inner-tune baseline AUROC `71.42%`
  - `extreme8_math_stats.json`: `75.31%`
  - `extreme9_math_stats.json`: `81.41%`
  - `extreme8_science_stats.json`: `55.15%`
  - `extreme9_science_stats.json`: `57.69%`
  - `extreme8_code_stats.json`: `52.31%`
  - `extreme9_code_stats.json`: `42.65%`

### Interpretation

The same extra local-confidence features that help global/math/science do **not** transfer cleanly to coding. That is strong evidence against a single unified feature weighting.

The caution is that current programming rankings are based on **one programming dataset** (`livecodebench_v5`), so the split should be treated as “supported but not fully generalized.”

## H2 — “Coding CoT is shorter” is a distribution difference, not a core feature

### Verdict

**Supported.**

### Evidence

- Global position-ablation averages in `position_ablation_summary.json` show that major pairwise selectors generally improve with more context:
  - `medoid`: `64.19 → 68.50` from `0-1` to `0-512`
  - `knn-medoid`: `62.75 → 68.16`
  - `dbscan-medoid`: `66.04 → 69.16`
  - `tournament-copeland`: `67.64 → 68.25`
  - `consensus-min`: `63.05 → 67.61`
  - `consensus-max`: `64.38 → 68.89`
- `deepconf` and `min-confidence` are flat across windows in the saved summary, which means raw token-confidence aggregates do not gain from longer windows in the same way as pairwise structure-based selectors.
- On `livecodebench_v5`, short windows are competitive but not clearly superior:
  - `full_sequence` best: `tournament-copeland 62.28%`
  - `window_0-2` best: `dbscan-medoid 61.68%`
  - `window_0-8` best: `ensemble-medoid 61.68%`
  - `window_0-1` best: `min-confidence 59.88%`

### Interpretation

Coding may expose usable signal earlier, but the data do **not** support using raw “shorter is better” as a core modeling rule. The safer framing is:

- length is a **control variable**,
- prefix behavior is a **structure variable**,
- the useful signal is more likely to be **how quickly confidence stabilizes**, not raw output length itself.

## H3 — Coding is prefix-dominant; math/science rely on full trajectory

### Verdict

**Only weakly supported / not yet proven.**

### Evidence

- `livecodebench_v5` is relatively prefix-friendly:
  - best short-prefix result is `61.68%`
  - best full-sequence result is `62.28%`
  - gap is only `0.60pp`
- `gpqa` also nearly saturates early:
  - full best `66.16%`
  - best short-prefix `65.66%`
  - gap is only `0.51pp`
- For small math datasets, the best short-window selector sometimes matches or slightly beats full-sequence, but these datasets are small enough that selector-level noise is still substantial:
  - `aime24`: gap `0.00pp`
  - `aime25`: short prefix beats full by `3.33pp`
  - `brumo25`: full beats short by `3.33pp`
  - `hmmt25`: short prefix beats full by `3.33pp`
- The refreshed position-only category report (`result/tmp_rank_positions/selector_rankings_20260405_023040.json`) warns that there is **insufficient data** to compute stable per-window `programming` or `science` sub-rankings beyond a single dataset’s window tasks.

### Interpretation

The current evidence says:

- coding does **not** need the whole sequence to become competitive,
- but science may also saturate early,
- and math is not cleanly “full-trajectory dependent” in the saved quick-window results.

So H3 should be downgraded from “working conclusion” to “still-open hypothesis.”

Operationally:

- it is reasonable to design **prefix-aware coding features**,
- but it is **not yet justified** to hard-code a strong “code only needs prefix / math needs full path” claim.

## H4 — Dynamic / topology / segmented stability features matter more than raw neuron membership

### Verdict

**Partially supported: dynamic and segmented-stability yes; graph topology no.**

### Evidence for dynamic / segmented-stability

- Reflection dynamics are real and measurable:
  - `results/reflection_dynamics/threshold_sweep_summary.json` shows best reflection threshold at `0.20`
  - best single-feature LOO mean improves to `71.7%`
- The existing docs already identified `reflection_count_r` as one of the strongest single features.
- Extreme8 blind evaluation remains strong:
  - `results/extreme8_experiments/20260402_112323/summary_20260402_112323.json`
  - `best_only` mean accuracy `≈ 72.52%`
  - same mean for `best_plus_worst` and `worst_avoid`, which implies the useful signal is concentrated in the “best” direction
- Extreme9 helps on global/math/science inner-tune AUROC, which is consistent with the idea that **tail quality / local worst-window / post-reflection stability** add useful information beyond the original three Extreme8 features.

### Evidence against graph topology as current priority

- Structural-baseline mean accuracy across datasets:
  - `knn-medoid`: `69.81%`
  - `dbscan-medoid`: `67.93%`
  - `medoid`: `67.85%`
  - `graph-degree`: `65.04%`
  - `deepconf`: `60.99%`
  - `local-conf-tail`: `43.03%`
- Dataset wins:
  - `knn-medoid`: 4 wins/ties
  - `dbscan-medoid`: 2 wins/ties
  - `graph-degree`: only 1 tie (`hmmt25`)
- `results/scans/lcb/lcb_knn_scan.json` reinforces this on coding:
  - `graph_deg25`: `SelAcc@10 = 60.49%`, but `pairwise = 48.68%`
  - `graph_deg30`: `SelAcc@10 = 60.39%`, `pairwise = 48.36%`
  - `graph_deg40`: `SelAcc@10 = 59.83%`, `pairwise = 48.46%`
  - graph variants can boost top-band selection a bit, but they degrade overall ordering quality toward random

### Interpretation

The useful “structural” signal currently looks like:

- reflection count / reflection timing,
- local confidence minima,
- head-vs-tail stability,
- tail variance / post-event recovery,
- prefix saturation.

The less useful signal, given current evidence, is:

- graph centrality / local clustering as a primary decision rule.

This is exactly why skipping Extreme10 after Phase 3 was the correct move.

## Additional Findings

### Aggregated SelAcc@10 objective is not yet better than the simpler baseline

From `docs/EXTREME12_V2_EXPERIMENT.md`:

- `baseline12_pointwise` remains the recommended configuration
- aggregated tuning can improve one metric while breaking guardrails on another
- inner-tune SelAcc overfits badly relative to validation

This means:

- the objective **matters**,
- but the current aggregated-search setup is still too unstable to replace the baseline.

### The main failure mode is still large-domain robustness

From `docs/EXTREME12_TEST_ANALYSIS.md`:

- the biggest drag is still `gpqa + lcb_v5`
- they account for roughly `75%` of test-side samples in that analysis
- baseline12-pointwise is close to random on those hard domains

This remains the core product problem:

- small math datasets are already fairly strong,
- large science/coding datasets are where model capacity is being lost.

## Recommended Next Moves

### 1. Treat code as a separate feature-design problem now

Do **not** assume that the math/global Extreme9 recipe transfers to code.

The training artifacts suggest the opposite.

### 2. Keep pushing local tail / reflection / saturation features

The highest-ROI family now is:

- prefix best-window quality,
- head-vs-tail settle gap,
- reflection density,
- tail variance,
- post-reflection recovery.

These are much more justified than raw length.

### 3. Do not spend more cycles on graph topology yet

Graph features have not earned their complexity.

If graph terms return later, they should come back as small auxiliary terms, not as the main next hypothesis.

### 4. Use cross-domain evaluation as a decision gate, not as a fishing expedition

Once `cross_domain_eval` actually finishes, the key question is not “which model is best on average?” but:

- does a math/global model transfer into code?
- does a code-tuned model transfer back into math/science?
- does specialization buy real in-domain gain that survives cross-domain drop?

### 5. If you want one concrete implementation target next

The best next implementation target is a **code-oriented prefix/tail selector** rather than Extreme10.

The existing `prefix_saturation` direction is aligned with the evidence in this note.

## Bottom Line

If I had to collapse everything into one actionable statement:

> The data now justify splitting off coding, focusing on prefix/tail/reflection stability, and explicitly **not** prioritizing graph-topology expansion.

That is the shortest path that is still strongly supported by the experiments already on disk.
