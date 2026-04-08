# GPQA RankSVM / Math SVM Status — 2026-04-08 01:00 UTC

## Report Meta

- Report date-time: `2026-04-08 01:00 UTC`
- Time precision: hour-level
- Scope:
  - GPQA `RankSVM`
  - GPQA `RankSVM -> hybrid`
  - math-domain `LinearSVC / RankSVM`
- Report purpose:
  - preserve the already-produced progress summary as a dated experiment note
  - mark the initial situation, the core operations, and the current conclusions

## Initial State

- GPQA initial state:
  - pure `RankSVM` had already been implemented and verified as a narrow GPQA-only baseline
  - best pure GPQA `RankSVM` improved ranking quality, but still failed the top-slot gate
  - an earlier one-off SVM-backed shortlist probe had already shown that the passing point matched the existing `science_hybrid_round3` winner rather than surpassing it
- Math initial state:
  - there was no dedicated math SVM sweep script in the repo
  - math still relied mainly on the generic `ml_features` line and frozen structural baselines such as `knn-medoid`
  - the user explicitly requested extending SVM exploration into math and not limiting the search to previously defined features

## Time-Stamped Operation Log

- `2026-04-07 15:00 UTC`
  - initial operation:
    - run and save the narrow GPQA pure `RankSVM` round-1 baseline
  - core content:
    - validate whether pairwise hinge can improve GPQA science selection without changing coding
  - artifact:
    - `result/gpqa_ranksvm_round1_20260407.json`

- `2026-04-07 18:00 UTC`
  - initial operation:
    - complete the broad pure GPQA `RankSVM` sweep
  - core content:
    - search loss / `C` / backend / feature-family combinations for the pure linear SVM line
  - artifact:
    - `result/gpqa_ranksvm_sweep_20260407_full/gpqa_ranksvm_sweep.json`

- `2026-04-07 19:00 UTC`
  - initial operation:
    - run a GPQA `RankSVM`-backed shortlist-hybrid probe
  - core content:
    - test whether SVM margins can repair GPQA `Hit@1` once converted into shortlist decisions
  - artifact:
    - `result/gpqa_ranksvm_hybrid_probe_20260407/gpqa_ranksvm_hybrid_probe.json`

- `2026-04-08 00:00 UTC`
  - initial operation:
    - add reusable math SVM infra and a reusable GPQA `RankSVM` hybrid sweep script
  - core content:
    - create:
      - `nad/core/selectors/math_svm_impl.py`
      - `scripts/run_math_svm_sweep.py`
      - `scripts/run_gpqa_ranksvm_hybrid_sweep.py`
    - extend math search space beyond the original 12 features by adding derived consensus / disagreement / interaction features

- `2026-04-08 00:00 UTC`
  - initial operation:
    - finish an extra runwise math-only sweep on the missing feature families
  - core content:
    - verify whether the best math SVM sits outside the first coarse feature-family subset
  - artifact:
    - `result/math_svm_sweep_train_20260408_runwise_extra1/math_svm_sweep.json`

- `2026-04-08 01:00 UTC`
  - initial operation:
    - continue the coarse math sweep and the extra math `RankSVM` sweep
  - core content:
    - compare:
      - runwise `LinearSVC`
      - pairwise `RankSVM`
      - original 12 features
      - augmented math features
  - artifacts:
    - `result/math_svm_sweep_train_20260408_coarse1/math_svm_sweep.partial.json`
    - `result/math_svm_sweep_train_20260408_ranksvm_extra1/math_svm_sweep.partial.json`

- `2026-04-08 01:00 UTC`
  - initial operation:
    - restart the reusable GPQA `RankSVM` hybrid sweep after fixing the gate-score bug
  - core content:
    - fix `baseline_gate_scores` to use `recency_conf_mean`
    - add `precomputed.pkl` caching so the slow GPQA extraction is not wasted on reruns
  - artifact:
    - `result/gpqa_ranksvm_hybrid_sweep_20260408_focus2/run.log`

## GPQA

- Pure `RankSVM` implementation is in:
  - `nad/core/selectors/gpqa_ranksvm_impl.py`
  - `scripts/run_gpqa_ranksvm_round1.py`
  - `scripts/run_gpqa_ranksvm_sweep.py`
- Confirmed pure-GPQA result:
  - best pure `RankSVM` reaches roughly:
    - `Hit@1 = 63.64%`
    - `SelAcc@10% = 64.98%`
    - `Pairwise = 61.14%`
  - this improves ranking quality, but still misses the GPQA top-slot gate.
- Confirmed SVM-backed shortlist probe:
  - `Hit@1 = 68.18%`
  - passes the gate
  - but matches the existing promoted `science_hybrid_round3` operating point instead of beating it.
- New reusable sweep script:
  - `scripts/run_gpqa_ranksvm_hybrid_sweep.py`
  - supports RankSVM feature/loss/C sweeps and margin→probability mappings
  - now caches extraction to `precomputed.pkl` to avoid redoing the slow GPQA feature build

## Math

- New math SVM implementation:
  - `nad/core/selectors/math_svm_impl.py`
  - `scripts/run_math_svm_sweep.py`
- Math sweep now covers:
  - runwise `LinearSVC`
  - pairwise `RankSVM`
  - multiple feature families
  - additional derived consensus / disagreement / interaction features
- Current best completed math SVM operating points on the `train` profile:
  - best runwise:
    - `runwise__all_aug__squared_hinge__C0p10__bias__balanced`
    - `Hit@1 = 74.44%`
    - `SelAcc@10% = 100.00%`
    - `Pairwise = 73.91%`
    - `AUROC = 90.97%`
  - best RankSVM seen so far:
    - `ranksvm__distance_confidence__squared_hinge__C0p03__nobias__utility`
    - `Hit@1 = 74.44%`
    - `SelAcc@10% = 99.31%`
    - `Pairwise = 75.62%`
    - `AUROC = 78.34%`
- Important comparison:
  - baseline `knn-medoid` is still the math top-1 winner at `Hit@1 = 75.56%`
  - current best SVM variants trail by exactly one problem
  - `knn-medoid` strictly dominates the current best SVM on top-1 correctness, so there is no obvious narrow SVM-only ensemble win yet

## Core Operation Summary

- GPQA core operation:
  - keep SVM introduction narrow
  - verify the ceiling of pure linear `RankSVM`
  - then shift effort to `RankSVM -> hybrid` because GPQA’s real bottleneck is still top-slot `Hit@1`
- Math core operation:
  - build a reusable SVM experiment line instead of one-off notebooks or ad hoc probes
  - search both runwise and pairwise SVM formulations
  - explicitly expand beyond inherited features instead of staying locked to the original `ml_features` columns
- Current operational conclusion:
  - GPQA: pure SVM is likely near its ceiling; only the hybrid conversion path still has meaningful upside
  - Math: SVM is already strong on ranking quality, but has not yet surpassed `knn-medoid` on top-1

## Artifacts

- GPQA:
  - `result/gpqa_ranksvm_round1_20260407.json`
  - `result/gpqa_ranksvm_sweep_20260407_full/gpqa_ranksvm_sweep.json`
  - `result/gpqa_ranksvm_hybrid_probe_20260407/gpqa_ranksvm_hybrid_probe.json`
- Math:
  - `result/math_svm_sweep_train_20260408_coarse1/`
  - `result/math_svm_sweep_train_20260408_runwise_extra1/`
  - `result/math_svm_sweep_train_20260408_ranksvm_extra1/`

## Current recommendation

- GPQA: keep pushing only on `RankSVM -> hybrid` conversions; pure linear SVM is already close to exhausted.
- Math: SVM is useful for full-list ranking quality, but has not yet beaten `knn-medoid` on top-1; if continuing, the best next move is a very narrow hybrid against `knn-medoid`, not more blind SVM hyperparameter expansion.
