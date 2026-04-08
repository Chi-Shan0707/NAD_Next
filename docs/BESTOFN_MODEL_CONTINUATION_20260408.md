# Best-of-N Continuation Tracks — 2026-04-08

## 1. Current submission line stays frozen

This repo now keeps two separate tracks:

1. **Submission line**: conservative score recovery only.
2. **Research line**: continue trying `LambdaSVM`, `DeepSets`, and `SetTransformerLite` for future Best-of-N improvements.

The current submission ordering stays:

- `ds_aime25_only`
- `no_math_patch`
- `ds_qwen_aime25_brumo25`

Those candidates are documented in:

- `docs/BESTOFN_SCORE_RECOVERY_20260408.md`

and exported in:

- `submission/BestofN/extreme12/patches/`

This means:

- `code_v2` remains the current promoted coding default.
- `science_hybrid_round3` remains the current promoted science patch.
- full `math_deepsets_round1` remains a reverted default after the user-reported `Submission #92` failure.

## 2. New research families added

This round adds thin, Best-of-N-only research support for:

- `LambdaSVM`
- `DeepSets round2`
- `SetTransformerLite`

The new code is deliberately narrow:

- **no Early-Stop changes**
- **no new feature family**
- **no graph-heavy expansion**
- **no raw neuron rows**
- still only uses existing run-level structured features

Runtime note:

- if `torch` is unavailable in the active environment, the new research runners now **skip** `DeepSets` / `SetTransformerLite` explicitly and still keep the `LambdaSVM` lane runnable

### 2.1 LambdaSVM

Added:

- `nad/core/selectors/lambda_svm_core.py`
- `nad/core/selectors/math_lambdasvm_impl.py`
- `nad/core/selectors/gpqa_lambdasvm_impl.py`
- `nad/core/selectors/code_lambdasvm_impl.py`

Operational definition here:

- linear pairwise hinge model
- optional pair weighting via `pair_weight_mode`
- current supported modes:
  - `uniform`
  - `dcg_delta`

The implementation keeps the same thin scorer shape as existing SVM-style utilities:

- `fit(...)`
- `score_group(...)`
- `save(...)`
- `load(...)`

### 2.2 SetTransformerLite

Added:

- `nad/core/selectors/set_transformer_lite_core.py`
- `nad/core/selectors/math_set_transformer_lite_impl.py`
- `nad/core/selectors/gpqa_set_transformer_lite_impl.py`
- `nad/core/selectors/code_set_transformer_lite_impl.py`

This is intentionally a **small** set model:

- single self-attention block
- small head count
- existing run-level structured features only
- same training loop style as the current DeepSets line

It is not a broad transformer-platform expansion.

## 3. New research runners

Added:

- `scripts/run_math_bestofn_research_round2.py`
- `scripts/run_gpqa_bestofn_research_round2.py`
- `scripts/run_code_bestofn_research_round2.py`

These runners all follow the same ordering:

1. run `LambdaSVM`
2. run `DeepSets round2`
3. only run `SetTransformerLite` if the earlier two families do not produce a gate-passing candidate

### 3.1 Math runner

`scripts/run_math_bestofn_research_round2.py`

Fixed search scope:

- `LambdaSVM`
  - feature families: `all_aug`, `consensus_aug`
  - losses: `hinge`, `squared_hinge`
  - `C`: `0.1`, `1.0`
  - pair weighting: `uniform`, `dcg_delta`
- `DeepSets round2`
  - pooling: `mean`, `max`
  - dims: `(16, 8)`, `(24, 12)`
  - pairwise aux: `0.25`, `0.50`
- `SetTransformerLite`
  - heads: `2`, `4`
  - pairwise aux: `0.0`, `0.25`

The math runner still evaluates against the **current no-math submission-line baseline**, and still treats broad full-math replacement as non-default.

### 3.2 GPQA runner

`scripts/run_gpqa_bestofn_research_round2.py`

Fixed search scope:

- `LambdaSVM`
  - feature views: `base`, `margin_dominance`
  - losses: `hinge`, `squared_hinge`
  - `C`: `0.1`, `1.0`
  - pair weighting: `uniform`, `dcg_delta`
- `DeepSets round2`
  - pooling: `mean`, `max`
  - dims: `(16, 8)`, `(24, 12)`
  - pairwise aux: `0.25`, `0.50`
- `SetTransformerLite`
  - heads: `2`, `4`
  - pairwise aux: `0.0`, `0.25`

The GPQA runner always compares against `science_hybrid_round3`.

### 3.3 Coding runner

`scripts/run_code_bestofn_research_round2.py`

Fixed search scope:

- `LambdaSVM`
  - losses: `hinge`, `squared_hinge`
  - `C`: `0.1`, `1.0`
  - pair weighting: `dcg_delta`
- `DeepSets round2`
  - pooling: `mean`, `max`
  - dims: `(16, 8)`, `(24, 12)`
  - pairwise aux: `0.25`, `0.50`
- `SetTransformerLite`
  - heads: `2`, `4`
  - pairwise aux: `0.0`, `0.25`

The coding runner always compares against `code_v2`.

## 4. How to run

### 4.1 Keep current submission line

```bash
source .venv/bin/activate
python scripts/run_bestofn_score_recovery_20260408.py
```

### 4.2 Continue research

```bash
source .venv/bin/activate
python scripts/run_math_bestofn_research_round2.py
python scripts/run_gpqa_bestofn_research_round2.py
python scripts/run_code_bestofn_research_round2.py
```

## 5. Current interpretation

This repo now cleanly separates:

- **what to submit next**
- **what to continue researching next**

That avoids mixing the conservative score-recovery line with wider model exploration.
