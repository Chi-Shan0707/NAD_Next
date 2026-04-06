# Code v2 Candidate

Date: `2026-04-05`

## Summary

This note records the implementation status of the narrow `code_v2_candidate`
line that follows `code_baseline_v1`.

Implemented files:

- `nad/core/selectors/code_v2_impl.py`
- `plugins/prefix_saturation_v2_selector.py`
- `scripts/run_code_v2_candidate.py`

Current status:

- the candidate feature family is implemented
- the selector stays separate from `code_baseline_v1`
- the evaluation script mirrors the phase-2 protocol:
  - compare against `code_baseline_v1`
  - leave-one-out over the v2 feature set
  - rerun the same `{thr, lb, pf}` small grid
  - rerun blind DS/Qwen overlap checks

## Implemented Feature Set

`code_v2_candidate` keeps the narrow coding structure line:

- `prefix_best_window_quality`
- `head_tail_gap`
- `tail_variance`
- `post_reflection_recovery`
- `last_block_instability`

Default experimental weights:

- `prefix_best_window_quality = 0.40`
- `head_tail_gap = 0.06`
- `tail_variance = 0.16`
- `post_reflection_recovery = 0.22`
- `last_block_instability = 0.16`

Interpretation:

- `reflection_density` is intentionally removed from the promoted score path
- `head_tail_gap` is demoted to a small auxiliary term
- the new feature family is `last_block_instability`, exactly as required by the
  phase-2 handoff

## Runtime Note

`scripts/run_code_v2_candidate.py` is substantially heavier than the science
round-1 evaluation because it keeps the original coding protocol intact:

- per-problem dense distance matrix
- leave-one-out
- full small-grid replay
- blind DS/Qwen overlap pass

On this machine, the script runs correctly but is expensive enough that the full
exhaustive pass is better treated as a long-running job rather than a quick
smoke command.

## Bottom Line

The repo now has:

- a frozen `code_baseline_v1`
- an implemented `code_v2_candidate`
- a dedicated script to rerun the exact promotion gate for that candidate

What is **not** changed yet:

- `code_baseline_v1` is still the only promoted coding baseline
- no new submission patch is exported from the v2 line by default
- graph and `earlystop_v3` remain out of scope
