# 2026-04-05 Code Selector Handoff

## 1. Current state

The code-oriented selector slice is complete.

Implementation commit:

- `e725687` — `selectors: optimize code-oriented dynamic plugin`

Core files:

- `nad/core/selectors/code_dynamic_impl.py`
- `plugins/prefix_saturation_selector.py`

Primary validation note:

- `docs/CODE_SELECTOR_VALIDATION_20260405.md`

Validation artifacts:

- `result/tmp_selector_validation/lcb_code_selector_metrics_ja_full.json`
- `result/tmp_selector_validation/lcb_code_selector_analyze_ja_full.json`
- `result/tmp_selector_validation/lcb_code_selector_accuracy_ja_full.json`

## 2. Final result

On:

- `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808`

Deterministic ranking metrics are:

| Selector | Hit@1 | Pairwise | SelAcc@10 |
|---|---:|---:|---:|
| `min-confidence` | `59.88%` | `49.51%` | `52.15%` |
| `tournament-copeland` | `58.08%` | `49.92%` | `59.27%` |
| `PrefixSaturationSelector` | `59.28%` | `51.27%` | `61.70%` |

Decision:

- keep `PrefixSaturationSelector` as the current best coding-oriented selector for this slice

## 3. Important interpretation

- The plugin wins on the actual target objective: `SelAcc@10`.
- It also improves `Pairwise` over both compared baselines.
- `Hit@1` is slightly below `min-confidence`, but only by `0.60pp`, which stayed inside the guardrail used for this slice.
- `nad.cli accuracy` is still useful as an end-to-end smoke check, but it is not the deciding metric here.
- `tournament-copeland` looks better in raw CLI accuracy because the built-in selector samples from a softmax over Copeland wins; the deterministic ranking report uses raw Copeland scores instead.

## 4. What changed during finalization

The main last-mile fix was not a feature rewrite; it was making the reflection feature practical on long coding traces:

- reflection scanning is now bounded and local
- reflection detection now short-circuits once an event is established
- plugin defaults remain conservative and code-oriented

The final default reflection threshold is:

- `0.30`

I checked `0.20` as well, but it lost on overall `SelAcc@10` tradeoff.

## 5. What not to do next

- do **not** reopen graph-topology expansion for this slice
- do **not** let follow-up work drift into refactoring `nad/ops/earlystop_v3.py` unless scope is explicitly expanded
- do **not** blindly commit unrelated worktree files; this tree still contains many unrelated untracked or modified files

## 6. Most natural next steps

If continuing from here, the next best options are:

1. keep the plugin file-based and run one more coding-only transfer check on another coding cache if available
2. wire the validated feature family into a later learned coding selector or `earlystop_v3` only if scope is intentionally expanded
3. add a small reporting script that reproduces the deterministic `SelAcc@10` / `Pairwise` / `Hit@1` table from `result/tmp_selector_validation/lcb_code_selector_metrics_ja_full.json`

## 7. Bottom line

This slice is no longer in “prototype waiting for validation” status.

It is now:

- implemented
- validated
- documented
- committed
