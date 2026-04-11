# Evaluation Guide

Tasks, submission format, and metric definitions.

## Overview

This leaderboard evaluates LLM output verifiers: methods that score or rank the outputs of a large language model without running them.

## Current Dataset

### Task 1 & 2

12 test caches across 6 benchmarks: outputs from `DeepSeek-R1-0528-Qwen3-8B` and `Qwen3-4B-Thinking-2507` on `AIME 2024`, `AIME 2025`, `BrumO 2025`, `GPQA`, `HMMT 2025`, and `LiveCodeBench v5`. Metrics are averaged across all 12 caches.

Cache keys:

- `DS-R1/aime24`
- `DS-R1/aime25`
- `DS-R1/brumo25`
- `DS-R1/gpqa`
- `DS-R1/hmmt25`
- `DS-R1/lcb_v5`
- `Qwen3-4B/aime24`
- `Qwen3-4B/aime25`
- `Qwen3-4B/brumo25`
- `Qwen3-4B/gpqa`
- `Qwen3-4B/hmmt25`
- `Qwen3-4B/lcb_v5`

### Task 3

RL Checkpoints: 11 training checkpoints of `Qwen3-4B` fine-tuned with RL (`base`, `step-100` ... `step-1000`).

There are three evaluation tasks, each testing a different use-case of your verifier. You submit a JSON file containing your scores; the system computes all metrics automatically.

## Task 1

### Best-of-N Selection

#### Scenario

An LLM generates `N` candidate answers (samples) for each problem. Your verifier assigns a confidence score to every sample, and the system uses these scores to pick the best answer. Higher score = more likely correct.

#### What you submit

`scores[cache_key][problem_id][sample_id] = one float per sample`

All 12 cache keys are required. For every cache, your submission must exactly cover all expected problem IDs and sample IDs. Metrics are computed per cache and averaged.

```json
{
  "task": "best_of_n",
  "method_name": "my_method",
  "scores": {
    "DS-R1/aime24": {
      "60": { "0": 0.92, "1": 0.35 },
      "61": { "64": 0.15, "65": 0.88 }
    },
    "DS-R1/aime25": {},
    "DS-R1/brumo25": {},
    "DS-R1/gpqa": {},
    "DS-R1/hmmt25": {},
    "DS-R1/lcb_v5": {},
    "Qwen3-4B/aime24": {},
    "Qwen3-4B/aime25": {},
    "Qwen3-4B/brumo25": {},
    "Qwen3-4B/gpqa": {},
    "Qwen3-4B/hmmt25": {},
    "Qwen3-4B/lcb_v5": {}
  }
}
```

#### Metrics

| Metric | What it measures |
|---|---|
| AUROC primary | Binary classification: can your score distinguish correct from incorrect samples across all problems? `1.0` = perfect separation, `0.5` = random. |
| Hit@1 | For each problem, pick the highest-scoring sample. Fraction of problems where that sample is correct. |
| Hit@3 | For each problem, pick the top-3 scoring samples. Fraction of problems where at least one is correct. |
| Selective Acc @10% | Keep only the top 10% highest-scoring samples globally. What fraction of them are correct? |
| Pairwise Accuracy | Among all `(correct, incorrect)` pairs within each problem, how often does the correct sample score higher? |

The Best-of-N leaderboard is ranked by average rank across all 5 metrics, not by AUROC alone. For each metric, users are ranked from best to worst, then the 5 ranks are averaged. Lower is better.

## Task 2

### Early Stop

#### Scenario

While the LLM is still generating a response, you can peek at its partial output at 10 progress checkpoints (`10%`, `20%`, ..., `100%`). At each point your verifier gives a score predicting whether the final answer will be correct. The goal: detect correctness as early as possible so you can stop wasting compute on hopeless samples.

#### What you submit

`scores[cache_key][problem_id][sample_id] = a list of 10 floats, one per position (10% -> 100%)`

All 12 cache keys are required, and each cache must exactly cover all expected problem IDs and sample IDs.

```json
{
  "task": "early_stop",
  "method_name": "my_method",
  "scores": {
    "DS-R1/aime24": {
      "60": {
        "0": [0.10, 0.25, 0.38, 0.52, 0.61, 0.70, 0.76, 0.82, 0.88, 0.92],
        "1": [0.05, 0.12, 0.18, 0.22, 0.28, 0.30, 0.33, 0.34, 0.35, 0.35]
      }
    },
    "DS-R1/aime25": {},
    "DS-R1/brumo25": {},
    "DS-R1/gpqa": {},
    "DS-R1/hmmt25": {},
    "DS-R1/lcb_v5": {},
    "Qwen3-4B/aime24": {},
    "Qwen3-4B/aime25": {},
    "Qwen3-4B/brumo25": {},
    "Qwen3-4B/gpqa": {},
    "Qwen3-4B/hmmt25": {},
    "Qwen3-4B/lcb_v5": {}
  }
}
```

#### Metrics

At each of the 10 positions, three metrics are computed, forming three curves:

| Curve | What it measures at each position |
|---|---|
| AUROC curve | Can the score at this progress point distinguish correct from incorrect samples? Higher and earlier is better. |
| Selective Acc curve | If we keep only the top 10% scoring samples at this position, what fraction are correct? |
| Stop Acc curve | For each problem, pick the top-1 sample by score at this position. Fraction of problems where it is correct. |

These curves are then summarized into scalar metrics:

| Metric | Definition |
|---|---|
| AUC of AUROC primary | Area under the AUROC curve (trapezoidal). Captures how high and how early your AUROC rises. |
| AUC of Selective Acc | Area under the Selective Accuracy curve. |
| Earliest AUROC > 0.6 | The first position where AUROC exceeds `0.6`, e.g. `30%` means your verifier is useful at `30%` generation progress. |
| Stop Acc @100% | Hit@1 at the final position (`100%`). Essentially Best-of-N selection accuracy using the full output. |

The Early Stop leaderboard is ranked by average rank across 32 numeric metrics: `auc_of_auroc`, `auc_of_selective_acc`, and all 30 curve values (10 positions each for `AUROC`, `Selective Acc`, and `Stop Acc`). Lower is better.

## Task 3

### Checkpoint Ranking

#### Scenario

An LLM undergoes RL training, producing 11 checkpoints (`base`, `step-100`, `step-200`, ..., `step-1000`). Your verifier assigns a quality score to each checkpoint, predicting how accurate that checkpoint is, without running an evaluation suite. The system compares your ranking against ground-truth accuracy.

#### What you submit

`scores[checkpoint_name] = one float per checkpoint`

There must be exactly 11 entries, matching the checkpoint names exactly.

```json
{
  "task": "checkpoint_ranking",
  "method_name": "my_method",
  "scores": {
    "base": 12.5,
    "step-100": 14.2,
    "step-200": 15.8,
    "step-300": 17.1,
    "step-400": 18.3,
    "step-500": 19.0,
    "step-600": 19.5,
    "step-700": 20.1,
    "step-800": 20.6,
    "step-900": 20.9,
    "step-1000": 21.2
  }
}
```

#### Metrics

| Metric | What it measures |
|---|---|
| Spearman rho primary | Rank correlation between your predicted ranking and the true accuracy ranking. `+1` = perfect agreement, `0` = no correlation. |
| Pearson r | Linear correlation between your scores and ground-truth accuracy values. |
| Kendall tau | Fraction of concordant pairs minus discordant pairs, normalized. Another rank correlation measure that is stricter on ties. |
| Top-1 Hit | Does your highest-scoring checkpoint match the actually best checkpoint? `1` = yes, `0` = no. |
| Top-3 Hit | How many of the true top-3 checkpoints appear in your predicted top-3? (`0` to `3`) |

## Overall Scoring

The Overall leaderboard aggregates performance across Best-of-N and Early Stop using equal-weight average ranking. Checkpoint Ranking has its own leaderboard but does not participate in the overall score.

To avoid double-counting, Early Stop metrics at the `100%` position are excluded from the overall scoring, since they test the same thing as the corresponding Best-of-N metrics (`AUROC`, `Selective Acc`, `Hit@1` on the full output).

For each of the 34 individual metrics (`5` BoN + `29` ES), users are ranked from best to worst.

- `BoN_avg = mean(5 BoN ranks)`
- `ES_avg = mean(29 ES ranks)`
- `Score = BoN_avg + ES_avg`

Each task contributes exactly `1/2` of the final score. Lower is better. Only users who have submitted to both tasks are included.

## Example Downloads

Download minimal example JSON files to see the exact structure:

- `example_best_of_n.json`
- `example_early_stop.json`
- `example_checkpoint_ranking.json`

Note: examples contain only 2 to 3 entries per level for readability. Real submissions must cover all problems, samples, and checkpoints.

## Title

LLM Evaluation Leaderboard
