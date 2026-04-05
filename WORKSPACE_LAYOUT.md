# Workspace Layout

This file is the quick map for the NAD_Next workspace after cleanup on 2026-04-05.

## Source code
- `nad/`: core package, CLI, pipeline, ops, IO, selectors.
- `plugins/`: custom selector implementations and experiments.
- `scripts/`: one-off runners, export helpers, ranking utilities.
- `tools/`: standalone utilities.
- `cookbook/`: reproducible workflows and setup helpers.
- `cot_viewer/`, `minimal_visualization_next/`: local visualization apps.

## Documentation
- `docs/`: primary documentation root.
- `docs/handoffs/2026-04-05/`: dated handoff notes for the coding-selector line.
- `docs/reference/`: stable reference notes such as feature inventories and runtime constraints.
- `docs/ideas/`: exploratory design notes.
- `docs/archive/raw_sessions/`: archived raw session transcripts kept for history, not for active reading.

## Structured artifacts
- `results/`: curated experiment outputs that are worth keeping and comparing.
- `results/scans/`: small one-off scan outputs grouped by topic.
- `results/selector_rankings/20260404/`: ranking CSV/JSON snapshots moved out of the repo root.
- `submission/`: exportable submission JSON files, split by strategy family.
- `submission/BestofN/extreme8/` and `submission/BestofN/extreme12/`: organized BestofN trees for baseline vs patched exports.
- `result/`: temporary/generated local runs and validation outputs.

## Root-level files intentionally kept
- `README.md`: project overview.
- `AGENTS.md`, `CLAUDE.md`: local agent instructions.
- `pyproject.toml`, `requirements.txt`, `nad_config.json`: environment/config.
- `MUI_HUB`: symlink to external cache storage.
- `k_token_per_kink.py`: standalone analysis entry kept at the root so direct script execution keeps working with current imports.

## Conventions used in this cleanup
- Dated handoffs now live under `docs/handoffs/<date>/`.
- Loose ranking `json/csv` files are grouped under `results/selector_rankings/<date>/`.
- Raw terminal/session dumps are archived under `docs/archive/raw_sessions/`.
- Core code paths stay unchanged to avoid breaking scripts.
