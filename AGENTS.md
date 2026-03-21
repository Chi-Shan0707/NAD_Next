# Repository Guidelines

## Project Structure & Module Organization
`nad/` contains the main Python package: `cli/` exposes commands, `pipeline/` orchestrates cache building and analysis, `ops/` holds scoring logic, and `io/` handles cache loading. Use `plugins/` for custom selectors, `tools/` for standalone utilities, and `scripts/` for ranking or plotting helpers. `cookbook/00_setup` through `cookbook/05_deepconf_analysis` are the primary reproducible workflows. `minimal_visualization_next/` is a separate Flask + Plotly viewer. Treat `result/` and `*.log` files as generated output.

## Build, Test, and Development Commands
Run all commands from the repository root.

- `bash cookbook/00_setup/install.sh` — install core and optional Python dependencies.
- `bash cookbook/00_setup/verify.sh` — verify Python, packages, and the `MUI_HUB` symlink.
- `python3 -m nad.cli analyze --cache-root MUI_HUB/cache/... --selectors all --out result.json` — run selector analysis on one cache.
- `python3 -m nad.cli accuracy --selection result.json --cache-root MUI_HUB/cache/... --out accuracy.json` — score selections against ground truth.
- `python3 scripts/rank_selectors.py --results-dir ./result/all_model_TIMESTAMP --csv --json` — aggregate results across tasks.
- `bash cookbook/01_cache_browser/cache_browser.sh --background` — start the cache browser on port `5003`.

## Coding Style & Naming Conventions
Target Python 3.9+ and follow existing conventions: 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and explicit long-form CLI flags. Match the current style of small typed helper functions. No formatter or linter is configured in `pyproject.toml`, so avoid broad reformatting-only diffs. Keep plugin selectors in files like `plugins/my_selector.py`; selector classes should subclass `Selector` and return group-local indices.

## Testing Guidelines
This repository does not yet have a formal unit-test suite. At minimum, run `bash cookbook/00_setup/verify.sh` before submitting changes. Use targeted smoke checks for the area you touched, such as `python3 -m nad.cli --help`, a relevant cookbook script, or `bash scripts/test_legacy_adapter.sh` for compatibility work. For selector or cache changes, validate against a real `MUI_HUB/cache/...` dataset.

## Commit & Pull Request Guidelines
The visible history starts with `Initial commit: NAD Next framework`; keep commit subjects short, imperative, and specific, for example `cli: tighten analyze validation`. Pull requests should summarize the affected pipeline stage, list the commands used for verification, note any `MUI_HUB` or `nad_config.json` assumptions, and include screenshots for browser or visualization changes. Do not commit large cache artifacts, local logs, or generated result directories unless the change explicitly requires tracked fixtures.

## Configuration & Performance Notes
Prefer relative `MUI_HUB/...` cache paths because the helper scripts assume repo-root execution. On this machine, keep `THREADS × PARALLEL_JOBS <= 16` when adjusting parallel analysis scripts.
