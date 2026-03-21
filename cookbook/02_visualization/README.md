# Chapter 02 — Visualization Server

## What is the Visualization Server?

The Visualization Server is an interactive web UI for deep-diving into a single NAD cache.
While the Cache Browser (Chapter 01) gives you a high-level overview across all caches,
the Visualization Server lets you explore the **contents** of one cache in detail:

- Browse neuron activation patterns problem by problem
- See which runs are correct / incorrect and how they differ
- Inspect token-level entropy and confidence distributions
- View selector performance per problem
- Decode token IDs into human-readable text (requires tokenizer)

It is powered by Flask + Plotly and runs entirely locally.

---

## Prerequisites

- Chapter 00 completed — all 12 checks green (`bash cookbook/00_setup/verify.sh`)
- The `MUI_HUB` symlink created (Chapter 00, Step 3)
- `nad_config.json` at the repo root with the correct model search path

### nad_config.json

The server needs to locate the model directory to load the tokenizer.
This is configured in `nad_config.json` at the repository root:

```json
{
  "model_search_dirs": [
    "/home/jovyan/public-ro/model"
  ]
}
```

The server extracts the model name from the cache's `meta.json`
(e.g. `DeepSeek-R1-0528-Qwen3-8B`) and searches each listed directory for a
folder with that exact name.

---

## Starting the server

Run from the repository root, passing the cache path **as a relative path via the `MUI_HUB` symlink**:

```bash
bash cookbook/02_visualization/visualization.sh \
  MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  --background
```

Then open the server in your browser.

> **Important:** Always use the `MUI_HUB/...` relative path, not the absolute
> `/home/jovyan/public-ro/...` path. The script prepends `../` internally, so an
> absolute path would produce a broken path like `..//home/jovyan/...`.

### Accessing the server

| Environment | URL |
|-------------|-----|
| Direct (local machine) | `http://localhost:5002` |
| JupyterHub | `http://<host>/user/<username>/proxy/5002/` |

The server handles both cases automatically via `ProxyFix` middleware —
no configuration needed regardless of whether a proxy is present.

### Available caches

| Dataset | Relative path |
|---------|--------------|
| aime24 (Sep 2025) | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610` |
| aime24 (Nov 2025) | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20251126_073502` |
| aime25 | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime25/cache_neuron_output_1_act_no_rms_20251126_114548` |
| brumo25 | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/brumo25/cache_neuron_output_1_act_no_rms_20251126_071142` |
| gpqa | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/gpqa/cache_neuron_output_1_act_no_rms_20251126_111853` |
| hmmt25 | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/hmmt25/cache_neuron_output_1_act_no_rms_20251126_223151` |
| livecodebench_v5 | `MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/livecodebench_v5/cache_neuron_output_1_act_no_rms_20251127_032808` |

---

## All options

```
visualization.sh [cache_dir] [port] [options]

  --background         Run in background (recommended)
  --kill               Stop a running server
  --status             Check whether the server is running
  --list-caches        List all detected cache directories
  --port PORT          Server port (default: 5002)
  --host HOST          Bind host (default: 0.0.0.0)
  --max-cache-mb SIZE  LRU cache memory limit in MB (default: 256)
  --debug              Enable Flask debug mode
  --no-browser         Do not auto-open browser
  -h, --help           Show help
```

---

## Managing the server

```bash
# Check if it is running
bash cookbook/02_visualization/visualization.sh --status

# Stop it
bash cookbook/02_visualization/visualization.sh --kill

# Watch live logs (background mode)
tail -f visualization.log
```

---

## What a healthy startup looks like

Check `visualization.log` after starting. You should see:

```
📁 Model path: /datacenter/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/
🔍 Model path not found: /datacenter/models/...
📁 Resolved model path: ... -> /home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B
✓ Loaded tokenizer via AutoTokenizer from /home/jovyan/public-ro/model/DeepSeek-R1-0528-Qwen3-8B
✓ Tokenizer pre-loaded successfully
   Problems : 30
   Samples  : 1920
   Tokenizer: ✓ Loaded
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: hmmlearn` | Package missing | `bash cookbook/00_setup/install.sh` |
| `ModuleNotFoundError: tokenizers` | Package missing | `bash cookbook/00_setup/install.sh` |
| `Tokenizer: ✗ Not available` | Model dir not found | Check `nad_config.json` has the correct path |
| Cache path not found `..//home/...` | Absolute path passed | Use `MUI_HUB/cache/...` relative path instead |
| Port 5002 already in use | Another server running | Use `--port 5003` or stop the other server first |
| Main page loads but API calls return 404 | JupyterHub proxy prefix not forwarded | Already fixed via `ProxyFix` + `BASE_URL` injection — ensure you are running the latest version of `app.py` and `index.html` |

---

## Next steps

Proceed to **Chapter 03 — Running Your First Analysis** to run `step2_analyze.sh`
and compute selector accuracy on a cache.
