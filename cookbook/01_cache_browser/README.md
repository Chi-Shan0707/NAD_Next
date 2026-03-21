# Chapter 01 — Cache Browser

## What is the Cache Browser?

NAD experiments run on **pre-built caches** — binary snapshots of neuron activation
data collected from a language model during inference. Before running any analysis
you need to know which caches exist, what datasets they cover, how many samples they
contain, and what the baseline model accuracy is.

The Cache Browser solves this by providing a **local web UI** that scans the cache
directories and presents all of that information on one page. Instead of navigating
folders and reading `meta.json` files manually, you open a browser tab and get an
instant summary.

It is also useful for **picking the right cache** before running `step2_analyze.sh`:
you can compare build dates, sample counts, and temperatures across caches for the
same dataset, then copy the exact path you need.

---

## What it shows

The browser surfaces two groups of caches on a single page:

| Section | What it lists |
|---------|--------------|
| **MUI Public Caches** | All caches under `MUI_HUB/cache/`, organised by model → dataset |
| **Local Caches** | Any `cache_*/` directories in the repository root |

For each cache it displays: dataset name, cache type, number of samples, number of
problems, model accuracy, temperature, and build date.

A search box at the top filters all cards in real time.

---

## Prerequisites

- Chapter 00 completed (`verify.sh` all green)
- `flask` installed (included in `install.sh`)
- The `MUI_HUB` symlink created

---

## Starting the server

Run from the repository root:

```bash
# Foreground (Ctrl-C to stop)
bash cookbook/01_cache_browser/cache_browser.sh

# Background (recommended)
bash cookbook/01_cache_browser/cache_browser.sh --background
```

Then open **http://localhost:5003** in your browser.

---

## All options

```
./cache_browser.sh [options]

  --port PORT       Server port            (default: 5003)
  --host HOST       Bind host              (default: 0.0.0.0)
  --vis-port PORT   Visualization server port for links  (default: 5002)
  --debug           Enable Flask debug mode
  --background      Run in background (writes PID to /tmp/nad_cache_browser.pid)
  --kill            Stop a running background server
  --status          Check whether the server is running
  -h, --help        Show help
```

---

## Managing the server

```bash
# Check if it is running
bash cookbook/01_cache_browser/cache_browser.sh --status

# Stop it
bash cookbook/01_cache_browser/cache_browser.sh --kill

# Watch live logs (background mode)
tail -f cache_browser.log
```

---

## JSON API

The server also exposes a REST API, useful for scripting.

### `GET /api/caches`

Returns all caches as structured JSON.

```bash
curl http://localhost:5003/api/caches | python3 -m json.tool
```

Top-level keys:

```
{
  "mui_public": {
    "<model>": {
      "<dataset>": [ { ...cache metadata... }, ... ]
    }
  },
  "local": [ { ...cache metadata... } ],
  "stats": {
    "total_models":   1,
    "total_datasets": 6,
    "total_caches":   7,
    "total_samples":  "32,960"
  }
}
```

Example — list every dataset and how many caches it has:

```bash
curl -s http://localhost:5003/api/caches | python3 -c "
import json, sys
d = json.load(sys.stdin)
for model, datasets in d['mui_public'].items():
    print(f'Model: {model}')
    for ds, caches in datasets.items():
        print(f'  {ds}: {len(caches)} cache(s)')
print()
print('Stats:', d['stats'])
"
```

Output:

```
Model: DeepSeek-R1-0528-Qwen3-8B
  aime24: 2 cache(s)
  aime25: 1 cache(s)
  brumo25: 1 cache(s)
  gpqa: 1 cache(s)
  hmmt25: 1 cache(s)
  livecodebench_v5: 1 cache(s)

Stats: {'total_caches': 7, 'total_datasets': 6, 'total_models': 1, 'total_samples': '32,960'}
```

### `GET /api/cache/<path>`

Returns metadata for a single cache.

```bash
curl -s http://localhost:5003/api/cache/MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B/aime24/cache_neuron_output_1_act_no_rms_20250902_025610 \
  | python3 -m json.tool
```

Key fields returned:

| Field | Description |
|-------|-------------|
| `num_samples` | Total inference runs (e.g. 1920) |
| `num_problems` | Distinct problems (e.g. 30) |
| `accuracy` | Pass rate across all runs (e.g. 75.36%) |
| `cache_type` | Layer type, e.g. `L1 Act` |
| `temperature` | Sampling temperature used during inference |
| `date` | Cache build date |
| `version` | Cache format version |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `MUI Public Caches` section is empty | Confirm `MUI_HUB` symlink exists (`bash cookbook/00_setup/verify.sh`) |
| Port 5003 already in use | Start with `--port 5004` (or any free port) |
| `ModuleNotFoundError: flask` | Run `bash cookbook/00_setup/install.sh` |
| Server started but page doesn't load | Check `tail -20 cache_browser.log` for errors |

---

## Next steps

With the Cache Browser running you can visually confirm which caches are available
before running any analysis.

Proceed to **Chapter 02 — Running Your First Analysis** to run `step2_analyze.sh`
on one of these caches.
