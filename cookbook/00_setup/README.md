# Chapter 00 — Environment Setup

This chapter walks you through everything you need before running any experiment.
By the end, your environment will be ready to reproduce all results in this cookbook.

---

## Prerequisites

- Python **3.9+** (tested on 3.13)
- A Unix-like shell (Linux / macOS)
- Access to the NAD_Next repository

---

## Step 1 — Clone / locate the repository

```bash
# If you already have the repo, navigate to it:
cd /path/to/NAD_Next

# Verify you are in the right place:
ls step2_analyze.sh   # should print the filename
```

---

## Step 2 — Install Python dependencies

Run the installer script from the repository root:

```bash
bash cookbook/00_setup/install.sh
```

What it installs:

| Package | Role | Required |
|---------|------|----------|
| `numpy` | Array operations, distance computation | Yes |
| `pyroaring` | Roaring Bitmap Jaccard backend (3-5× faster) | Yes |
| `flask` | Cache Browser web UI | No |
| `plotly` | Interactive plots | No |
| `hmmlearn` | Visualization server (`step1.1`) | No |
| `tokenizers` | Token ID → text decoding in visualization | No |
| `transformers` | Token ID → text decoding in visualization | No |
| `psutil` | Memory profiling (`--enable-profiling`) | No |
| `tqdm` | Progress bars | No |

The script skips packages that are already installed, so it is safe to re-run.

---

## Step 3 — Verify the environment

Run the verification script from the repository root:

```bash
bash cookbook/00_setup/verify.sh
```

It checks three things:

1. **Python version** — must be 3.9 or newer
2. **All packages** — reports installed version or flags missing ones
3. **MUI_HUB symlink** — confirms the cache directory is reachable

Expected output when everything is ready:

```
[1/3] Python version
  ✓ Python 3.13 (>= 3.9)

[2/3] Python packages
  ✓ numpy     [required]  (2.4.2)
  ✓ pyroaring [required]  (1.0.4)
  ✓ flask     [web UI]    (3.1.3)
  ✓ plotly    [visualization] (6.6.0)
  ✓ psutil    [profiling] (7.2.2)
  ✓ tqdm      [progress bars] (4.67.3)

[3/3] MUI_HUB symlink
  ✓ MUI_HUB → /home/jovyan/public-ro/MUI_HUB
  ✓ cache root accessible  (7 cache directories found)

────────────────────────────────────
All 9 checks passed. You are ready to proceed.
```

If any check fails, re-run `install.sh` and then `verify.sh` again.

---

## Summary checklist

- [ ] Python 3.9+ installed
- [ ] `bash cookbook/00_setup/install.sh` completed without errors
- [ ] `bash cookbook/00_setup/verify.sh` reports **All checks passed**

Once all boxes are checked, proceed to **Chapter 01 — Running Your First Analysis**.
