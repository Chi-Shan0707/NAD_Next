#!/usr/bin/env python3
"""Minimal import checker for the SVDomain paper environment."""

from __future__ import annotations

import importlib
import sys


MODULES = [
    ("numpy", "required numeric"),
    ("pyroaring", "required bitmap"),
    ("sklearn", "required ml"),
    ("joblib", "required serialization"),
    ("flask", "viewer backend"),
    ("plotly", "viewer frontend support"),
    ("hmmlearn", "viewer dependency"),
    ("tokenizers", "token decoding"),
    ("transformers", "token decoding"),
    ("psutil", "profiling"),
    ("tqdm", "progress"),
]


def main() -> int:
    ok = 0
    bad = 0
    print("=== SVDomain environment import check ===")
    for module_name, note in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK ] {module_name:<12} version={version:<12} note={note}")
            ok += 1
        except Exception as exc:  # pragma: no cover - diagnostic script
            print(f"[FAIL] {module_name:<12} error={exc}")
            bad += 1

    print(f"\nsummary: ok={ok} fail={bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
