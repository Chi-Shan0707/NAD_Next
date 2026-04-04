#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _rank_scale(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return [100.0]
    order = np.argsort(arr, kind='mergesort')
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    scaled = 1.0 + 99.0 * ranks / float(arr.size - 1)
    return scaled.tolist()


def _minmax_scale(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    lo = float(arr.min())
    hi = float(arr.max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError('non-finite values encountered')
    if np.isclose(lo, hi):
        return [100.0 for _ in values]
    scaled = 1.0 + 99.0 * (arr - lo) / (hi - lo)
    return scaled.tolist()


def _default_out_path(in_path: Path, method: str) -> Path:
    return in_path.with_name(f'{in_path.stem}_scale100_{method}{in_path.suffix}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Rescale Best-of-N submission scores into the 1-100 range')
    ap.add_argument('input', help='Input submission JSON path')
    ap.add_argument('--out', default='', help='Optional output JSON path')
    ap.add_argument('--method', choices=('rank', 'minmax'), default='rank', help='Rescaling method per cache_key/problem_id block')
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else _default_out_path(in_path, args.method)
    data = json.loads(in_path.read_text(encoding='utf-8'))
    scores = data.get('scores')
    if not isinstance(scores, dict):
        raise SystemExit('Invalid submission JSON: missing scores mapping')

    scaler = _rank_scale if args.method == 'rank' else _minmax_scale

    for cache_key, problem_map in scores.items():
        if not isinstance(problem_map, dict):
            raise SystemExit(f'Invalid problem map for {cache_key}')
        for problem_id, sample_map in problem_map.items():
            if not isinstance(sample_map, dict):
                raise SystemExit(f'Invalid sample map for {cache_key}/{problem_id}')
            sample_ids = list(sample_map.keys())
            values = [float(sample_map[sample_id]) for sample_id in sample_ids]
            scaled = scaler(values)
            scores[cache_key][problem_id] = {
                sample_id: float(value)
                for sample_id, value in zip(sample_ids, scaled)
            }

    data['score_postprocess'] = {
        'scale': '1-100',
        'method': args.method,
        'note': 'Monotonic per-cache_key/per-problem rescaling applied after raw export.',
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    print(out_path)


if __name__ == '__main__':
    main()
