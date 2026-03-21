#!/usr/bin/env python3
"""
Position Ablation Visualization Script

Generates accuracy vs token consumption plots for each cache and selector.
Also generates CSV summary files with Avg@64 baseline and selector metrics.

Usage:
    python scripts/plot_position_ablation.py --result-dir result/position_ablation_xxx --cache-base MUI_HUB/cache/DeepSeek-R1-0528-Qwen3-8B
"""

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Configure matplotlib for better output
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


WINDOWS = ['0-1', '0-2', '0-8', '0-32', '0-128', '0-512']
WINDOW_TOKENS = {
    '0-1': 32,
    '0-2': 64,
    '0-8': 256,
    '0-32': 1024,
    '0-128': 4096,
    '0-512': 16384,
}
SELECTORS = ['min-activation', 'medoid', 'knn-medoid', 'dbscan-medoid']


def load_sample_tokens(cache_path: str) -> List[int]:
    """Load token counts for each sample from rows/token_row_ptr."""
    rows_dir = Path(cache_path) / 'rows'

    trp = np.memmap(rows_dir / 'token_row_ptr.int64', dtype=np.int64, mode='r')
    srp = np.memmap(rows_dir / 'sample_row_ptr.int64', dtype=np.int64, mode='r')

    num_samples = len(srp) - 1
    sample_tokens = []

    for i in range(num_samples):
        row_start = int(srp[i])
        row_end = int(srp[i + 1])
        if row_end > row_start:
            base = int(trp[row_start])
            total = int(trp[row_end]) - base
        else:
            total = 0
        sample_tokens.append(total)

    return sample_tokens


def load_sample_mapping(cache_path: str) -> Dict[Tuple[int, int], int]:
    """Load (problem_id, run_index) -> global_idx mapping from meta.json."""
    meta_path = Path(cache_path) / 'meta.json'

    with open(meta_path) as f:
        meta = json.load(f)

    mapping = {}
    for i, s in enumerate(meta['samples']):
        mapping[(s['problem_id'], s['run_index'])] = i

    return mapping


def calculate_token_consumption(
    result_data: dict,
    sample_tokens: List[int],
    sample_mapping: Dict[Tuple, int],
    selector: str,
    window_tokens: int
) -> int:
    """
    Calculate total token consumption for a selector.

    Token Consumption = selection_cost + remaining_cost
    - selection_cost: sum of min(token_i, W) for all samples
    - remaining_cost: selected_token - W (if > 0)
    """
    total_consumption = 0

    for problem_id_str, problem_data in result_data['problems'].items():
        run_ids = problem_data['run_ids']

        # Get token counts for this problem's samples
        # Note: run_ids in result may be global indices, but run_index in meta.json
        # is local to each problem (1-512). Use enumerate to get local index.
        problem_tokens = []
        for local_idx, run_id in enumerate(run_ids):
            # run_index in meta.json starts from 1
            # Try multiple key formats: string, int, original
            run_index = local_idx + 1
            global_idx = None

            # Try string key first (most common in newer caches)
            global_idx = sample_mapping.get((problem_id_str, run_index))

            # Try int key if string didn't work
            if global_idx is None:
                try:
                    problem_id_int = int(problem_id_str)
                    global_idx = sample_mapping.get((problem_id_int, run_index))
                except ValueError:
                    pass

            if global_idx is not None and global_idx < len(sample_tokens):
                problem_tokens.append(sample_tokens[global_idx])
            else:
                problem_tokens.append(0)

        # Selection cost: all samples contribute min(token, W)
        selection_cost = sum(min(t, window_tokens) for t in problem_tokens)

        # Get selected sample's remaining cost
        if selector in problem_data.get('selectors', {}):
            selected_idx = problem_data['selectors'][selector]
            if 0 <= selected_idx < len(problem_tokens):
                selected_tokens = problem_tokens[selected_idx]
                remaining_cost = max(0, selected_tokens - window_tokens)
            else:
                remaining_cost = 0
        else:
            remaining_cost = 0

        total_consumption += selection_cost + remaining_cost

    return total_consumption


def load_accuracy(result_dir: str, cache_name: str, window: str, selector: str) -> float:
    """Load accuracy for a specific window and selector."""
    # Find the dataset directory containing this cache
    for dataset in os.listdir(result_dir):
        dataset_path = Path(result_dir) / dataset
        if not dataset_path.is_dir():
            continue

        cache_path = dataset_path / cache_name
        if cache_path.exists():
            acc_file = cache_path / f'accuracy_{window}.json'
            if acc_file.exists():
                with open(acc_file) as f:
                    data = json.load(f)
                return data.get('selector_accuracy', {}).get(selector, 0.0)

    return 0.0


def load_avg64_accuracy(acc_file_path: Path) -> Optional[float]:
    """Load Avg@64 baseline accuracy from accuracy file."""
    if not acc_file_path.exists():
        return None
    with open(acc_file_path) as f:
        data = json.load(f)
    # Check selector_accuracy first, then top-level
    selector_acc = data.get('selector_accuracy', {})
    return selector_acc.get('avg64@', selector_acc.get('avg@64',
           data.get('avg64@', data.get('avg@64', data.get('baseline_accuracy')))))


def calculate_avg64_token_consumption(
    sample_tokens: List[int],
    sample_mapping: Dict[Tuple, int],
    result_data: dict
) -> int:
    """
    Calculate token consumption for Avg@64 baseline.

    Avg@64 means reading all tokens from all samples (no early stopping).
    Token Consumption = sum of all sample tokens
    """
    total_consumption = 0

    for problem_id_str, problem_data in result_data['problems'].items():
        run_ids = problem_data['run_ids']

        for local_idx, run_id in enumerate(run_ids):
            run_index = local_idx + 1
            global_idx = None

            # Try string key first
            global_idx = sample_mapping.get((problem_id_str, run_index))

            # Try int key if string didn't work
            if global_idx is None:
                try:
                    problem_id_int = int(problem_id_str)
                    global_idx = sample_mapping.get((problem_id_int, run_index))
                except ValueError:
                    pass

            if global_idx is not None and global_idx < len(sample_tokens):
                total_consumption += sample_tokens[global_idx]

    return total_consumption


def generate_summary_csv(
    output_dir: Path,
    dataset: str,
    cache_name: str,
    selectors: List[str],
    selector_data: Dict[str, Dict],
    avg64_accuracy: Optional[float],
    avg64_tokens: int
):
    """Generate CSV summary file for a cache."""
    csv_path = output_dir / dataset / cache_name / 'summary.csv'
    os.makedirs(csv_path.parent, exist_ok=True)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Method', 'Window', 'Accuracy (%)', 'Token Consumption', 'Token Consumption (M)'])

        # Avg@64 baseline (full sequence, no early stopping)
        avg64_acc_str = f'{avg64_accuracy:.2f}' if avg64_accuracy is not None else 'N/A'
        writer.writerow(['Avg@64', 'full', avg64_acc_str, avg64_tokens, f'{avg64_tokens/1e6:.2f}'])

        # Each selector at window 0-1
        for selector in selectors:
            if selector in selector_data:
                data = selector_data[selector]
                acc = data.get('accuracy_0-1', 0.0)
                tokens = data.get('tokens_0-1', 0)
                writer.writerow([selector, '0-1', f'{acc:.2f}', tokens, f'{tokens/1e6:.2f}'])

        # Empty row as separator
        writer.writerow([])

        # Full data for all windows
        writer.writerow(['# Full data for all windows'])
        writer.writerow(['Method', 'Window', 'Accuracy (%)', 'Token Consumption', 'Token Consumption (M)'])

        for selector in selectors:
            if selector in selector_data:
                data = selector_data[selector]
                for window in WINDOWS:
                    acc = data.get(f'accuracy_{window}', 0.0)
                    tokens = data.get(f'tokens_{window}', 0)
                    writer.writerow([selector, window, f'{acc:.2f}', tokens, f'{tokens/1e6:.2f}'])

    return csv_path


def plot_selector(
    cache_name: str,
    dataset: str,
    selector: str,
    accuracies: List[float],
    token_consumptions: List[float],
    output_path: str
):
    """Generate a dual-axis plot for a selector."""
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # X-axis: token positions (log scale)
    x_values = [WINDOW_TOKENS[w] for w in WINDOWS]

    # Left Y-axis: Accuracy (blue)
    color1 = 'tab:blue'
    ax1.set_xlabel('Early Stopping Position (log scale)', fontsize=12)
    ax1.set_ylabel('Accuracy', color=color1, fontsize=12)
    line1, = ax1.plot(x_values, accuracies, 'o-', color=color1, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)

    # Only show even power labels (2^6, 2^8, 2^10, 2^12, 2^14) for cleaner look
    major_ticks = [64, 256, 1024, 4096, 16384]  # 2^6, 2^8, 2^10, 2^12, 2^14
    major_labels = ['$2^{6}$', '$2^{8}$', '$2^{10}$', '$2^{12}$', '$2^{14}$']
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels(major_labels)
    ax1.set_xlim(min(x_values) * 0.7, max(x_values) * 1.3)  # Add some padding
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Set Y-axis range for accuracy
    acc_min = min(accuracies) if accuracies else 0
    acc_max = max(accuracies) if accuracies else 100
    acc_range = acc_max - acc_min
    ax1.set_ylim(max(0, acc_min - acc_range * 0.1), min(100, acc_max + acc_range * 0.1))

    # Right Y-axis: Token Consumption (red)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Token Consumption (M)', color=color2, fontsize=12)
    # Convert to millions
    token_m = [t / 1e6 for t in token_consumptions]
    line2, = ax2.plot(x_values, token_m, 's--', color=color2, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title
    selector_display = selector.replace('-', ' ').title()
    plt.title(f'{selector_display}', fontsize=14, fontweight='bold')

    # Tight layout
    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def process_cache(
    result_dir: str,
    cache_base: str,
    dataset: str,
    cache_name: str,
    output_dir: str
):
    """Process a single cache and generate plots and CSV for all selectors."""
    cache_path = Path(cache_base) / dataset / cache_name
    result_cache_dir = Path(result_dir) / dataset / cache_name

    if not cache_path.exists():
        print(f"  Cache not found: {cache_path}")
        return

    if not result_cache_dir.exists():
        print(f"  Result not found: {result_cache_dir}")
        return

    # Load sample tokens and mapping
    try:
        sample_tokens = load_sample_tokens(str(cache_path))
        sample_mapping = load_sample_mapping(str(cache_path))
    except Exception as e:
        print(f"  Error loading cache data: {e}")
        return

    # Collect data for CSV
    selector_data = {}
    avg64_accuracy = None
    avg64_tokens = 0

    # Load Avg@64 from the first available accuracy file
    for window in WINDOWS:
        acc_file = result_cache_dir / f'accuracy_{window}.json'
        if acc_file.exists():
            avg64_accuracy = load_avg64_accuracy(acc_file)
            break

    # Calculate Avg@64 token consumption (full sequence, no early stopping)
    result_file = result_cache_dir / f'window_{WINDOWS[0]}_result.json'
    if result_file.exists():
        with open(result_file) as f:
            result_data = json.load(f)
        avg64_tokens = calculate_avg64_token_consumption(
            sample_tokens, sample_mapping, result_data
        )

    # Process each selector
    for selector in SELECTORS:
        accuracies = []
        token_consumptions = []
        selector_data[selector] = {}

        for window in WINDOWS:
            # Load accuracy
            acc_file = result_cache_dir / f'accuracy_{window}.json'
            if acc_file.exists():
                with open(acc_file) as f:
                    acc_data = json.load(f)
                acc = acc_data.get('selector_accuracy', {}).get(selector, 0.0)
            else:
                acc = 0.0
            accuracies.append(acc)
            selector_data[selector][f'accuracy_{window}'] = acc

            # Load result and calculate token consumption
            result_file = result_cache_dir / f'window_{window}_result.json'
            if result_file.exists():
                with open(result_file) as f:
                    result_data = json.load(f)
                consumption = calculate_token_consumption(
                    result_data, sample_tokens, sample_mapping,
                    selector, WINDOW_TOKENS[window]
                )
            else:
                consumption = 0
            token_consumptions.append(consumption)
            selector_data[selector][f'tokens_{window}'] = consumption

        # Generate plot
        # Directory structure: dataset/cache_name/selector.png
        output_path = Path(output_dir) / dataset / cache_name / f'{selector}.png'
        plot_selector(cache_name, dataset, selector, accuracies, token_consumptions, str(output_path))
        print(f"    Generated: {output_path}")

    # Generate CSV summary
    csv_path = generate_summary_csv(
        Path(output_dir), dataset, cache_name, SELECTORS,
        selector_data, avg64_accuracy, avg64_tokens
    )
    print(f"    Generated: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate position ablation plots')
    parser.add_argument('--result-dir', required=True, help='Result directory from position ablation')
    parser.add_argument('--cache-base', required=True, help='Base directory containing cache files')
    parser.add_argument('--output-dir', default=None, help='Output directory for charts (default: result_dir/charts)')
    parser.add_argument('--selectors', default=None, help='Comma-separated list of selectors (default: min-activation,medoid,knn-medoid,dbscan-medoid)')
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    cache_base = Path(args.cache_base)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir / 'charts'

    if args.selectors:
        global SELECTORS
        SELECTORS = args.selectors.split(',')

    print(f"Result directory: {result_dir}")
    print(f"Cache base: {cache_base}")
    print(f"Output directory: {output_dir}")
    print(f"Selectors: {SELECTORS}")
    print()

    # Find all datasets and caches
    total_plots = 0
    for dataset in sorted(os.listdir(result_dir)):
        dataset_path = result_dir / dataset
        if not dataset_path.is_dir() or dataset in ['charts', '.logs']:
            continue

        print(f"Processing dataset: {dataset}")

        for cache_name in sorted(os.listdir(dataset_path)):
            cache_path = dataset_path / cache_name
            if not cache_path.is_dir():
                continue

            print(f"  Cache: {cache_name}")
            process_cache(str(result_dir), str(cache_base), dataset, cache_name, str(output_dir))
            total_plots += len(SELECTORS)

    print()
    print(f"Total plots generated: {total_plots}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
