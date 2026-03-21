#!/usr/bin/env python3
"""
NeurIPS-style visualization for downsample ablation experiments.

Generates a figure with 18 subplots (3 models × 6 datasets), showing:
- X-axis: k values (2, 4, 8, 16, 32, 64) on log scale
- Y-axis: Accuracy (%)
- Lines: 5 selectors with different colors/markers
- Shading: 95% confidence interval

Usage:
    python scripts/plot_downsample_ablation.py --results-dir result/downsample_ablation_multimodel_20251204_020133
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Tuple

# NeurIPS style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Colorblind-friendly palette (IBM Design)
COLORS = {
    'min-activation': '#648FFF',   # Blue
    'max-activation': '#DC267F',   # Magenta
    'medoid': '#FFB000',           # Gold
    'knn-medoid': '#FE6100',       # Orange
    'dbscan-medoid': '#785EF0',    # Purple
    'avg64@': '#999999',           # Gray (baseline)
    'con64@': '#2E8B57',           # SeaGreen (baseline)
}

MARKERS = {
    'min-activation': 'o',
    'max-activation': 's',
    'medoid': '^',
    'knn-medoid': 'D',
    'dbscan-medoid': 'v',
    'avg64@': '*',
    'con64@': 'P',
}

SELECTOR_LABELS = {
    'min-activation': 'Min-Act',
    'max-activation': 'Max-Act',
    'medoid': 'Medoid',
    'knn-medoid': 'KNN-Medoid',
    'dbscan-medoid': 'DBSCAN-Medoid',
    'avg64@': 'Avg@64',
    'con64@': 'Con@64',
}

DATASET_LABELS = {
    'aime24': 'AIME24',
    'aime25': 'AIME25',
    'gpqa': 'GPQA',
    'humaneval': 'HumanEval',
    'livecodebench_v5': 'LiveCodeBench',
    'mbpp': 'MBPP',
}

MODEL_LABELS = {
    'DeepSeek-R1-0528-Qwen3-8B': 'DeepSeek-R1-8B',
    'Qwen3-4B-Instruct-2507': 'Qwen3-4B-Inst',
    'Qwen3-4B-Thinking-2507': 'Qwen3-4B-Think',
}


def load_results(results_dir: Path) -> Dict:
    """Load all accuracy JSON files and organize by (model, dataset, selector, k)."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        model = model_dir.name

        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name

            for acc_file in dataset_dir.glob('accuracy_k*_seed*.json'):
                # Parse k and seed from filename
                parts = acc_file.stem.split('_')
                k = int(parts[1][1:])  # k16 -> 16
                seed = int(parts[2][4:])  # seed0 -> 0

                with open(acc_file) as f:
                    acc_data = json.load(f)

                for selector, accuracy in acc_data['selector_accuracy'].items():
                    data[model][dataset][selector][k].append(accuracy)

    return data


def compute_statistics(data: Dict) -> Dict:
    """Compute mean and 95% CI for each configuration."""
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model, datasets in data.items():
        for dataset, selectors in datasets.items():
            for selector, k_values in selectors.items():
                for k, accuracies in k_values.items():
                    arr = np.array(accuracies)
                    n = len(arr)
                    mean = np.mean(arr)
                    std = np.std(arr, ddof=1) if n > 1 else 0
                    ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0

                    stats[model][dataset][selector][k] = {
                        'mean': mean,
                        'std': std,
                        'ci95': ci95,
                        'error': std,  # Use std for error bars (±1σ)
                        'n': n,
                        'min': np.min(arr),
                        'max': np.max(arr),
                    }

    return stats


def plot_figure(stats: Dict, output_path: Path, layout: str = '3x6', error_type: str = 'std', datasets: List[str] = None):
    """Generate the main figure with all subplots."""

    models = sorted(stats.keys())
    if datasets is None:
        datasets = ['aime24', 'gpqa', 'humaneval']
    # Main selectors (as lines with markers)
    selectors = ['medoid', 'knn-medoid', 'dbscan-medoid']
    # Baseline selectors (as horizontal dashed lines)
    baseline_selectors = ['avg64@', 'con64@']

    # Determine available k values (exclude k=2)
    k_values = sorted([k for k in stats[models[0]][datasets[0]][selectors[0]].keys() if k != 2])

    n_models = len(models)
    n_datasets = len(datasets)

    # Adaptive figure size based on number of datasets and models
    if layout == '3x6':
        # Models as rows, datasets as columns
        fig_width = max(4, 4 * n_datasets + 2)
        fig_height = 2.5 * n_models + 1
        fig, axes = plt.subplots(n_models, n_datasets, figsize=(fig_width, fig_height), sharex=True)
        if n_datasets == 1 and n_models == 1:
            axes = np.array([[axes]])
        elif n_datasets == 1:
            axes = axes.reshape(-1, 1)
        elif n_models == 1:
            axes = axes.reshape(1, -1)
    else:  # 6x3 or horizontal layout
        # Models as columns, datasets as rows (horizontal for single dataset)
        fig_width = 4 * n_models + 2
        fig_height = max(3, 2.5 * n_datasets + 1)
        fig, axes = plt.subplots(n_datasets, n_models, figsize=(fig_width, fig_height), sharex=True)
        if n_datasets == 1 and n_models == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes.reshape(-1, 1)
        elif n_datasets == 1:
            axes = axes.reshape(1, -1)

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            if layout == '3x6':
                ax = axes[i, j]
            else:
                ax = axes[j, i]

            all_means = []
            all_errors = []

            for selector in selectors:
                if selector not in stats[model][dataset]:
                    continue

                k_data = stats[model][dataset][selector]
                ks = sorted([k for k in k_data.keys() if k != 2])  # Exclude k=2
                means = [k_data[k]['mean'] for k in ks]

                # Select error type
                if error_type == 'std':
                    errors = [k_data[k]['std'] for k in ks]
                elif error_type == 'ci95':
                    errors = [k_data[k]['ci95'] for k in ks]
                elif error_type == 'minmax':
                    errors_lower = [k_data[k]['mean'] - k_data[k]['min'] for k in ks]
                    errors_upper = [k_data[k]['max'] - k_data[k]['mean'] for k in ks]
                else:
                    errors = [k_data[k]['std'] for k in ks]

                all_means.extend(means)
                if error_type == 'minmax':
                    all_errors.extend(errors_lower)
                    all_errors.extend(errors_upper)
                else:
                    all_errors.extend(errors)

                color = COLORS[selector]
                marker = MARKERS[selector]
                label = SELECTOR_LABELS[selector]

                # Plot line with markers
                ax.plot(ks, means, color=color, marker=marker,
                       label=label, markerfacecolor='white', markeredgecolor=color,
                       markeredgewidth=1.2, zorder=3)

                # Plot shading
                if error_type == 'minmax':
                    lower = [k_data[k]['min'] for k in ks]
                    upper = [k_data[k]['max'] for k in ks]
                else:
                    lower = [m - e for m, e in zip(means, errors)]
                    upper = [m + e for m, e in zip(means, errors)]
                ax.fill_between(ks, lower, upper, color=color, alpha=0.2, zorder=2)

            # Plot baseline selectors as horizontal dashed lines
            for baseline in baseline_selectors:
                if baseline in stats[model][dataset]:
                    # Get value from k=64 (or any k, since baselines are constant)
                    k_data = stats[model][dataset][baseline]
                    if k_data:
                        # Use the first available k value
                        first_k = list(k_data.keys())[0]
                        baseline_value = k_data[first_k]['mean']
                        all_means.append(baseline_value)

                        color = COLORS.get(baseline, '#888888')
                        label = SELECTOR_LABELS.get(baseline, baseline)

                        # Plot as horizontal dashed line spanning the x range
                        ax.axhline(y=baseline_value, color=color, linestyle='--',
                                   linewidth=1.5, label=label, zorder=1, alpha=0.8)

            # Formatting
            ax.set_xscale('log', base=2)
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.set_xlim(k_values[0] * 0.7, k_values[-1] * 1.4)

            # Adaptive Y-axis range based on data
            if all_means:
                data_min = min(all_means) - max(all_errors) if all_errors else min(all_means)
                data_max = max(all_means) + max(all_errors) if all_errors else max(all_means)
                data_spread = data_max - data_min

                # Add padding (10% of spread, minimum 2%)
                padding = max(data_spread * 0.1, 2)
                y_min = data_min - padding
                y_max = data_max + padding

                # Clamp to valid range
                y_min = max(0, y_min)
                y_max = min(100, y_max)

                # Round to nice numbers
                y_min = np.floor(y_min / 2) * 2
                y_max = np.ceil(y_max / 2) * 2

                ax.set_ylim(y_min, y_max)

            # Labels depend on layout
            dataset_label = DATASET_LABELS.get(dataset, dataset)
            model_label = MODEL_LABELS.get(model, model)

            if layout == '3x6':
                # Vertical layout: datasets as columns, models as rows
                if i == 0:
                    ax.set_title(dataset_label, fontsize=11, fontweight='bold', pad=6)
                if j == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                    ax.annotate(model_label, xy=(-0.35, 0.5), xycoords='axes fraction',
                               fontsize=10, fontweight='bold', ha='center', va='center',
                               rotation=90)
            else:
                # Horizontal layout: models as columns, datasets as rows
                if j == 0:
                    ax.set_title(model_label, fontsize=11, fontweight='bold', pad=6)
                if i == 0:
                    ax.set_ylabel('Accuracy (%)', fontsize=9)
                    if n_datasets > 1:
                        ax.annotate(dataset_label, xy=(-0.35, 0.5), xycoords='axes fraction',
                                   fontsize=10, fontweight='bold', ha='center', va='center',
                                   rotation=90)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

    # Common X label
    fig.text(0.5, 0.02, 'Number of Samples per Problem (k)', ha='center', fontsize=11)

    # Legend (only show once, at the top)
    handles, labels = axes[0, 0].get_legend_handles_labels() if layout == '3x6' else axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=True, fancybox=False, edgecolor='black',
               fontsize=9, columnspacing=1.5)

    plt.tight_layout(rect=[0.06, 0.04, 1, 0.94])

    # Save
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.1)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")

    plt.close(fig)


def plot_individual_figures(stats: Dict, output_dir: Path):
    """Generate individual figures for each model-dataset combination."""

    models = sorted(stats.keys())
    datasets = ['aime24', 'aime25', 'gpqa', 'humaneval', 'livecodebench_v5']
    # Remove max-activation, keep 4 selectors
    selectors = ['medoid', 'knn-medoid', 'dbscan-medoid']

    # Determine available k values
    k_values = sorted(list(stats[models[0]][datasets[0]][selectors[0]].keys()))

    output_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(4, 3))

            all_means = []
            all_errors = []

            for selector in selectors:
                if selector not in stats[model][dataset]:
                    continue

                k_data = stats[model][dataset][selector]
                ks = sorted(k_data.keys())
                means = [k_data[k]['mean'] for k in ks]
                errors = [k_data[k]['std'] for k in ks]  # Use std (±1σ)

                all_means.extend(means)
                all_errors.extend(errors)

                color = COLORS[selector]
                marker = MARKERS[selector]
                label = SELECTOR_LABELS[selector]

                # Plot line with markers
                ax.plot(ks, means, color=color, marker=marker,
                       label=label, markerfacecolor='white', markeredgecolor=color,
                       markeredgewidth=1.2, zorder=3)

                # Plot ±1σ shading
                lower = [m - e for m, e in zip(means, errors)]
                upper = [m + e for m, e in zip(means, errors)]
                ax.fill_between(ks, lower, upper, color=color, alpha=0.2, zorder=2)

            # Formatting
            ax.set_xscale('log', base=2)
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.set_xlim(k_values[0] * 0.7, k_values[-1] * 1.4)

            ax.set_xlabel('Number of Samples per Problem (k)')
            ax.set_ylabel('Accuracy (%)')

            # Adaptive Y-axis range based on data
            if all_means:
                data_min = min(all_means) - max(all_errors) if all_errors else min(all_means)
                data_max = max(all_means) + max(all_errors) if all_errors else max(all_means)
                data_spread = data_max - data_min

                # Add padding (10% of spread, minimum 2%)
                padding = max(data_spread * 0.1, 2)
                y_min = data_min - padding
                y_max = data_max + padding

                # Clamp to valid range
                y_min = max(0, y_min)
                y_max = min(100, y_max)

                # Round to nice numbers
                y_min = np.floor(y_min / 2) * 2
                y_max = np.ceil(y_max / 2) * 2

                ax.set_ylim(y_min, y_max)

            # Title
            dataset_label = DATASET_LABELS.get(dataset, dataset)
            model_label = MODEL_LABELS.get(model, model)
            ax.set_title(f'{model_label} - {dataset_label}')

            # Legend
            ax.legend(loc='best', fontsize=7, framealpha=0.9)

            # Grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

            plt.tight_layout()

            # Save
            filename = f'{model}_{dataset}'
            fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight', pad_inches=0.1)
            fig.savefig(output_dir / f'{filename}.png', bbox_inches='tight', pad_inches=0.1)

            plt.close(fig)

    print(f"Saved individual figures to: {output_dir}")


def plot_uncertainty_trend(stats: Dict, output_path: Path):
    """Plot how uncertainty (std) decreases as k increases."""

    models = sorted(stats.keys())
    datasets = ['aime24', 'aime25', 'gpqa', 'humaneval', 'livecodebench_v5']
    selectors = ['medoid', 'knn-medoid', 'dbscan-medoid']

    # Get k values
    k_values = sorted(list(stats[models[0]][datasets[0]][selectors[0]].keys()))

    fig, axes = plt.subplots(3, 6, figsize=(18, 8), sharex=True)

    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]

            for selector in selectors:
                if selector not in stats[model][dataset]:
                    continue

                k_data = stats[model][dataset][selector]
                ks = sorted(k_data.keys())
                stds = [k_data[k]['std'] for k in ks]

                color = COLORS[selector]
                marker = MARKERS[selector]
                label = SELECTOR_LABELS[selector]

                ax.plot(ks, stds, color=color, marker=marker,
                       label=label, markerfacecolor='white', markeredgecolor=color,
                       markeredgewidth=1.2, linewidth=1.5)

            # Formatting
            ax.set_xscale('log', base=2)
            ax.set_xticks(k_values)
            ax.set_xticklabels([str(k) for k in k_values])
            ax.set_xlim(k_values[0] * 0.7, k_values[-1] * 1.4)
            ax.set_ylim(bottom=0)

            # Title (only first row)
            dataset_label = DATASET_LABELS.get(dataset, dataset)
            if i == 0:
                ax.set_title(dataset_label, fontsize=11, fontweight='bold', pad=6)

            # Y label (only first column)
            if j == 0:
                model_label = MODEL_LABELS.get(model, model)
                ax.set_ylabel('Std (%)', fontsize=9)
                ax.annotate(model_label, xy=(-0.35, 0.5), xycoords='axes fraction',
                           fontsize=10, fontweight='bold', ha='center', va='center',
                           rotation=90)

            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)

    # Common X label
    fig.text(0.5, 0.02, 'Number of Samples per Problem (k)', ha='center', fontsize=11)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=True, fancybox=False, edgecolor='black',
               fontsize=9, columnspacing=1.5)

    plt.tight_layout(rect=[0.06, 0.04, 1, 0.94])

    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.1)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")

    plt.close(fig)


def plot_uncertainty_summary(stats: Dict, output_path: Path):
    """Plot average uncertainty across all tasks as k increases - single summary plot."""

    models = sorted(stats.keys())
    datasets = ['aime24', 'aime25', 'gpqa', 'humaneval', 'livecodebench_v5']
    selectors = ['medoid', 'knn-medoid', 'dbscan-medoid']

    # Get k values
    k_values = sorted(list(stats[models[0]][datasets[0]][selectors[0]].keys()))

    fig, ax = plt.subplots(figsize=(8, 5))

    for selector in selectors:
        # Average std across all model-dataset combinations for each k
        avg_stds = []
        std_of_stds = []

        for k in k_values:
            all_stds = []
            for model in models:
                for dataset in datasets:
                    if selector in stats[model][dataset] and k in stats[model][dataset][selector]:
                        all_stds.append(stats[model][dataset][selector][k]['std'])

            if all_stds:
                avg_stds.append(np.mean(all_stds))
                std_of_stds.append(np.std(all_stds))
            else:
                avg_stds.append(0)
                std_of_stds.append(0)

        color = COLORS[selector]
        marker = MARKERS[selector]
        label = SELECTOR_LABELS[selector]

        ax.plot(k_values, avg_stds, color=color, marker=marker,
               label=label, markerfacecolor='white', markeredgecolor=color,
               markeredgewidth=1.5, linewidth=2, markersize=8)

        # Shading for std of stds
        lower = [a - s for a, s in zip(avg_stds, std_of_stds)]
        upper = [a + s for a, s in zip(avg_stds, std_of_stds)]
        ax.fill_between(k_values, lower, upper, color=color, alpha=0.15)

    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlim(k_values[0] * 0.7, k_values[-1] * 1.4)
    ax.set_ylim(bottom=0)

    ax.set_xlabel('Number of Samples per Problem (k)', fontsize=12)
    ax.set_ylabel('Sampling Uncertainty (Std %)', fontsize=12)
    ax.set_title('Uncertainty Decreases as Sample Size Increases', fontsize=13, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add annotation
    ax.annotate('k=64: All samples selected\n(No sampling uncertainty)',
                xy=(64, 0.5), xytext=(32, 3),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.tight_layout()

    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.1)
    fig.savefig(output_path.with_suffix('.png'), bbox_inches='tight', pad_inches=0.1)
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved: {output_path.with_suffix('.png')}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot downsample ablation results')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: results_dir/figures/downsample_ablation)')
    parser.add_argument('--layout', type=str, default='3x6', choices=['3x6', '6x3'],
                       help='Subplot layout')
    parser.add_argument('--individual', action='store_true',
                       help='Also generate individual figures')
    parser.add_argument('--datasets', type=str, default=None,
                       help='Comma-separated list of datasets to include (e.g., aime24,gpqa)')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / 'figures' / 'downsample_ablation'

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    data = load_results(results_dir)

    print("Computing statistics...")
    stats = compute_statistics(data)

    # Parse datasets if specified
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = None  # Use default

    # Generate 3 figures with different error types
    error_types = {
        'std': 'Standard Deviation (±1σ)',
        'ci95': '95% Confidence Interval',
        'minmax': 'Min-Max Range',
    }

    for error_type, error_label in error_types.items():
        print(f"Generating figure with {error_label}...")
        output_file = output_path.parent / f'downsample_ablation_{error_type}'
        plot_figure(stats, output_file, args.layout, error_type, datasets)

    # Generate uncertainty trend plots
    print("Generating uncertainty trend plot (18 subplots)...")
    plot_uncertainty_trend(stats, output_path.parent / 'uncertainty_trend')

    print("Generating uncertainty summary plot (single figure)...")
    plot_uncertainty_summary(stats, output_path.parent / 'uncertainty_summary')

    if args.individual:
        print("Generating individual figures...")
        plot_individual_figures(stats, output_path.parent / 'individual')

    print("Done!")


if __name__ == '__main__':
    main()
