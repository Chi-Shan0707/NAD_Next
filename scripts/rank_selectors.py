#!/usr/bin/env python3
"""
NAD Selector Ranking CLI

Computes multi-task selector rankings across all model×dataset combinations.
Reads actual problem counts from meta.json files (never hardcoded).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nad.ops.selector_ranking import (
    compute_global_scores,
    bootstrap_score_ci,
    load_task_data,
    format_ranking_table,
    calculate_selector_consistency,
    calculate_complementarity_matrix,
    calculate_dataset_specialization,
    calculate_problem_difficulty,
    filter_tasks_by_category,
    get_category_stats,
    TASK_CATEGORIES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Dataset problem counts - will be loaded from meta.json
DATASET_INFO = {
    # These are just display names and expected ranges for validation
    "aime24": "AIME 2024 math problems",
    "aime25": "AIME 2025 math problems",
    "gpqa": "Graduate-level science Q&A",
    "humaneval": "Code evaluation benchmark",
    "livecodebench": "Live coding benchmark",
    "mbpp": "Mostly Basic Python Problems",
}

def load_all_results(results_dir: Path, load_per_problem: bool = False, model_filter: str = "all", analysis_mode: str = "full") -> Tuple[pd.DataFrame, Dict]:
    """
    Load all accuracy results from results_all_models directory structure.
    Dynamically reads problem counts from meta.json files.

    Args:
        results_dir: Path to results directory
        load_per_problem: Whether to load per-problem data
        model_filter: Filter for model names ("llama", "qwen", "deepseek", or "all")
        analysis_mode: Which results to load ("full", "positions", or "all")
    """
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    rows = []
    per_problem_data = {}
    tasks_loaded = 0
    tasks_failed = 0

    # Check what types of data are available across all models
    available_data_types = set()
    for model_name, model_data in summary.get("models", {}).items():
        for dataset_name, dataset_data in model_data.items():
            if "full_sequence" in dataset_data:
                available_data_types.add("full")
            if any(key in dataset_data for key in ["window_0-1", "window_0-2", "window_0-8"]):
                available_data_types.add("positions")

    # Log available data types
    if available_data_types:
        if "full" in available_data_types and "positions" in available_data_types:
            logger.info("Available data: Full sequences and position windows")
        elif "full" in available_data_types:
            logger.info("Available data: Full sequences only")
        elif "positions" in available_data_types:
            logger.info("Available data: Position windows only")
    else:
        logger.warning("No selector accuracy data found in summary")

    for model_name, model_data in summary.get("models", {}).items():
        # Apply model filter
        if model_filter != "all":
            model_lower = model_name.lower()
            # Support comma-separated list of filters
            filters = [f.strip().lower() for f in model_filter.split(',')]

            # Check if any of the filters match
            match_found = False
            for filter_term in filters:
                if filter_term in model_lower:
                    match_found = True
                    break

            if not match_found:
                continue

        for dataset_name, dataset_data in model_data.items():
            if dataset_data.get("status") != "success":
                logger.warning(f"Skipping failed task: {model_name}/{dataset_name}")
                tasks_failed += 1
                continue

            # Get cache directory - use relative location
            cache_name = dataset_data.get("cache", "")
            # Construct path to actual cache location (relative path)
            actual_cache_path = Path("./MUI_public/cache") / model_name / dataset_name / cache_name

            # Also try local path as fallback
            local_cache_path = results_dir / model_name / dataset_name / cache_name

            # Load actual problem count from meta.json (NOT hardcoded!)
            try:
                # Try actual cache location first
                if actual_cache_path.exists():
                    cache_path = actual_cache_path
                elif local_cache_path.exists():
                    cache_path = local_cache_path
                else:
                    raise FileNotFoundError(f"Cache not found in either {actual_cache_path} or {local_cache_path}")

                task_data = load_task_data(cache_path, results_path=results_dir)
                n = task_data["n"]

                if n == 0:
                    logger.warning(f"Task {model_name}/{dataset_name} has 0 problems, skipping")
                    continue

                logger.debug(f"Loaded {model_name}/{dataset_name}: n={n} problems from {cache_path}")

                # Store per-problem data if available
                if load_per_problem:
                    # Load accuracy file from results directory instead of cache
                    accuracy_path = results_dir / model_name / dataset_name / cache_name / "accuracy_full_sequence.json"
                    if accuracy_path.exists():
                        with open(accuracy_path) as f:
                            acc_data = json.load(f)
                        if "per_problem" in acc_data:
                            task_id = f"{model_name}/{dataset_name}"
                            per_problem_data[task_id] = acc_data["per_problem"]
                            logger.debug(f"Loaded per-problem data for {task_id}")
                    elif "per_problem" in task_data and task_data["per_problem"]:
                        task_id = f"{model_name}/{dataset_name}"
                        per_problem_data[task_id] = task_data["per_problem"]

            except Exception as e:
                logger.warning(f"Could not load meta.json for {model_name}/{dataset_name}: {e}")
                # Try to continue with summary data only, but warn about missing n
                n = 100  # Fallback, but will log warning
                logger.warning(f"Using fallback n={n} for {model_name}/{dataset_name}")

            # Load position window data if not in summary (for backward compatibility)
            # Check if position data is missing from summary but files exist
            needs_position_data = any(window not in dataset_data for window in ["window_0-1", "window_0-2", "window_0-8"])

            if needs_position_data:
                result_path = results_dir / model_name / dataset_name / cache_name
                if result_path.exists():
                    # Check for position window accuracy files
                    position_data_loaded = []
                    for window_name in ["window_0-1", "window_0-2", "window_0-8"]:
                        if window_name not in dataset_data:  # Only load if missing
                            acc_file = result_path / f"accuracy_{window_name}.json"
                            if acc_file.exists():
                                try:
                                    with open(acc_file) as f:
                                        acc_data = json.load(f)
                                    # Try both field names for compatibility
                                    dataset_data[window_name] = acc_data.get("selector_accuracy", acc_data.get("selector_accuracies", {}))
                                    position_data_loaded.append(window_name)
                                except Exception as e:
                                    logger.warning(f"Could not load {acc_file}: {e}")

                    if position_data_loaded:
                        logger.debug(f"Loaded position window data from files for {model_name}/{dataset_name}: {', '.join(position_data_loaded)}")

            # Get selector accuracies based on analysis mode
            selectors_to_process = []

            # Track what data types are available
            has_full_data = "full_sequence" in dataset_data and dataset_data["full_sequence"]
            has_position_data = any(key in dataset_data and dataset_data[key]
                                   for key in ["window_0-1", "window_0-2", "window_0-8"])

            # Log data availability for this task
            if has_full_data and has_position_data:
                logger.debug(f"Data available for {model_name}/{dataset_name}: Full sequence and position windows")
            elif has_full_data:
                logger.debug(f"Data available for {model_name}/{dataset_name}: Full sequence only")
            elif has_position_data:
                logger.debug(f"Data available for {model_name}/{dataset_name}: Position windows only")
            else:
                logger.debug(f"No selector accuracy data found for {model_name}/{dataset_name}")

            if analysis_mode in ["full", "all"]:
                # Load full sequence results
                full_seq = dataset_data.get("full_sequence", {})
                if full_seq:
                    for selector, accuracy in full_seq.items():
                        if accuracy > 1.5:
                            accuracy = accuracy / 100.0
                        selectors_to_process.append((f"{model_name}/{dataset_name}", "full", selector, accuracy, n))

            if analysis_mode in ["positions", "all"]:
                # Load position window results
                for window_key, window_name in [("window_0-1", "window_0-1"),
                                                ("window_0-2", "window_0-2"),
                                                ("window_0-8", "window_0-8")]:
                    window_data = dataset_data.get(window_key, {})
                    if window_data:
                        for selector, accuracy in window_data.items():
                            if accuracy > 1.5:
                                accuracy = accuracy / 100.0
                            selectors_to_process.append((f"{model_name}/{dataset_name}", window_name, selector, accuracy, n))

            # If we're in full mode and have data, show rankings
            if analysis_mode == "full" and selectors_to_process:
                task_id = f"{model_name}/{dataset_name}"
                sorted_selectors = [(sel, acc) for _, _, sel, acc, _ in selectors_to_process]
                sorted_selectors.sort(key=lambda x: x[1], reverse=True)

                # Display top selectors for this task
                if sorted_selectors:
                    # Show full ranking if verbose
                    if logger.getEffectiveLevel() <= logging.DEBUG:
                        logger.debug(f"\n  Selector rankings for {task_id} (n={n} problems):")
                        logger.debug("  " + "─" * 60)
                        logger.debug(f"  {'Rank':<6} {'Selector':<20} {'Accuracy':<12} {'Status'}")
                        logger.debug("  " + "─" * 60)
                        for i, (sel, acc) in enumerate(sorted_selectors, 1):
                            marker = "⭐ TOP" if i <= 3 else ""
                            rank_str = f"#{i}"
                            logger.debug(f"  {rank_str:<6} {sel:<20} {acc:>8.4f}     {marker}")
                        logger.debug("  " + "─" * 60)
                    else:
                        # Show top 3 in compact form
                        top_3 = sorted_selectors[:3]
                        top_str = ", ".join([f"{sel}:{acc:.3f}" for sel, acc in top_3])
                        logger.info(f"  Top selectors for {task_id}: {top_str}")

            # Add all results to rows
            for task_id, window_type, selector, accuracy, n_problems in selectors_to_process:
                if window_type != "full":
                    # For position windows, append window type to task_id
                    task_id_with_window = f"{task_id}/{window_type}"
                else:
                    task_id_with_window = task_id

                rows.append({
                    "task_id": task_id_with_window,
                    "selector": selector,
                    "accuracy": accuracy,
                    "n": n_problems,
                    "model": model_name,
                    "dataset": dataset_name,
                    "cache": cache_name,
                    "window": window_type
                })

            if selectors_to_process:
                tasks_loaded += 1

    logger.info(f"Loaded {tasks_loaded} tasks successfully, {tasks_failed} failed")

    if not rows:
        # Provide helpful error message based on analysis mode
        if analysis_mode == "positions":
            error_msg = (
                "No position window data found in results directory.\n"
                "Position analysis requires that step2.2_batch_analyze.sh was run with 'positions' or 'all' mode.\n\n"
                "To generate position data:\n"
                "  ./step2.2_batch_analyze.sh positions    # For position windows only\n"
                "  ./step2.2_batch_analyze.sh all          # For both full and position data\n\n"
                "Or use --analysis-mode full to analyze existing full sequence data."
            )
        elif analysis_mode == "all":
            error_msg = (
                "No data found for 'all' mode analysis.\n"
                "This mode requires both full sequence and position window data.\n\n"
                "Please run:\n"
                "  ./step2.2_batch_analyze.sh all\n\n"
                "Or use --analysis-mode full to analyze only full sequence data."
            )
        else:
            error_msg = (
                "No data loaded from results directory.\n"
                "Please ensure step2.2_batch_analyze.sh has been run successfully."
            )
        raise ValueError(error_msg)

    return pd.DataFrame(rows), per_problem_data

def main():
    parser = argparse.ArgumentParser(
        description="Compute multi-task selector rankings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data source
    parser.add_argument(
        "--results-dir",
        type=Path,
        default="results_all_models",
        help="Directory containing analysis results"
    )

    # Task filtering
    parser.add_argument(
        "--min-spread",
        type=float,
        default=0.005,
        help="Minimum accuracy spread to include task"
    )

    # Score weights
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for RNS in final score"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Weight for (1-regret) in final score"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="Weight for Copeland in final score"
    )

    # Task weighting
    parser.add_argument(
        "--task-weight-mode",
        choices=["sqrt", "micro", "macro"],
        default="sqrt",
        help="How to weight tasks: sqrt(n), n, or 1"
    )

    # Coverage settings
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.6,
        help="Minimum coverage for MAIN tier"
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=10,
        help="Minimum tasks for MAIN tier"
    )
    parser.add_argument(
        "--coverage-penalty",
        type=float,
        default=0.0,
        help="Coverage penalty lambda (0=no penalty)"
    )

    # Bootstrap settings
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Number of bootstrap samples for CI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap CI computation"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save JSON report"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save CSV report"
    )

    # Display
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    # Enhanced metrics
    parser.add_argument(
        "--consistency-analysis",
        action="store_true",
        help="Include selector consistency analysis"
    )
    parser.add_argument(
        "--complementarity-analysis",
        action="store_true",
        help="Include selector complementarity analysis"
    )
    parser.add_argument(
        "--dataset-specialization",
        action="store_true",
        help="Include dataset specialization analysis"
    )
    parser.add_argument(
        "--problem-difficulty",
        action="store_true",
        help="Include problem difficulty analysis"
    )

    # Category-specific rankings
    parser.add_argument(
        "--compute-categories",
        action="store_true",
        help="Compute category-specific sub-rankings (programming, math_science)"
    )

    # Position window-specific rankings
    parser.add_argument(
        "--separate-windows",
        action="store_true",
        help="Compute separate rankings for each position window (0-1, 0-2, 0-8)"
    )

    # Model filtering
    parser.add_argument(
        "--model-filter",
        type=str,
        default="all",
        help="Filter models to analyze: comma-separated list (e.g., 'qwen,deepseek') or 'all' (default: all)"
    )

    # Analysis mode (full sequence vs positions)
    parser.add_argument(
        "--analysis-mode",
        type=str,
        default="full",
        choices=["full", "positions", "all"],
        help="Analysis mode: full (full sequence only), positions (position windows only), all (both)"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate score weights
    weight_sum = args.alpha + args.beta + args.gamma
    if abs(weight_sum - 1.0) > 0.001:
        logger.warning(f"Score weights sum to {weight_sum:.3f}, not 1.0")

    # Load data
    logger.info(f"Loading results from {args.results_dir}")

    # Log model filter if applied
    if args.model_filter != "all":
        logger.info(f"Filtering models: {args.model_filter}")

    # Log analysis mode if not default
    if args.analysis_mode != "full":
        logger.info(f"Analysis mode: {args.analysis_mode}")

    # Check if any enhanced metrics are requested
    need_per_problem = (args.consistency_analysis or args.complementarity_analysis or
                       args.dataset_specialization or args.problem_difficulty)
    df_input, per_problem_data = load_all_results(
        args.results_dir,
        load_per_problem=need_per_problem,
        model_filter=args.model_filter,
        analysis_mode=args.analysis_mode
    )

    # Get unique tasks and selectors
    unique_tasks = df_input["task_id"].unique()
    unique_selectors = df_input["selector"].unique()

    logger.info(f"Found {len(unique_tasks)} tasks, {len(unique_selectors)} selectors")

    # Store all rankings (global + categories if requested)
    all_rankings = {}
    all_task_details = {}
    all_ci_results = {}
    all_diff_results = {}
    category_stats = {}

    # Compute global scores
    logger.info("Computing global rankings...")
    df_scores, df_task_details = compute_global_scores(
        df_input,
        min_spread=args.min_spread,
        weight_mode=args.task_weight_mode,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        min_coverage=args.min_coverage,
        min_tasks=args.min_tasks,
        coverage_penalty_lambda=args.coverage_penalty,
    )

    all_rankings['global'] = df_scores
    all_task_details['global'] = df_task_details

    # Compute category-specific rankings if requested
    if args.compute_categories:
        for category_name in TASK_CATEGORIES.keys():
            logger.info(f"Computing {category_name} category rankings...")

            # Filter data for this category
            df_category = filter_tasks_by_category(df_input, category_name)

            if df_category.empty or df_category['task_id'].nunique() < 2:
                logger.warning(f"Insufficient data for {category_name} category (needs at least 2 tasks)")
                continue

            # Get category statistics
            category_stats[category_name] = get_category_stats(df_category, category_name)

            # Compute scores for this category
            df_cat_scores, df_cat_details = compute_global_scores(
                df_category,
                min_spread=args.min_spread,
                weight_mode=args.task_weight_mode,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                min_coverage=args.min_coverage,
                min_tasks=min(args.min_tasks, df_category['task_id'].nunique()),  # Adjust min_tasks for category
                coverage_penalty_lambda=args.coverage_penalty,
            )

            all_rankings[category_name] = df_cat_scores
            all_task_details[category_name] = df_cat_details

    # Compute window-specific rankings if requested
    if args.separate_windows and args.analysis_mode in ["positions", "all"]:
        logger.info(f"Computing separate window rankings (separate_windows={args.separate_windows}, analysis_mode={args.analysis_mode})")

        # Check for con64@ and avg64@ in position mode and warn
        position_incompatible_selectors = ['con64@', 'avg64@']
        found_incompatible = [s for s in position_incompatible_selectors if s in df_input['selector'].unique()]
        if found_incompatible:
            logger.warning(f"⚠️  Note: Selectors {found_incompatible} require full sequence context and cannot be implemented in position mode")
            logger.warning(f"   These selectors will show as placeholders in position window results")

        # List of windows to analyze
        windows = ["window_0-1", "window_0-2", "window_0-8"]

        for window_name in windows:
            # Filter data for this specific window
            logger.debug(f"Filtering for {window_name}...")
            df_window = df_input[df_input['window'] == window_name]
            logger.debug(f"Filtered {window_name}: {df_window.shape[0]} rows, {df_window['task_id'].nunique()} tasks")

            if df_window.empty or df_window['task_id'].nunique() < 2:
                logger.warning(f"Insufficient data for {window_name} (needs at least 2 tasks)")
                continue

            logger.info(f"Computing {window_name} rankings...")

            # Get window statistics
            window_stats = {
                'tasks_count': df_window['task_id'].nunique(),
                'models_count': df_window['model'].nunique() if 'model' in df_window.columns else 0,
                'datasets': df_window['dataset'].unique().tolist() if 'dataset' in df_window.columns else [],
                'mean_accuracy': df_window.groupby('selector')['accuracy'].mean().mean(),
                'median_accuracy': df_window.groupby('selector')['accuracy'].median().median(),
            }
            category_stats[window_name] = window_stats

            # Compute scores for this window
            df_win_scores, df_win_details = compute_global_scores(
                df_window,
                min_spread=args.min_spread,
                weight_mode=args.task_weight_mode,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                min_coverage=args.min_coverage,
                min_tasks=min(args.min_tasks, df_window['task_id'].nunique()),
                coverage_penalty_lambda=args.coverage_penalty,
            )

            all_rankings[window_name] = df_win_scores
            all_task_details[window_name] = df_win_details

            # Also compute category-specific rankings for this window
            if args.compute_categories:
                for category_name in TASK_CATEGORIES.keys():
                    # Filter by both window and category
                    df_window_cat = filter_tasks_by_category(df_window, category_name)

                    if df_window_cat.empty or df_window_cat['task_id'].nunique() < 2:
                        logger.warning(f"Insufficient data for {window_name}/{category_name} (needs at least 2 tasks)")
                        continue

                    logger.info(f"Computing {window_name}/{category_name} rankings...")

                    # Create combined key for this window+category combination
                    combined_key = f"{window_name}_{category_name}"

                    # Get statistics for this combination
                    window_cat_stats = {
                        'tasks_count': df_window_cat['task_id'].nunique(),
                        'models_count': df_window_cat['model'].nunique() if 'model' in df_window_cat.columns else 0,
                        'datasets': df_window_cat['dataset'].unique().tolist() if 'dataset' in df_window_cat.columns else [],
                        'mean_accuracy': df_window_cat.groupby('selector')['accuracy'].mean().mean(),
                        'median_accuracy': df_window_cat.groupby('selector')['accuracy'].median().median(),
                    }
                    category_stats[combined_key] = window_cat_stats

                    # Compute scores for this window+category combination
                    df_win_cat_scores, df_win_cat_details = compute_global_scores(
                        df_window_cat,
                        min_spread=args.min_spread,
                        weight_mode=args.task_weight_mode,
                        alpha=args.alpha,
                        beta=args.beta,
                        gamma=args.gamma,
                        min_coverage=args.min_coverage,
                        min_tasks=min(args.min_tasks, df_window_cat['task_id'].nunique()),
                        coverage_penalty_lambda=args.coverage_penalty,
                    )

                    all_rankings[combined_key] = df_win_cat_scores
                    all_task_details[combined_key] = df_win_cat_details

    # For backward compatibility, use global as default
    df_scores = all_rankings['global']
    df_task_details = all_task_details['global']

    # Bootstrap CI if requested
    df_ci = None
    P_diff = None

    if not args.no_bootstrap:
        logger.info(f"Computing bootstrap CI with B={args.bootstrap_samples} samples...")
        df_ci, P_diff = bootstrap_score_ci(
            df_input,
            B=args.bootstrap_samples,
            seed=args.seed,
            min_spread=args.min_spread,
            weight_mode=args.task_weight_mode,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            min_coverage=args.min_coverage,
            min_tasks=args.min_tasks,
            coverage_penalty_lambda=args.coverage_penalty,
        )
        all_ci_results['global'] = df_ci
        all_diff_results['global'] = P_diff

        # Bootstrap for categories if requested
        if args.compute_categories:
            for category_name in TASK_CATEGORIES.keys():
                if category_name not in all_rankings:
                    continue

                logger.info(f"Computing bootstrap CI for {category_name} category...")
                df_category = filter_tasks_by_category(df_input, category_name)

                df_cat_ci, P_cat_diff = bootstrap_score_ci(
                    df_category,
                    B=args.bootstrap_samples,
                    seed=args.seed,
                    min_spread=args.min_spread,
                    weight_mode=args.task_weight_mode,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    min_coverage=args.min_coverage,
                    min_tasks=min(args.min_tasks, df_category['task_id'].nunique()),
                    coverage_penalty_lambda=args.coverage_penalty,
                )
                all_ci_results[category_name] = df_cat_ci
                all_diff_results[category_name] = P_cat_diff

        # Bootstrap for windows if requested
        if args.separate_windows and args.analysis_mode in ["positions", "all"]:
            windows = ["window_0-1", "window_0-2", "window_0-8"]
            for window_name in windows:
                if window_name not in all_rankings:
                    continue

                logger.info(f"Computing bootstrap CI for {window_name}...")
                df_window = df_input[df_input['window'] == window_name]

                df_win_ci, P_win_diff = bootstrap_score_ci(
                    df_window,
                    B=args.bootstrap_samples,
                    seed=args.seed,
                    min_spread=args.min_spread,
                    weight_mode=args.task_weight_mode,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    min_coverage=args.min_coverage,
                    min_tasks=min(args.min_tasks, df_window['task_id'].nunique()),
                    coverage_penalty_lambda=args.coverage_penalty,
                )
                all_ci_results[window_name] = df_win_ci
                all_diff_results[window_name] = P_win_diff

                # Bootstrap for window+category combinations if requested
                if args.compute_categories:
                    for category_name in TASK_CATEGORIES.keys():
                        combined_key = f"{window_name}_{category_name}"
                        if combined_key not in all_rankings:
                            continue

                        logger.info(f"Computing bootstrap CI for {combined_key}...")
                        df_window_cat = filter_tasks_by_category(df_window, category_name)

                        df_win_cat_ci, P_win_cat_diff = bootstrap_score_ci(
                            df_window_cat,
                            B=args.bootstrap_samples,
                            seed=args.seed,
                            min_spread=args.min_spread,
                            weight_mode=args.task_weight_mode,
                            alpha=args.alpha,
                            beta=args.beta,
                            gamma=args.gamma,
                            min_coverage=args.min_coverage,
                            min_tasks=min(args.min_tasks, df_window_cat['task_id'].nunique()),
                            coverage_penalty_lambda=args.coverage_penalty,
                        )
                        all_ci_results[combined_key] = df_win_cat_ci
                        all_diff_results[combined_key] = P_win_cat_diff

        # Merge CI into scores
        if df_ci is not None and not df_ci.empty:
            df_scores = df_scores.merge(
                df_ci[["selector", "Score_ci_lower", "Score_ci_upper"]],
                on="selector",
                how="left"
            )

    # Print results
    print("\n" + "="*80)
    print(" " * 20 + "NAD Selector Multi-Task Ranking Report")
    print("="*80)

    # Configuration
    print("\nConfiguration:")
    print(f"  Tasks analyzed: {len(df_task_details['task_id'].unique())}/{len(unique_tasks)}")

    # Show model filter if applied
    if args.model_filter != "all":
        print(f"  Model filter: {args.model_filter}")

    # Show excluded tasks
    analyzed_tasks = set(df_task_details['task_id'].unique())
    excluded_tasks = set(unique_tasks) - analyzed_tasks
    if excluded_tasks:
        print(f"  Excluded tasks (spread ≤ {args.min_spread}):")
        for task in sorted(excluded_tasks):
            print(f"    - {task}")

    print(f"  Score weights: α={args.alpha} (RNS), β={args.beta} (1-regret), γ={args.gamma} (Copeland)")
    print(f"  Task weights: {args.task_weight_mode}")

    if not args.no_bootstrap:
        print(f"  Bootstrap: B={args.bootstrap_samples}, seed={args.seed}")

    # Function to print a ranking table section
    def print_ranking_section(category_key, df_cat_scores, df_cat_ci=None, title_suffix=""):
        """Helper function to print a ranking section."""
        print("\n" + "="*80)

        # Check if we're in a window context and need to mark unimplementable selectors
        is_window_context = category_key.startswith('window_')
        if is_window_context:
            # Create a copy to avoid modifying the original
            df_cat_scores = df_cat_scores.copy()
            # Mark selectors that cannot be implemented in position mode
            position_incompatible_selectors = ['con64@', 'avg64@']
            df_cat_scores['selector'] = df_cat_scores['selector'].apply(
                lambda x: f"{x} (*)" if x in position_incompatible_selectors else x
            )

        if category_key == 'global':
            tasks_in_category = len(all_task_details[category_key]['task_id'].unique())
            print(f"GLOBAL RANKINGS (All {tasks_in_category} Tasks){title_suffix}:")
        elif category_key.startswith('window_'):
            # Handle window-specific rankings
            stats = category_stats.get(category_key, {})
            tasks_in_category = stats.get('tasks_count', 0)

            # Check if this is a window+category combination
            if '_programming' in category_key or '_math_science' in category_key:
                # Extract window and category parts
                # Handle cases like "window_0-1_programming"
                if '_programming' in category_key:
                    window_part = category_key.replace('_programming', '')
                    cat_part = 'programming'
                elif '_math_science' in category_key:
                    window_part = category_key.replace('_math_science', '')
                    cat_part = 'math_science'
                else:
                    parts = category_key.rsplit('_', 1)  # Split from the right
                    window_part = parts[0]
                    cat_part = parts[1] if len(parts) > 1 else ""

                window_label = window_part.replace('window_', 'Position ').replace('-', ' to ')
                if cat_part == 'programming':
                    cat_label = 'Programming'
                elif cat_part == 'math_science':
                    cat_label = 'Math/Science'
                else:
                    cat_label = cat_part.replace('_', '/')

                datasets_str = ', '.join(stats.get('datasets', []))
                print(f"{window_label.upper()} - {cat_label.upper()} RANKINGS ({tasks_in_category} Tasks: {datasets_str}){title_suffix}:")
            else:
                # Regular window ranking
                window_label = category_key.replace('window_', 'Position ').replace('-', ' to ')
                print(f"{window_label.upper()} RANKINGS ({tasks_in_category} Tasks){title_suffix}:")
        else:
            stats = category_stats.get(category_key, {})
            datasets_str = ', '.join(stats.get('datasets', []))
            tasks_in_category = stats.get('tasks', 0)
            print(f"{category_key.upper().replace('_', '/')} RANKINGS ({tasks_in_category} Tasks: {datasets_str}){title_suffix}:")

        print("─"*80)

        # Format table
        show_ci = df_cat_ci is not None and "Score_ci_lower" in df_cat_scores.columns
        table_str = format_ranking_table(df_cat_scores, show_ci=show_ci)
        print(table_str)

        # Add footnote if we marked any selectors
        if is_window_context and any(df_cat_scores['selector'].str.contains('(*)', regex=False, na=False)):
            print("\n(*) Not implementable in position mode - requires full sequence context")

    # Print global rankings
    print_ranking_section('global', df_scores, df_ci)

    # Print category-specific rankings if computed
    if args.compute_categories:
        for category_name in ['programming', 'math_science']:
            if category_name in all_rankings:
                cat_ci = all_ci_results.get(category_name) if not args.no_bootstrap else None
                if cat_ci is not None and not cat_ci.empty:
                    # Merge CI into category scores
                    cat_scores_with_ci = all_rankings[category_name].merge(
                        cat_ci[["selector", "Score_ci_lower", "Score_ci_upper"]],
                        on="selector",
                        how="left"
                    )
                else:
                    cat_scores_with_ci = all_rankings[category_name]

                print_ranking_section(category_name, cat_scores_with_ci, cat_ci)

    # Print window-specific rankings if computed
    if args.separate_windows and args.analysis_mode in ["positions", "all"]:
        windows = ["window_0-1", "window_0-2", "window_0-8"]
        for window_name in windows:
            if window_name in all_rankings:
                win_ci = all_ci_results.get(window_name) if not args.no_bootstrap else None
                if win_ci is not None and not win_ci.empty:
                    # Merge CI into window scores
                    win_scores_with_ci = all_rankings[window_name].merge(
                        win_ci[["selector", "Score_ci_lower", "Score_ci_upper"]],
                        on="selector",
                        how="left"
                    )
                else:
                    win_scores_with_ci = all_rankings[window_name]

                print_ranking_section(window_name, win_scores_with_ci, win_ci)

                # Print category-specific rankings for this window
                if args.compute_categories:
                    for category_name in ['programming', 'math_science']:
                        combined_key = f"{window_name}_{category_name}"
                        if combined_key in all_rankings:
                            win_cat_ci = all_ci_results.get(combined_key) if not args.no_bootstrap else None
                            if win_cat_ci is not None and not win_cat_ci.empty:
                                # Merge CI into window+category scores
                                win_cat_scores_with_ci = all_rankings[combined_key].merge(
                                    win_cat_ci[["selector", "Score_ci_lower", "Score_ci_upper"]],
                                    on="selector",
                                    how="left"
                                )
                            else:
                                win_cat_scores_with_ci = all_rankings[combined_key]

                            print_ranking_section(combined_key, win_cat_scores_with_ci, win_cat_ci)

    # Pairwise comparisons (top differences)
    if P_diff is not None and not P_diff.empty:
        print("\n" + "="*80)
        print("Pairwise Comparisons (Top Differences):")
        print("─"*80)
        print(f"{'Comparison':<30} {'Δ_mean':<10} {'95% CI(Δ)':<20} {'P(s1>s2)':<10}")
        print("─"*80)

        # Sort by absolute difference
        P_diff["abs_diff"] = P_diff["mean_diff"].abs()
        P_diff_sorted = P_diff.sort_values("abs_diff", ascending=False).head(10)

        for _, row in P_diff_sorted.iterrows():
            ci_str = f"[{row['diff_ci_lower']:.3f}, {row['diff_ci_upper']:.3f}]"
            print(f"{row['comparison']:<30} {row['mean_diff']:<10.3f} {ci_str:<20} {row['prob_greater']:<10.3f}")

    # Per-task summary
    print("\n" + "="*80)
    print("Per-Task Summary (Problem counts from meta.json):")
    print("─"*80)
    print(f"{'Task':<40} {'n':<6} {'Weight':<8} {'Best':<18} {'Acc':<8}")
    print("─"*80)

    task_summary = []
    for tid, g in df_task_details.groupby("task_id"):
        best_idx = g["accuracy"].idxmax()
        best_sel = g.loc[best_idx, "selector"]
        best_acc = g.loc[best_idx, "accuracy"]
        n = g.iloc[0]["n"]
        weight = np.sqrt(n) if args.task_weight_mode == "sqrt" else (n if args.task_weight_mode == "micro" else 1.0)

        task_summary.append({
            "task": tid,
            "n": n,
            "weight": weight,
            "best_selector": best_sel,
            "best_accuracy": best_acc
        })

    task_summary_df = pd.DataFrame(task_summary).sort_values("n", ascending=False)
    for _, row in task_summary_df.head(15).iterrows():
        task_short = row['task']
        if len(task_short) > 40:
            task_short = task_short[:37] + "..."
        print(f"{task_short:<40} {row['n']:<6d} {row['weight']:<8.2f} {row['best_selector']:<18} {row['best_accuracy']*100:<8.1f}%")

    print("="*80)

    # Enhanced metrics analysis if requested
    if args.consistency_analysis:
        print("\n" + "="*80)
        print("Selector Consistency Analysis:")
        print("─"*80)
        df_consistency = calculate_selector_consistency(df_task_details)
        print(f"{'Selector':<18} {'Mean Acc':<10} {'Std Dev':<10} {'CV':<8} {'Consistency':<12} {'Range':<10}")
        print("─"*80)
        for _, row in df_consistency.sort_values('consistency_score', ascending=False).iterrows():
            print(f"{row['selector']:<18} {row['mean_accuracy']:<10.3f} {row['std_accuracy']:<10.3f} "
                  f"{row['cv']:<8.3f} {row['consistency_score']:<12.3f} {row['range']:<10.3f}")

    if args.dataset_specialization:
        print("\n" + "="*80)
        print("Dataset Specialization Analysis:")
        print("─"*80)
        df_spec = calculate_dataset_specialization(df_task_details)
        if not df_spec.empty:
            print(f"{'Selector':<18} {'Specialization':<15} {'Best Dataset':<15} {'Best Acc':<10} {'Worst Dataset':<15} {'Worst Acc':<10}")
            print("─"*80)
            for _, row in df_spec.sort_values('specialization_score', ascending=False).head(10).iterrows():
                print(f"{row['selector']:<18} {row['specialization_score']:<15.3f} {row['best_dataset']:<15} "
                      f"{row['best_dataset_acc']:<10.3f} {row['worst_dataset']:<15} {row['worst_dataset_acc']:<10.3f}")

    if args.complementarity_analysis and per_problem_data:
        print("\n" + "="*80)
        print("Selector Complementarity Analysis (Top 10 Pairs):")
        print("─"*80)
        df_comp = calculate_complementarity_matrix(per_problem_data)
        # Find top complementary pairs
        comp_pairs = []
        for i in range(len(df_comp.index)):
            for j in range(i+1, len(df_comp.columns)):
                comp_pairs.append({
                    'pair': f"{df_comp.index[i]} + {df_comp.columns[j]}",
                    'complementarity': df_comp.iloc[i, j] + df_comp.iloc[j, i]
                })
        comp_pairs_df = pd.DataFrame(comp_pairs).sort_values('complementarity', ascending=False)
        print(f"{'Selector Pair':<50} {'Complementarity Score':<20}")
        print("─"*80)
        for _, row in comp_pairs_df.head(10).iterrows():
            print(f"{row['pair']:<50} {row['complementarity']:<20.3f}")

    if args.problem_difficulty and per_problem_data:
        print("\n" + "="*80)
        print("Problem Difficulty Analysis:")
        print("─"*80)
        difficulty_scores = calculate_problem_difficulty(per_problem_data)
        # Show distribution
        difficulties = list(difficulty_scores.values())
        if difficulties:
            print(f"  Total problems analyzed: {len(difficulties)}")
            print(f"  Average difficulty: {np.mean(difficulties):.3f}")
            print(f"  Difficulty range: [{np.min(difficulties):.3f}, {np.max(difficulties):.3f}]")
            print(f"  Hard problems (>0.8): {sum(1 for d in difficulties if d > 0.8)}")
            print(f"  Medium problems (0.3-0.8): {sum(1 for d in difficulties if 0.3 <= d <= 0.8)}")
            print(f"  Easy problems (<0.3): {sum(1 for d in difficulties if d < 0.3)}")

    # Save outputs if requested
    if args.output_dir or args.json or args.csv:
        output_dir = args.output_dir or Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON output
        if args.json:
            json_path = output_dir / f"selector_rankings_{timestamp}.json"

            # Prepare data for JSON
            if args.compute_categories:
                # Structured format with categories
                rankings_dict = {}
                task_details_dict = {}
                pairwise_diffs_dict = {}

                for cat_key in all_rankings.keys():
                    cat_scores = all_rankings[cat_key]

                    # Merge CI if available
                    if cat_key in all_ci_results and all_ci_results[cat_key] is not None:
                        cat_scores = cat_scores.merge(
                            all_ci_results[cat_key][["selector", "Score_ci_lower", "Score_ci_upper"]],
                            on="selector",
                            how="left"
                        )

                    rankings_dict[cat_key] = cat_scores.to_dict(orient="records")

                    if cat_key in all_task_details:
                        task_details_dict[cat_key] = all_task_details[cat_key].to_dict(orient="records") if args.verbose else None

                    if cat_key in all_diff_results and all_diff_results[cat_key] is not None:
                        pairwise_diffs_dict[cat_key] = all_diff_results[cat_key].to_dict(orient="records")
            else:
                # Backward compatible format
                rankings_dict = df_scores.to_dict(orient="records")
                task_details_dict = df_task_details.to_dict(orient="records") if args.verbose else None
                pairwise_diffs_dict = P_diff.to_dict(orient="records") if P_diff is not None else None

            # Add metadata
            output_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "tasks_analyzed": len(analyzed_tasks),
                    "tasks_excluded": list(excluded_tasks),
                    "selectors": len(df_scores),
                    "categories_computed": args.compute_categories,
                    "hyperparameters": {
                        "alpha": args.alpha,
                        "beta": args.beta,
                        "gamma": args.gamma,
                        "min_spread": args.min_spread,
                        "task_weight_mode": args.task_weight_mode,
                        "min_coverage": args.min_coverage,
                        "min_tasks": args.min_tasks,
                        "coverage_penalty": args.coverage_penalty,
                        "bootstrap_samples": args.bootstrap_samples if not args.no_bootstrap else 0,
                        "seed": args.seed,
                    }
                },
                "rankings": rankings_dict,
                "task_details": task_details_dict,
                "pairwise_differences": pairwise_diffs_dict,
            }

            # Add category statistics if computed
            if args.compute_categories and category_stats:
                output_data["category_summary"] = category_stats

            with open(json_path, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            logger.info(f"Saved JSON report to {json_path}")

        # CSV output
        if args.csv:
            # Save global rankings
            csv_path = output_dir / f"selector_rankings_{timestamp}.csv"
            df_scores.to_csv(csv_path, index=False)
            logger.info(f"Saved CSV report to {csv_path}")

            if args.verbose and df_task_details is not None:
                details_csv_path = output_dir / f"task_details_{timestamp}.csv"
                df_task_details.to_csv(details_csv_path, index=False)
                logger.info(f"Saved task details to {details_csv_path}")

            # Save category-specific CSV files if computed
            if args.compute_categories:
                for cat_key in all_rankings.keys():
                    if cat_key == 'global':
                        continue  # Already saved above

                    cat_scores = all_rankings[cat_key].copy()  # Make a copy to avoid modifying original

                    # If this is a window-specific ranking, mark unimplementable selectors
                    if cat_key.startswith('window_'):
                        position_incompatible_selectors = ['con64@', 'avg64@']
                        cat_scores['selector'] = cat_scores['selector'].apply(
                            lambda x: f"{x} (*)" if x in position_incompatible_selectors else x
                        )
                        # Add a note column
                        cat_scores['note'] = cat_scores['selector'].apply(
                            lambda x: 'Not implementable in position mode' if any(s in x for s in ['con64@ (*)', 'avg64@ (*)']) else ''
                        )

                    # Merge CI if available
                    if cat_key in all_ci_results and all_ci_results[cat_key] is not None:
                        cat_scores = cat_scores.merge(
                            all_ci_results[cat_key][["selector", "Score_ci_lower", "Score_ci_upper"]],
                            on="selector",
                            how="left"
                        )

                    cat_csv_path = output_dir / f"selector_rankings_{cat_key}_{timestamp}.csv"
                    cat_scores.to_csv(cat_csv_path, index=False)
                    logger.info(f"Saved {cat_key} CSV report to {cat_csv_path}")

                    if args.verbose and cat_key in all_task_details:
                        cat_details_csv = output_dir / f"task_details_{cat_key}_{timestamp}.csv"
                        all_task_details[cat_key].to_csv(cat_details_csv, index=False)
                        logger.info(f"Saved {cat_key} task details to {cat_details_csv}")

    logger.info("Ranking analysis complete")

if __name__ == "__main__":
    main()