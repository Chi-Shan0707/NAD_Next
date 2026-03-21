#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 K_token_per_kink.py 迁移到 NAD_Next 的缓存格式（v4.0/4.1）。
- 使用 nad.core.views.reader.CacheReader 读取 cache
- 使用 rows/ Row-CSR bank 与 token_data/ 元信息来重建按 token（或行）推进的"累计唯一神经元数"曲线
- 不再依赖 NAD_tmp 的 efficient_cache_utils / fast_neuron_reader
- Ground truth（是否正确）从 cache 根目录下的 evaluation_report_compact.json / evaluation_report.json + meta.json 自动构建

输出：
- 每个数据集一张交互式 HTML 曲线图（token_per_kink vs 准确率）
- 每个数据集一份 rank_accuracies.csv（每个 rank 的准确率与平均 token/kink）
- 每个数据集一份 *_token_per_kink_normalized.csv（将 token/kink 线性归一到[0,100]）
"""

import os
import csv
import json
import argparse
import time
import gc
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Literal
from concurrent.futures import ThreadPoolExecutor

import numpy as np
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.offline as pyo  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
# Optional tqdm (fallback to identity iterator if not installed)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, *args, **kwargs):
        return it

# ==== NEW: NAD_Next ====
from nad.core.views.reader import CacheReader
from nad.ops.uniques import extract_tokenwise_counts
from nad.ops.accuracy import load_correctness_map as nad_load_correctness_map
# =======================

# ==== Kink detection functions (imported from plugin to avoid code duplication) ====
from plugins.kink_selector import moving_average, detect_kinks_mad_legacy
# ===================================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _load_meta_json(cache_root: str) -> Dict:
    """Load meta.json for problem_id mapping."""
    meta_path = os.path.join(cache_root, 'meta.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到 meta.json: {meta_path}")
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return meta


def _compute_one_run_fast(run_id: int,
                          rows_srp: np.ndarray,
                          rows_rp: np.ndarray,
                          rows_keys: np.ndarray,
                          rows_trp: Optional[np.ndarray],
                          rows_slice_ids: Optional[np.ndarray],
                          meta: Dict,
                          correctness_map: Dict[int, bool],
                          token_axis: Literal['tokens','row'] = 'row') -> Optional[Dict]:
    """
    线程安全：单样本快速计算 token_per_kink（内联 NumPy 向量化算法）。

    参数：
        run_id: 样本ID
        rows_srp, rows_rp, rows_keys, rows_trp: 预加载的 rows/ bank 数组（memmap）
        meta: 元数据字典
        correctness_map: 正确性映射

    返回：
        包含 run_id, problem_id, token_per_kink 等信息的字典，或 None（如果样本为空）
    """
    tokens, counts = extract_tokenwise_counts(
        run_id,
        rows_srp,
        rows_rp,
        rows_keys,
        rows_slice_ids,
        rows_trp,
        token_axis=token_axis,
    )

    if tokens.size == 0:
        return None

    kinks = detect_kinks_mad_legacy(tokens, counts, smooth_window=5, z_threshold=2.5, min_jump=10)
    num_kinks = len(kinks)

    if token_axis == 'tokens':
        num_tokens = int(tokens[-1])
    else:
        num_tokens = int(tokens[-1]) + 1

    token_per_kink = float(num_tokens) / num_kinks if num_kinks > 0 else float('inf')
    is_correct = bool(correctness_map.get(run_id, False))
    pid = str(meta['samples'][run_id]['problem_id'])

    return {
        'run_id': run_id,
        'problem_id': pid,
        'token_per_kink': token_per_kink,
        'num_tokens': num_tokens,
        'num_kinks': num_kinks,
        'is_correct': is_correct,
    }


def process_dataset(cache_root: str, dataset_name: str, out_dir: str,
                    enable_profiling: bool = False, workers: int = 0,
                    token_axis: Literal['tokens','row'] = 'row') -> Optional[Dict]:
    """数据集处理：计算每题按 token_per_kink 排序后的 rank → 准确率"""
    print("="*60)
    print(f"📦 数据集: {dataset_name}")
    print(f"📂 Cache 目录: {cache_root}")
    print("="*60)

    # Profiling setup
    start_time = time.time()
    start_memory = None
    memory_samples = []
    if enable_profiling and PSUTIL_AVAILABLE:
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_samples.append(start_memory)

    # 1) 载入 correctness_map + meta
    t0 = time.time()
    try:
        correctness_map = nad_load_correctness_map(cache_root)
        meta = _load_meta_json(cache_root)
    except Exception as e:
        print(f"❌ 加载 ground truth 失败: {e}")
        return None
    if enable_profiling:
        print(f"✅ ground truth 条目数: {len(correctness_map)} (用时: {time.time()-t0:.2f}s)")
    else:
        print(f"✅ ground truth 条目数: {len(correctness_map)}")

    # 2) 打开 NAD_Next cache
    t0 = time.time()
    try:
        reader = CacheReader(cache_root)
    except Exception as e:
        print(f"❌ 打开 NAD_Next cache 失败: {e}")
        return None

    num_runs = reader.num_runs()
    if enable_profiling:
        print(f"✅ 发现样本数（runs）: {num_runs} (用时: {time.time()-t0:.2f}s)")
    else:
        print(f"✅ 发现样本数（runs）: {num_runs}")

    # 3) 预取 rows/ bank 指针，便于多线程共享（memmap 只读）
    rows_srp = reader.rows_sample_row_ptr
    rows_rp = reader.rows_row_ptr
    rows_keys = reader.rows_keys
    rows_trp = reader.rows_token_row_ptr
    rows_slice_ids = reader.rows_slice_ids

    if rows_srp is None or rows_rp is None or rows_keys is None:
        print("⚠️ 缺少 rows/ bank（rows_*），无法计算 token_per_kink；请用含 rows/ 的缓存重建。")
        return None

    # 4) 并行/串行计算
    per_problem_runs: Dict[str, List[Dict]] = defaultdict(list)
    print("📊 逐样本计算 token_per_kink …")
    t0 = time.time()

    # 确定 worker 数量（默认 0 = auto）
    workers = int(workers) if workers and workers > 0 else max(1, min(8, (os.cpu_count() or 4)))

    if workers == 1:
        # 串行处理（带进度条）
        iterator = tqdm(range(num_runs), desc="Processing samples", unit="sample") if enable_profiling else range(num_runs)
        for run_id in iterator:
            res = _compute_one_run_fast(run_id, rows_srp, rows_rp, rows_keys, rows_trp,
                                        rows_slice_ids, meta, correctness_map,
                                        token_axis=token_axis)
            if res is not None:
                per_problem_runs[res['problem_id']].append(res)

            # Memory sampling (every 100 samples)
            if enable_profiling and PSUTIL_AVAILABLE and run_id % 100 == 0:
                current_mem = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_mem)
    else:
        # 并行处理（线程池，使用 executor.map 降低 future 持有开销）
        with ThreadPoolExecutor(max_workers=workers) as executor:
            mapper = executor.map(
                lambda i: _compute_one_run_fast(i, rows_srp, rows_rp, rows_keys, rows_trp,
                                                rows_slice_ids, meta, correctness_map,
                                                token_axis),
                range(num_runs),
                chunksize=1
            )
            iterator = tqdm(mapper, total=num_runs, desc="Processing samples", unit="sample") if enable_profiling else mapper
            for idx, res in enumerate(iterator):
                if res is not None:
                    per_problem_runs[res['problem_id']].append(res)

                # Memory sampling (every 100 samples)
                if enable_profiling and PSUTIL_AVAILABLE and idx % 100 == 0:
                    current_mem = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_mem)

        # 尽早释放临时大对象
        gc.collect()

    # 5) 与 NAD_backup 对齐的题内过滤与排序：
    #    - 剔除 token_per_kink = inf 的样本；若整题全是 inf，则该题剔除
    #    - 按 token_per_kink 纯降序排序（不使用 (isfinite, value) 复合 key）
    filtered_problems: Dict[str, List[Dict]] = {}
    for pid, runs in per_problem_runs.items():
        valid_runs = [r for r in runs if np.isfinite(r['token_per_kink'])]
        if not valid_runs:
            continue
        # 纯降序，保持稳定性（sorted 是稳定的；提供 key 即可）
        valid_runs.sort(key=lambda r: r['token_per_kink'], reverse=True)
        filtered_problems[pid] = valid_runs

    max_runs = max((len(runs) for runs in filtered_problems.values()), default=0)
    print(f"📊 每道题最多有 {max_runs} 个 response")

    # 6) 统计每个 rank 的准确率与平均 token/kink
    rank_accuracies: List[Dict] = []
    for rank in range(max_runs):
        selected = []
        for pid, runs in filtered_problems.items():
            if rank < len(runs):
                selected.append(runs[rank])
        if not selected:
            continue
        acc = float(np.mean([1.0 if s['is_correct'] else 0.0 for s in selected]))
        # 此处 selected 均为 finite；直接求均值
        avg_tpk = float(np.mean([s['token_per_kink'] for s in selected]))
        rank_accuracies.append({
            'rank': rank + 1,
            'accuracy': acc,
            'avg_token_per_kink': avg_tpk,
            'total_samples': len(selected),
        })

    # 7) 导出 rank_accuracies.csv
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset_name}_rank_accuracies.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'accuracy', 'avg_token_per_kink', 'total_samples'])
        for r in rank_accuracies:
            w.writerow([r['rank'], r['accuracy'], r['avg_token_per_kink'], r['total_samples']])
    print(f"📝 已保存: {csv_path}")

    # 8) 生成交互式 HTML 曲线
    xs = [r['avg_token_per_kink'] for r in rank_accuracies]
    ys = [r['accuracy'] for r in rank_accuracies]
    ranks = [r['rank'] for r in rank_accuracies]
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines+markers',
            marker=dict(size=8, color=ranks, colorscale='Viridis', showscale=True,
                        colorbar=dict(title='Rank')),
            line=dict(width=2),
            text=[f"Rank {r['rank']}<br>Token/Kink: {r['avg_token_per_kink']:.2f}<br>Acc: {r['accuracy']:.4f}<br>Samples: {r['total_samples']}"
                  for r in rank_accuracies],
            hovertemplate='%{text}<extra></extra>',
            name=dataset_name
        ))
        fig.update_layout(
            title=f'{dataset_name}: Token/Kink vs Accuracy',
            xaxis_title='Average Token per Kink (Higher = Fewer Kinks)',
            yaxis_title='Accuracy',
            hovermode='closest',
            width=1000, height=600
        )
        html_path = os.path.join(out_dir, f"{dataset_name}_token_per_kink.html")
        pyo.plot(fig, filename=html_path, auto_open=False)
        print(f"📊 已保存交互式图表: {html_path}")
    else:
        print("ℹ️ plotly 未安装，跳过 HTML 图表生成")

    # 9) 保存归一化 CSV（将 xs 线性映射到 [0,100]）
    if xs:
        finite_x = [v for v in xs if np.isfinite(v)]
        if finite_x:
            mn, mx = float(np.min(finite_x)), float(np.max(finite_x))
            if mx > mn:
                normalized = [ (v - mn) / (mx - mn) * 100.0 if np.isfinite(v) else 100.0 for v in xs ]
            else:
                normalized = [50.0 for _ in xs]
        else:
            normalized = [50.0 for _ in xs]
    else:
        normalized = []
    norm_csv = os.path.join(out_dir, f"{dataset_name}_token_per_kink_normalized.csv")
    with open(norm_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank', 'normalized_token_per_kink', 'accuracy'])
        for i, r in enumerate(rank_accuracies):
            w.writerow([r['rank'], normalized[i], r['accuracy']])
    print(f"📝 已保存归一化 CSV: {norm_csv}")

    # Performance statistics
    if enable_profiling:
        total_time = time.time() - start_time
        processing_time = time.time() - t0
        print(f"\n{'='*60}")
        print(f"⚡ 性能统计:")
        print(f"{'='*60}")
        print(f"  总用时: {total_time:.2f}s")
        print(f"  样本处理用时: {processing_time:.2f}s ({num_runs/processing_time:.1f} samples/s)")
        if PSUTIL_AVAILABLE and start_memory is not None:
            # Final memory sample
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

            # Calculate memory statistics
            peak_memory = max(memory_samples) if memory_samples else current_memory
            avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else current_memory

            print(f"  起始内存: {start_memory:.1f} MB")
            print(f"  当前内存: {current_memory:.1f} MB")
            print(f"  峰值内存: {peak_memory:.1f} MB")
            print(f"  平均内存: {avg_memory:.1f} MB")
            print(f"  内存增长: {current_memory - start_memory:.1f} MB")
            print(f"  内存采样次数: {len(memory_samples)}")
        print(f"{'='*60}\n")

    return {
        'dataset_name': dataset_name,
        'rank_accuracies': rank_accuracies,
        'per_problem_runs': per_problem_runs,
        'filtered_problems': filtered_problems,
        'out_dir': out_dir,
    }


def plot_combined_normalized(all_results: List[Dict], out_dir: str) -> None:
    """生成所有数据集的归一化对比图"""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for result in all_results:
        if not result:
            continue
        name = result['dataset_name']
        ra = result['rank_accuracies']
        xs = [r['avg_token_per_kink'] for r in ra]
        ys = [r['accuracy'] for r in ra]
        if not xs:
            continue
        finite_x = [v for v in xs if np.isfinite(v)]
        if finite_x:
            mn, mx = float(np.min(finite_x)), float(np.max(finite_x))
            if mx > mn:
                nx = [ (v - mn) / (mx - mn) * 100.0 if np.isfinite(v) else 100.0 for v in xs ]
            else:
                nx = [50.0 for _ in xs]
        else:
            nx = [50.0 for _ in xs]
        plt.plot(nx, ys, marker='o', label=name)
    plt.xlabel('Normalized Token/Kink [0-100] (Higher = Fewer Kinks)')
    plt.ylabel('Accuracy')
    plt.title('Token/Kink vs Accuracy (Normalized, All Datasets)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    out_path = os.path.join(out_dir, 'token_per_kink_all_datasets_normalized.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存归一化对比图: {out_path}")
    plt.close()


def parse_datasets_arg(values: List[str]) -> List[Tuple[str, str]]:
    """将 --dataset NAME=PATH 的多值解析成 [(NAME, PATH), ...]"""
    result = []
    for v in values:
        if '=' not in v:
            raise argparse.ArgumentTypeError("--dataset 需要 NAME=PATH 形式")
        name, path = v.split('=', 1)
        result.append((name.strip(), path.strip()))
    return result


def main():
    parser = argparse.ArgumentParser(description="分析 token_per_kink 指标与准确率的关系（NAD_Next 版）")
    parser.add_argument(
        '--dataset', action='append', default=[],
        help='数据集配置：NAME=PATH（PATH 为 NAD_Next cache 根目录）。可重复传入多次。'
    )
    parser.add_argument('--out', default='./token_per_kink_out', help='输出目录')
    parser.add_argument('--profile', action='store_true', help='启用性能分析模式（显示进度条和详细性能统计）')
    parser.add_argument('--workers', type=int, default=0,
                        help='并行线程数（默认=0，即 min(8, CPU核数)；设为1则串行，建议按内存调优）')
    parser.add_argument('--token-axis', choices=['tokens','row'], default='row',
                        help='token 轴定义：tokens=真实 token 累计；row=按 slice_id（旧版语义）')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Profiling mode notification
    if args.profile:
        print("🔍 性能分析模式已启用")
        if not PSUTIL_AVAILABLE:
            print("⚠️  psutil 未安装，内存统计不可用。可通过 'pip install psutil' 安装。")
        print()

    # 若未显式传入，则尝试从环境变量 DATASETS（逗号分隔的 NAME=PATH 列表）读取
    datasets: List[Tuple[str, str]] = []
    if args.dataset:
        datasets = parse_datasets_arg(args.dataset)
    else:
        env = os.environ.get('DATASETS', '')
        if env:
            datasets = parse_datasets_arg([x.strip() for x in env.split(',') if x.strip()])

    if not datasets:
        print("⚠️ 未提供任何数据集（--dataset NAME=PATH）。示例：\n  --dataset AIME24=/path/to/cache_aime24 --dataset GSM8K=/path/to/cache_gsm8k")
        return

    all_results = []
    for name, cache_root in datasets:
        res = process_dataset(cache_root, name, args.out,
                              enable_profiling=args.profile, workers=args.workers,
                              token_axis=args.token_axis)
        all_results.append(res)

    # 跨数据集的归一化对比
    plot_combined_normalized(all_results, args.out)

    # 汇总打印（各数据集 Rank1 与最后一个 rank 的准确率差异）
    print("\n" + "="*60)
    print("📊 汇总结果:")
    print("="*60)
    for result in all_results:
        if not result:
            continue
        ranks = result['rank_accuracies']
        if len(ranks) >= 2:
            acc1 = ranks[0]['accuracy']
            acc_last = ranks[-1]['accuracy']
            diff = (acc1 - acc_last) * 100.0
            print(f"{result['dataset_name']:<15} Rank1(少 kink): {acc1*100:>6.2f}%  Rank{len(ranks)}(多 kink): {acc_last*100:>6.2f}%  Diff: {diff:>6.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
