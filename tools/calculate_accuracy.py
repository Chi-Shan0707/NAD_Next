#!/usr/bin/env python3
"""
NAD 选择器准确率评估工具

用法:
    python tools/calculate_accuracy.py \
        --selection ./results_all_selectors/full_sequence_result.json \
        --cache-root ./cache_aime24 \
        --out ./results_all_selectors/accuracy_report.json

功能:
    - 计算每个选择器的准确率 (正确数/总数)
    - 生成 problem-level 的详细结果
    - 打印格式化的准确率摘要表格
    - 保存详细报告到 JSON

输入格式:
    - selection: analyze 命令输出的 JSON，包含 problems[pid].selectors[selector_name] = run_id
    - cache-root: Cache 目录，自动查找 evaluation_report_compact.json 和 meta.json
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_sample_correctness_map(eval_report: Dict[str, Any],
                                   meta_json_path: str) -> Dict[int, bool]:
    """
    从evaluation report和meta.json构建 sample_id → is_correct 映射

    Args:
        eval_report: evaluation_report.json或evaluation_report_compact.json
        meta_json_path: meta.json文件路径

    Returns:
        {sample_id: is_correct}的映射字典
    """
    # 加载meta.json获取sample映射
    with open(meta_json_path, 'r') as f:
        meta = json.load(f)

    # 构建(problem_id, run_index) → sample_id映射
    sample_index = {}
    for sample_id, sample in enumerate(meta.get('samples', [])):
        problem_id = sample['problem_id']
        run_index = sample['run_index']
        sample_index[(problem_id, run_index)] = sample_id

    # 构建sample_id → is_correct映射
    correctness_map = {}
    for result in eval_report.get('results', []):
        problem_id = result['problem_id']
        for run in result.get('runs', []):
            run_index = run['run_index']
            is_correct = run['is_correct']

            key = (problem_id, run_index)
            if key in sample_index:
                sample_id = sample_index[key]
                correctness_map[sample_id] = is_correct

    return correctness_map


def calculate_accuracy(selection_results: Dict[str, Any],
                       ground_truth: Dict[str, Any],
                       meta_json_path: str = None) -> Dict[str, Any]:
    """
    计算选择器准确率

    Args:
        selection_results: analyze 命令输出的 JSON
        ground_truth: evaluation_report_compact.json 或旧的 sample_breakdown 格式
        meta_json_path: meta.json路径（当ground_truth是evaluation_report格式时需要）

    Returns:
        包含准确率统计的字典
    """
    # 检测ground_truth格式
    if 'results' in ground_truth and 'test_info' in ground_truth:
        # evaluation_report格式
        if not meta_json_path:
            raise ValueError("使用evaluation_report格式时需要提供--meta-json参数")

        # 构建sample_id → is_correct映射
        correctness_map = build_sample_correctness_map(ground_truth, meta_json_path)
    elif 'sample_breakdown' in ground_truth:
        # 旧的sample_breakdown格式 - 向后兼容
        sample_breakdown = ground_truth['sample_breakdown']
        # 构建correctness_map
        correctness_map = {}
        for pid_str, breakdown in sample_breakdown.items():
            for sample_id in breakdown.get('correct_samples', []):
                correctness_map[sample_id] = True
            for sample_id in breakdown.get('incorrect_samples', []):
                correctness_map[sample_id] = False
    else:
        raise ValueError("无法识别ground truth格式：需要'results'字段（evaluation_report）或'sample_breakdown'字段")

    problems = selection_results.get('problems', {})

    if not problems:
        raise ValueError("Selection results JSON 缺少 'problems' 字段")

    # 提取所有选择器的名称（从第一个问题中获取）
    first_problem_data = next(iter(problems.values()))
    selector_names = list(first_problem_data.get('selectors', {}).keys())

    if not selector_names:
        raise ValueError("Selection results 中没有找到任何选择器")

    # 为每个选择器计算准确率
    selector_stats = {}
    problem_details = {}

    for pid, problem_data in problems.items():
        problem_details[pid] = {}

        # 对每个选择器检查其选择是否正确
        for selector_name in selector_names:
            if selector_name not in problem_data['selectors']:
                logger.warning(f"问题 {pid} 缺少选择器 {selector_name} 的结果")
                continue

            selected_run_id = problem_data['selectors'][selector_name]

            # 从correctness_map查找是否正确
            if selected_run_id not in correctness_map:
                logger.warning(f"sample_id {selected_run_id} 不在 ground truth 中，跳过")
                continue

            is_correct = correctness_map[selected_run_id]

            # 初始化选择器统计（如果还不存在）
            if selector_name not in selector_stats:
                selector_stats[selector_name] = {
                    'correct': 0,
                    'incorrect': 0,
                    'total': 0,
                    'problems': []
                }

            # 更新统计
            selector_stats[selector_name]['total'] += 1
            if is_correct:
                selector_stats[selector_name]['correct'] += 1
            else:
                selector_stats[selector_name]['incorrect'] += 1

            selector_stats[selector_name]['problems'].append({
                'problem_id': pid,
                'selected_run_id': selected_run_id,
                'is_correct': is_correct
            })

            # 记录问题级别的详细结果
            problem_details[pid][selector_name] = {
                'selected_run_id': selected_run_id,
                'is_correct': is_correct
            }

    # 计算准确率百分比
    for selector_name, stats in selector_stats.items():
        total = stats['total']
        stats['accuracy'] = stats['correct'] / total if total > 0 else 0.0
        stats['accuracy_percent'] = stats['accuracy'] * 100

    return {
        'selector_accuracy': selector_stats,
        'problem_details': problem_details,
        'summary': {
            'total_problems': len(problem_details),
            'total_selectors': len(selector_stats),
            'selector_names': list(selector_stats.keys())
        }
    }


def print_accuracy_summary(results: Dict[str, Any]) -> None:
    """打印格式化的准确率摘要表格"""
    selector_stats = results['selector_accuracy']
    summary = results['summary']

    logger.info("\n" + "="*80)
    logger.info("NAD 选择器准确率报告")
    logger.info("="*80)
    logger.info(f"\n总问题数: {summary['total_problems']}")
    logger.info(f"总选择器数: {summary['total_selectors']}")
    logger.info("\n" + "-"*80)
    logger.info(f"{'选择器名称':<30s} {'准确率':>12s} {'正确数':>10s} {'总数':>10s}")
    logger.info("-"*80)

    # 按准确率降序排列
    sorted_selectors = sorted(
        selector_stats.items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )

    for selector_name, stats in sorted_selectors:
        acc_pct = stats['accuracy_percent']
        correct = stats['correct']
        total = stats['total']
        logger.info(f"{selector_name:<30s} {acc_pct:>11.2f}% {correct:>10d} {total:>10d}")

    logger.info("-"*80)

    # 计算平均准确率
    if selector_stats:
        avg_accuracy = sum(s['accuracy'] for s in selector_stats.values()) / len(selector_stats)
        logger.info(f"{'平均准确率':<30s} {avg_accuracy*100:>11.2f}%")

    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="计算 NAD 选择器准确率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python tools/calculate_accuracy.py \\
      --selection ./results_all_selectors/full_sequence_result.json \\
      --cache-root ./cache_aime24

  # 指定输出文件
  python tools/calculate_accuracy.py \\
      --selection ./results_all_selectors/full_sequence_result.json \\
      --cache-root ./cache_aime24 \\
      --out ./accuracy_report.json

  # 安静模式（不打印表格）
  python tools/calculate_accuracy.py \\
      --selection ./results.json \\
      --cache-root ./cache_aime24 \\
      --quiet
        """
    )

    parser.add_argument(
        '--selection',
        required=True,
        help='Selection results JSON (from analyze command)'
    )

    parser.add_argument(
        '--cache-root',
        required=True,
        help='Cache root directory (must contain meta.json and evaluation_report_compact.json)'
    )

    parser.add_argument(
        '--out',
        default='accuracy_report.json',
        help='Output JSON file path (default: accuracy_report.json)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output (only save to file)'
    )

    args = parser.parse_args()

    # 验证 selection results 存在
    if not Path(args.selection).exists():
        logger.error(f"Selection results 文件不存在: {args.selection}")
        return 1

    # 从 cache_root 自动查找需要的文件
    cache_root = Path(args.cache_root)
    if not cache_root.exists():
        logger.error(f"Cache 目录不存在: {cache_root}")
        return 1

    meta_json_path = cache_root / "meta.json"
    if not meta_json_path.exists():
        logger.error(f"在 cache 目录中未找到 meta.json: {meta_json_path}")
        return 1

    ground_truth_path = cache_root / "evaluation_report_compact.json"
    if not ground_truth_path.exists():
        logger.error(f"在 cache 目录中未找到 evaluation_report_compact.json: {ground_truth_path}")
        return 1

    try:
        # 加载数据
        if not args.quiet:
            logger.info(f"Cache 目录: {cache_root}")
            logger.info(f"加载 selection results: {args.selection}")
        selection_results = load_json(args.selection)

        if not args.quiet:
            logger.info(f"加载 evaluation report (精简版): {ground_truth_path}")
        ground_truth = load_json(str(ground_truth_path))

        # 计算准确率
        if not args.quiet:
            logger.info("计算准确率...")
        results = calculate_accuracy(selection_results, ground_truth, str(meta_json_path))

        # 打印摘要
        if not args.quiet:
            print_accuracy_summary(results)

        # 保存结果
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if not args.quiet:
            logger.info(f"✅ 准确率报告已保存到: {out_path}")

        return 0

    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
