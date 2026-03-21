#!/usr/bin/env python3
"""
自动从 meta.json 发现评估报告并生成 ground truth

目录映射规则（支持 MUI_Public 整体移动）：
    1. 从起始目录向上查找 MUI_Public 根目录（包含 neuron/ 和 infer_results/ 的目录）
    2. 评估报告位于: {MUI_Public}/infer_results/ + report_path（去掉 ./）

示例：
    神经元输出: {MUI_Public}/neuron/DeepSeek-R1-0528-Qwen3-8B/aime24/neuron_output_xxx/
    meta.json 中: report_path = "./evaluation_results_xxx/eval_xxx.json"
    评估报告:   {MUI_Public}/infer_results/evaluation_results_xxx/eval_xxx.json

用法:
    # 从 cache 的 meta.json 自动生成
    python tools/auto_generate_ground_truth.py --cache-root ./cache_v4_full

    # 从原始神经元输出目录生成
    python tools/auto_generate_ground_truth.py \
        --neuron-dir /path/to/MUI_Public/neuron/DeepSeek-R1-0528-Qwen3-8B/aime24/neuron_output_xxx
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_meta_json(start_dir: Path) -> Optional[Path]:
    """
    查找 meta.json 文件

    优先级：
    1. start_dir/meta.json (NAD cache)
    2. start_dir/*_meta.json (原始神经元输出)
    """
    # 优先检查 NAD cache 格式
    cache_meta = start_dir / "meta.json"
    if cache_meta.exists():
        return cache_meta

    # 检查原始神经元输出格式（*_meta.json）
    meta_files = list(start_dir.glob("*_meta.json"))
    if meta_files:
        # 如果有多个，优先选择包含 "ultra" 的
        ultra_metas = [m for m in meta_files if "ultra" in m.name]
        if ultra_metas:
            return ultra_metas[0]
        return meta_files[0]

    return None


def find_mui_public_root(start_dir: Path) -> Optional[Path]:
    """
    向上查找 MUI_Public 根目录

    识别标志：同时包含 neuron/ 和 infer_results/ 子目录

    Args:
        start_dir: 起始搜索目录

    Returns:
        MUI_Public 根目录，或 None
    """
    current = start_dir.resolve()

    # 向上搜索最多 10 层
    for _ in range(10):
        # 检查是否是 MUI_Public 根目录（包含 neuron/ 和 infer_results/）
        neuron_dir = current / "neuron"
        infer_results_dir = current / "infer_results"

        if neuron_dir.exists() and infer_results_dir.exists():
            return current

        # 向上一层
        parent = current.parent
        if parent == current:  # 到达文件系统根目录
            break
        current = parent

    return None


def find_eval_report(report_path: str, start_dir: Path) -> Optional[Path]:
    """
    根据文件名在 infer_results/ 及其子目录中递归查找评估报告

    规则：
    - 从当前目录向上查找 MUI_Public 根目录
    - 提取 report_path 的文件名
    - 在 {MUI_Public}/infer_results/ 及其所有子目录中递归查找该文件

    Args:
        report_path: meta.json 中的路径（可以是相对路径或绝对路径）
        start_dir: 神经元输出目录或 cache 目录

    Returns:
        评估报告绝对路径，或 None
    """
    # 查找 MUI_Public 根目录
    mui_public_root = find_mui_public_root(start_dir)
    if not mui_public_root:
        return None

    # 提取文件名
    filename = Path(report_path).name

    # 在 infer_results 目录及其子目录中递归查找
    infer_results_dir = mui_public_root / "infer_results"

    if not infer_results_dir.exists():
        return None

    # 递归查找匹配的文件
    matches = list(infer_results_dir.rglob(filename))

    if matches:
        # 如果有多个匹配，返回第一个（可以根据需要调整优先级）
        if len(matches) > 1:
            logger.warning(f"找到 {len(matches)} 个匹配文件，使用第一个: {matches[0]}")
            for m in matches:
                logger.warning(f"  - {m}")
        return matches[0]

    return None


def build_sample_index(meta_json: Dict[str, Any]) -> Dict[tuple, int]:
    """
    从 meta.json 构建 (problem_id, run_index) -> sample_id 映射
    """
    sample_index = {}
    samples = meta_json.get('samples', [])

    for sample_id, sample_info in enumerate(samples):
        problem_id = sample_info['problem_id']
        run_index = sample_info['run_index']
        sample_index[(problem_id, run_index)] = sample_id

    return sample_index


def extract_ground_truth(eval_report: Dict[str, Any],
                         sample_index: Dict[tuple, int]) -> Dict[str, Any]:
    """从评估报告提取 ground truth 信息"""
    sample_breakdown = {}
    metadata = {
        'dataset': eval_report['test_info']['dataset'],
        'timestamp': eval_report['test_info']['timestamp'],
        'total_problems': eval_report['test_info']['total_problems'],
        'n_runs_per_problem': eval_report['test_info']['n_runs']
    }

    # 遍历每个问题的结果
    for result in eval_report['results']:
        problem_id = result['problem_id']
        ground_truth_answer = result['ground_truth']
        prompt = result.get('prompt', '')  # 提取问题prompt
        runs = result['runs']

        correct_samples = []
        incorrect_samples = []

        # 遍历每个 run
        for run in runs:
            run_index = run['run_index']
            is_correct = run['is_correct']

            # 查找对应的 sample_id
            key = (problem_id, run_index)
            if key in sample_index:
                sample_id = sample_index[key]
            else:
                logger.warning(f"未找到 ({problem_id}, {run_index}) 的映射，跳过")
                continue

            if is_correct:
                correct_samples.append(sample_id)
            else:
                incorrect_samples.append(sample_id)

        # 记录该问题的分类（包含prompt）
        sample_breakdown[str(problem_id)] = {
            'prompt': prompt,  # 添加问题prompt
            'ground_truth': ground_truth_answer,
            'correct_samples': sorted(correct_samples),
            'incorrect_samples': sorted(incorrect_samples),
            'total_runs': len(runs),
            'correct_count': len(correct_samples),
            'incorrect_count': len(incorrect_samples),
            'accuracy': len(correct_samples) / len(runs) if runs else 0
        }

    return {
        'metadata': metadata,
        'sample_breakdown': sample_breakdown
    }


def main():
    parser = argparse.ArgumentParser(
        description='自动从 meta.json 发现评估报告并生成 ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 NAD cache 自动生成
  python tools/auto_generate_ground_truth.py --cache-root ./cache_v4_full

  # 从原始神经元输出目录生成
  python tools/auto_generate_ground_truth.py \\
      --neuron-dir ./MUI_public/neuron/DeepSeek-R1-0528-Qwen3-8B/aime24/neuron_output_1_act_no_rms_20250902_025610

  # 指定输出位置
  python tools/auto_generate_ground_truth.py \\
      --cache-root ./cache_v4_full \\
      --out ./custom_ground_truth.json
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--cache-root',
        type=str,
        help='NAD cache 根目录（包含 meta.json）'
    )

    group.add_argument(
        '--neuron-dir',
        type=str,
        help='原始神经元输出目录（包含 *_meta.json）'
    )

    parser.add_argument(
        '--out',
        type=str,
        help='输出文件路径（默认：<cache-root>/evaluation_report_compact.json）'
    )

    parser.add_argument(
        '--show-stats',
        action='store_true',
        help='显示详细统计信息'
    )

    args = parser.parse_args()

    # 确定起始目录
    if args.cache_root:
        start_dir = Path(args.cache_root).resolve()
    else:
        start_dir = Path(args.neuron_dir).resolve()

    if not start_dir.exists():
        logger.error(f"目录不存在: {start_dir}")
        return 1

    logger.info(f"起始目录: {start_dir}")

    try:
        # 步骤 1: 查找 meta.json
        logger.info("\n步骤 1: 查找 meta.json...")
        meta_path = find_meta_json(start_dir)

        if not meta_path:
            logger.error(f"在 {start_dir} 中未找到 meta.json 或 *_meta.json")
            return 1

        logger.info(f"  ✓ 找到: {meta_path.name}")

        # 加载 meta.json
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_json = json.load(f)

        logger.info(f"  ✓ 总样本数: {len(meta_json.get('samples', []))}")

        # 步骤 2: 从 meta.json 提取评估报告路径
        logger.info("\n步骤 2: 提取评估报告路径...")
        report_path = meta_json.get('report_path')

        if not report_path:
            logger.error("meta.json 中缺少 'report_path' 字段")
            return 1

        logger.info(f"  ✓ 相对路径: {report_path}")

        # 步骤 3: 查找 MUI_Public 根目录
        logger.info("\n步骤 3: 查找 MUI_Public 根目录...")
        mui_public_root = find_mui_public_root(start_dir)
        if not mui_public_root:
            logger.error(f"无法找到 MUI_Public 根目录")
            logger.error(f"  搜索起点: {start_dir}")
            logger.error(f"  识别标志: 同时包含 neuron/ 和 infer_results/ 子目录")
            return 1
        logger.info(f"  ✓ 找到: {mui_public_root}")

        # 步骤 4: 查找评估报告
        logger.info("\n步骤 4: 查找评估报告...")
        filename = Path(report_path).name
        logger.info(f"  文件名: {filename}")
        logger.info(f"  搜索范围: {{MUI_Public}}/infer_results/ (递归)")
        eval_report_path = find_eval_report(report_path, start_dir)

        if not eval_report_path:
            logger.error(f"未找到评估报告")
            logger.error(f"  搜索文件名: {filename}")
            logger.error(f"  搜索目录: {mui_public_root / 'infer_results'}")
            logger.error(f"\n请检查：")
            logger.error(f"  1. 评估报告文件是否存在于 infer_results/ 或其子目录中")
            logger.error(f"  2. 文件名是否匹配: {filename}")
            return 1

        logger.info(f"  ✓ 找到: {eval_report_path}")

        # 加载评估报告
        with open(eval_report_path, 'r', encoding='utf-8') as f:
            eval_report = json.load(f)

        logger.info(f"  ✓ 数据集: {eval_report['test_info']['dataset']}")
        logger.info(f"  ✓ 问题数: {eval_report['test_info']['total_problems']}")
        logger.info(f"  ✓ 每题 runs: {eval_report['test_info']['n_runs']}")

        # 步骤 5: 构建样本索引
        logger.info("\n步骤 5: 构建样本索引...")
        sample_index = build_sample_index(meta_json)
        logger.info(f"  ✓ 索引条目: {len(sample_index)}")

        # 步骤 6: 提取 ground truth
        logger.info("\n步骤 6: 提取 ground truth...")
        ground_truth = extract_ground_truth(eval_report, sample_index)

        # 显示统计信息
        if args.show_stats:
            logger.info("\n" + "="*70)
            logger.info("Ground Truth 统计")
            logger.info("="*70)

            metadata = ground_truth['metadata']
            logger.info(f"\n数据集: {metadata['dataset']}")
            logger.info(f"时间戳: {metadata['timestamp']}")
            logger.info(f"问题数: {metadata['total_problems']}")
            logger.info(f"每题 runs: {metadata['n_runs_per_problem']}")

            logger.info(f"\n问题级别准确率:")
            logger.info("-"*70)
            logger.info(f"{'问题ID':<10s} {'总数':>8s} {'正确':>8s} {'错误':>8s} {'准确率':>10s}")
            logger.info("-"*70)

            sample_breakdown = ground_truth['sample_breakdown']
            total_correct = 0
            total_samples = 0

            for pid in sorted(sample_breakdown.keys()):
                prob = sample_breakdown[pid]
                correct = prob['correct_count']
                total = prob['total_runs']
                incorrect = prob['incorrect_count']
                acc = prob['accuracy']

                logger.info(f"{pid:<10s} {total:>8d} {correct:>8d} {incorrect:>8d} {acc*100:>9.2f}%")

                total_correct += correct
                total_samples += total

            logger.info("-"*70)
            overall_acc = total_correct / total_samples if total_samples > 0 else 0
            logger.info(f"{'总计':<10s} {total_samples:>8d} {total_correct:>8d} {total_samples - total_correct:>8d} {overall_acc*100:>9.2f}%")
            logger.info("="*70 + "\n")

        # 步骤 7a: 复制完整的 evaluation report
        logger.info("\n步骤 7a: 复制完整的 evaluation report...")
        eval_report_copy_path = start_dir / "evaluation_report.json"

        import shutil
        shutil.copy2(eval_report_path, eval_report_copy_path)

        file_size_mb = eval_report_copy_path.stat().st_size / 1024 / 1024
        logger.info(f"  ✓ 完整版已复制: {eval_report_copy_path}")
        logger.info(f"  ✓ 文件大小: {file_size_mb:.1f} MB")

        # 步骤 7b: 生成精简版 ground_truth（删除大字段）
        logger.info("\n步骤 7b: 生成精简版 ground_truth...")

        # 深拷贝eval_report并删除大字段
        import copy
        compact_report = copy.deepcopy(eval_report)

        removed_fields = ['actual_prompt', 'generated_text']
        total_runs = 0
        for result in compact_report['results']:
            for run in result['runs']:
                total_runs += 1
                for field in removed_fields:
                    run.pop(field, None)

        logger.info(f"  ✓ 已从 {total_runs} 个runs中删除字段: {', '.join(removed_fields)}")

        # 确定输出路径
        if args.out:
            out_path = Path(args.out)
        elif args.cache_root:
            out_path = start_dir / "evaluation_report_compact.json"
        else:
            out_path = start_dir / "evaluation_report_compact.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(compact_report, f, indent=2, ensure_ascii=False)

        compact_size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info(f"  ✓ 精简版已保存: {out_path}")
        logger.info(f"  ✓ 文件大小: {compact_size_mb:.1f} MB (压缩率: {(1-compact_size_mb/file_size_mb)*100:.1f}%)")

        # 显示简要统计
        total_problems = len(compact_report['results'])
        total_samples = sum(len(r['runs']) for r in compact_report['results'])
        total_correct = sum(sum(1 for run in r['runs'] if run['is_correct']) for r in compact_report['results'])

        logger.info(f"\n{'='*70}")
        logger.info("✅ 完成！")
        logger.info(f"{'='*70}")
        logger.info(f"问题数: {total_problems}")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"正确样本: {total_correct} ({total_correct/total_samples*100:.2f}%)")
        logger.info(f"错误样本: {total_samples - total_correct} ({(total_samples - total_correct)/total_samples*100:.2f}%)")
        logger.info(f"{'='*70}\n")
        logger.info(f"生成的文件:")
        logger.info(f"  - 完整版: {eval_report_copy_path} ({file_size_mb:.1f} MB)")
        logger.info(f"  - 精简版: {out_path} ({compact_size_mb:.1f} MB)")
        logger.info(f"{'='*70}\n")

        return 0

    except Exception as e:
        logger.error(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
