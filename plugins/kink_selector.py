"""
kink_selector.py - 用户自定义选择器示例（基于 Token-per-Kink）

该文件展示了如何使用 NAD 核心库的通用组件编写外部插件选择器。
它不依赖 `kink.py`，而是直接使用：
  - nad.core.selectors.base.Selector（基类）
  - nad.ops.uniques.extract_tokenwise_counts（通用操作）

# 使用方法
step2_analyze.sh cache_aime24 full 64 roaring \
  --user-selector 'file:./plugins/kink_selector.py:KinkSelector'

或在Python中调用：
python3 -m nad.cli analyze \
  --cache-root cache_aime24 \
  --selectors 'all,file:./plugins/kink_selector.py:KinkSelector'

# 版本历史
v2.0 (2025-11-05): 使用 step3 的完整 kink 检测算法（与 k_token_per_kink.py 对齐）
v1.0 (初始版本): 简化的 MAD 阈值检测
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from nad.core.selectors.base import Selector, SelectorContext
from nad.ops.uniques import extract_tokenwise_counts


# ============================================================================
# Kink 检测算法（从 k_token_per_kink.py 复制，确保完全一致）
# ============================================================================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """简单的反射填充 + 移动平均"""
    if w <= 1 or len(x) == 0:
        return x.astype(np.float64)
    pad = w // 2
    xpad = np.pad(x.astype(np.float64), (pad, pad), mode='reflect')
    kern = np.ones(w, dtype=np.float64) / w
    y = np.convolve(xpad, kern, mode='valid')
    return y[:len(x)]


def detect_kinks_mad_legacy(
    tokens: np.ndarray,
    neuron_counts: np.ndarray,
    smooth_window: int = 5,
    z_threshold: float = 2.5,
    min_jump: int = 10
) -> List[Tuple[int, float, float]]:
    """
    旧版 kink 检测实现（与 k_token_per_kink.py 完全一致）：
      - 在斜率变化 np.diff(slopes) 上做 MAD 检测
      - 使用 0.6745 因子、单侧阈值 (z > z_threshold)
      - 使用 dy[i+1] - dy[i] >= min_jump 过滤

    Args:
        tokens: Token 索引数组
        neuron_counts: 每个 token 对应的累积唯一神经元数
        smooth_window: 移动平均窗口大小（默认 5）
        z_threshold: Z-score 阈值（默认 2.5）
        min_jump: 最小跳跃阈值（默认 10）

    Returns:
        Kink 列表：[(token_idx, slope_before, slope_after), ...]
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    neuron_counts = np.asarray(neuron_counts, dtype=np.float64)

    if len(tokens) < 10 or len(neuron_counts) < 10:
        return []

    # 1. 移动平均平滑
    smoothed = moving_average(neuron_counts, smooth_window)

    # 2. 计算斜率
    dx = np.diff(tokens)
    dx = np.maximum(dx, 1e-8)
    dy = np.diff(smoothed)
    slopes = dy / dx

    # 3. 计算斜率变化
    slope_changes = np.diff(slopes)

    # 4. MAD 异常检测
    mad = np.median(np.abs(slope_changes - np.median(slope_changes)))
    if mad < 1e-8:
        return []

    z_scores = 0.6745 * (slope_changes - np.median(slope_changes)) / mad

    # 5. 检测 kink：z_score > threshold 且 满足最小跳跃
    kinks: List[Tuple[int, float, float]] = []
    for i in range(len(z_scores)):
        if z_scores[i] > z_threshold:
            # i 对应 slopes[i], slopes[i+1]；tokens 对应 i+1
            if (dy[i+1] - dy[i]) >= min_jump:
                kinks.append((int(tokens[i+1]), float(slopes[i]), float(slopes[i+1])))
    return kinks


# ============================================================================
# KinkSelector 类（NAD 选择器插件）
# ============================================================================

class KinkSelector(Selector):
    """
    基于 Token-per-Kink 的选择器（Fewer Kinks Better）。

    策略：选择 tokens-per-kink 最大的样本（即 kink 数量相对较少的样本）。
    Kink 定义：(uniq count) 序列中的跳变点，基于 MAD 阈值检测。

    版本 2.0: 使用 step3 (k_token_per_kink.py) 的完整检测算法
    """

    def __init__(self, smooth_window: int = 5, z_threshold: float = 2.5, min_jump: int = 10):
        """
        Args:
            smooth_window: 移动平均窗口大小（默认 5，与 step3 一致）
            z_threshold: Z-score 阈值（默认 2.5，与 step3 一致）
            min_jump: 最小跳跃阈值（默认 10，与 step3 一致）
        """
        self.smooth_window = smooth_window
        self.z_threshold = z_threshold
        self.min_jump = min_jump
        self._context: SelectorContext | None = None

    def bind(self, context: SelectorContext) -> None:
        """接收 NAD 管线注入的上下文（cache, run_ids, views, pos_window 等）"""
        self._context = context

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int:
        """
        选择 Token-per-Kink 最大的样本（kink 数量最少 = 更稳定）。

        Args:
            D: [R, R] 距离矩阵（本选择器忽略）
            run_stats: {"lengths": [...], "views": [...]}

        Returns:
            int: 组内索引（0 到 R-1）
        """
        if self._context is None:
            raise RuntimeError("KinkSelector 需要先调用 bind() 注入上下文")

        cache = self._context.cache
        run_ids = self._context.run_ids
        pos_window = self._context.pos_window
        pos_size = self._context.pos_size

        # 对每个 run 计算 Token-per-Kink
        tpk_scores = []
        for rid in run_ids:
            # 从 CacheReader 获取row-bank数组（如果存在）
            if (cache.rows_sample_row_ptr is None or
                cache.rows_row_ptr is None or
                cache.rows_keys is None):
                # Row-bank 不可用，使用简单的长度作为fallback
                tpk_scores.append(1.0)
                continue

            # 调用通用组件：extract_tokenwise_counts
            tokens, uniq_counts = extract_tokenwise_counts(
                run_id=rid,
                rows_srp=cache.rows_sample_row_ptr,
                rows_rp=cache.rows_row_ptr,
                rows_keys=cache.rows_keys,
                rows_slice_ids=cache.rows_slice_ids,
                rows_trp=cache.rows_token_row_ptr,
                token_axis='row'
            )

            # 使用完整的 kink 检测算法（与 step3 一致）
            kinks = detect_kinks_mad_legacy(
                tokens, uniq_counts,
                smooth_window=self.smooth_window,
                z_threshold=self.z_threshold,
                min_jump=self.min_jump
            )
            num_kinks = len(kinks)

            # Token-per-Kink 计数方法（与 k_token_per_kink.py 保持一致）
            # 注意：必须使用 tokens[-1] + 1 而非 len(uniq_counts)
            # 原因：在稀疏数据中（某些 row 位置没有激活），两者会产生不同的结果：
            #   - tokens[-1] + 1: 表示从 0 到最大位置索引的完整范围
            #   - len(uniq_counts): 只统计有激活数据的位置数量
            # 使用 tokens[-1] + 1 与 step3 的计数方法一致，确保准确率对齐（80%）
            if len(tokens) > 0:
                n_tokens = int(tokens[-1]) + 1  # 使用最大索引 + 1
            else:
                n_tokens = 0

            if num_kinks > 0:
                tpk = n_tokens / num_kinks
            else:
                tpk = float('inf')  # 无 kink

            tpk_scores.append(tpk)

        # 选择 Token-per-Kink 最大的样本（kink 最少）
        # 注意：与 k_token_per_kink.py 保持一致，需要先过滤掉 inf 值
        # 原因：num_kinks = 0 (TPK = inf) 意味着完全没有突变点，可能不是好的选择
        # step3 的做法是：在有 kink 的样本中，选择 kink 最少的
        tpk_scores_array = np.array(tpk_scores)
        valid_indices = np.where(np.isfinite(tpk_scores_array))[0]

        if len(valid_indices) == 0:
            # 所有样本的 TPK 都是 inf（都没有 kink），fallback 到第一个样本
            return 0

        # 在有限值中选择最大的 TPK
        valid_tpk = tpk_scores_array[valid_indices]
        best_valid_idx = int(np.argmax(valid_tpk))
        best_idx = int(valid_indices[best_valid_idx])
        return best_idx
