
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class SelectorSpec:
    name: str
    params: Dict[str, Any] | None = None

@dataclass(frozen=True)
class SelectorContext:
    """
    供用户自定义选择器使用的上下文（不包含任何用户算法）。
    注意：这是稳定的"接口类型"，可以安全暴露给仓库外部的插件代码。
    """
    cache: Any                           # NAD CacheReader (avoid circular import)
    problem_id: str                      # 当前问题ID
    run_ids: List[int]                   # 当前问题下的全局 run_id 列表
    views: List[Any]                     # 对应 run_id 的视图（keys/weights），与 run_ids 对齐
    pos_window: Optional[Tuple[int,int]] = None  # (lo, hi) 以 position 为单位；None 代表全序列
    pos_size: int = 32

class Selector:
    """
    Base class for run selection strategies.

    Contract (IMPORTANT):
    - The select() method receives a distance matrix D of shape (n, n) where n is the
      number of runs in the current problem group.
    - The method MUST return **group-internal indices** (0 to n-1), NOT global run_ids.
    - The analysis pipeline will map these indices to actual global run_ids using the
      run_ids list for the problem group.
    - Return type can be:
        * Single int: one selected run's index
        * List[int]: multiple selected runs' indices

    Args:
        D: Distance matrix of shape (n, n) for n runs in the group
        run_stats: Dictionary containing run metadata, e.g., {"lengths": np.ndarray}

    Returns:
        int or list[int]: Group-internal index/indices (0 to n-1)

    Example:
        If a problem group has run_ids = [100, 101, 102, 103] and the selector
        returns index 2, the analysis pipeline will map this to global run_id 102.
    """
    def bind(self, context: 'SelectorContext') -> None:
        """可选：在调用 select() 之前由管线注入上下文。
        用户插件可以覆写本方法来获取 cache/run_ids/窗口等信息。
        核心内置选择器不需要上下文时可直接忽略本方法。
        """
        # 默认实现：将 context 存到实例属性，便于子类访问
        self._context = context
        return

    def select(self, D: np.ndarray, run_stats: Dict[str, np.ndarray]) -> int | list[int]:
        """
        Select run(s) based on distance matrix and statistics.

        Returns group-internal indices (0 to n-1), which will be mapped to global
        run_ids by the analysis pipeline.
        """
        raise NotImplementedError
