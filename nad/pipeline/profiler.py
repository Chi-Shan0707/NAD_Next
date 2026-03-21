"""
Performance profiler for NAD analysis pipeline.

Tracks time and memory usage across different stages of the analysis.
"""

from __future__ import annotations
import time
import gc
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available. Memory monitoring disabled.")


@dataclass
class StageMetrics:
    """Metrics for a single profiling stage"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    memory_samples: List[float] = field(default_factory=list)
    sub_stages: Dict[str, 'StageMetrics'] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds"""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    @property
    def memory_avg_mb(self) -> float:
        """Average memory in MB"""
        if not self.memory_samples:
            return 0.0
        return sum(self.memory_samples) / len(self.memory_samples) / (1024 * 1024)

    @property
    def memory_peak_mb(self) -> float:
        """Peak memory in MB"""
        if not self.memory_samples:
            return 0.0
        return max(self.memory_samples) / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "time_seconds": round(self.duration, 3),
            "memory_avg_mb": round(self.memory_avg_mb, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
        }
        if self.sub_stages:
            result["sub_stages"] = {
                name: stage.to_dict()
                for name, stage in self.sub_stages.items()
            }
        return result


class PerformanceMonitor:
    """Monitor performance of NAD analysis pipeline"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.stages: Dict[str, StageMetrics] = {}
        self.active_stages: List[StageMetrics] = []
        self.global_start_time = time.perf_counter()
        self.memory_samples: List[float] = []

    def _get_memory(self) -> float:
        """Get current memory usage in bytes (RSS)"""
        if not PSUTIL_AVAILABLE or not self.enabled:
            return 0.0
        try:
            return float(self.process.memory_info().rss)
        except Exception:
            return 0.0

    def _sample_memory(self, stage: Optional[StageMetrics] = None):
        """Sample current memory and store it"""
        mem = self._get_memory()
        self.memory_samples.append(mem)
        if stage is not None:
            stage.memory_samples.append(mem)

    @contextmanager
    def stage(self, name: str, sample_interval: int = 1):
        """
        Context manager for profiling a stage.

        Args:
            name: Stage name
            sample_interval: Memory sampling interval (every N operations)
        """
        if not self.enabled:
            yield None
            return

        # Force garbage collection for accurate memory measurement
        gc.collect()

        # Create stage metrics
        stage_metrics = StageMetrics(name=name, start_time=time.perf_counter())
        self._sample_memory(stage_metrics)

        # Add to active stages stack
        if self.active_stages:
            # This is a sub-stage
            parent = self.active_stages[-1]
            parent.sub_stages[name] = stage_metrics
        else:
            # This is a top-level stage
            self.stages[name] = stage_metrics

        self.active_stages.append(stage_metrics)

        logger.debug(f"Starting: {name}")

        try:
            yield stage_metrics
        finally:
            # Sample memory one more time
            self._sample_memory(stage_metrics)

            # Record end time
            stage_metrics.end_time = time.perf_counter()

            # Remove from active stages
            self.active_stages.pop()

            # Log stage summary
            logger.debug(f"Completed: {name} | "
                        f"Time: {stage_metrics.duration:.3f}s | "
                        f"Memory: {stage_metrics.memory_avg_mb:.1f}MB avg, "
                        f"{stage_metrics.memory_peak_mb:.1f}MB peak")

    def sample_memory_in_stage(self):
        """Sample memory for the current active stage"""
        if not self.enabled or not self.active_stages:
            return
        self._sample_memory(self.active_stages[-1])

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary as dictionary"""
        total_time = time.perf_counter() - self.global_start_time
        peak_memory_mb = max(self.memory_samples) / (1024 * 1024) if self.memory_samples else 0.0

        return {
            "total_time_seconds": round(total_time, 3),
            "total_memory_peak_mb": round(peak_memory_mb, 2),
            "stages": {
                name: stage.to_dict()
                for name, stage in self.stages.items()
            }
        }

    def print_summary(self):
        """Log formatted performance summary"""
        if not self.enabled:
            return

        summary = self.get_summary()

        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Time: {summary['total_time_seconds']:.3f}s")
        logger.info(f"Peak Memory: {summary['total_memory_peak_mb']:.1f}MB")
        logger.info("\nStage Breakdown:")
        logger.info("-" * 80)
        logger.info(f"{'Stage':<30} {'Time (s)':>12} {'Mem Avg (MB)':>15} {'Mem Peak (MB)':>15}")
        logger.info("-" * 80)

        for stage_name, stage_data in summary['stages'].items():
            logger.info(f"{stage_name:<30} {stage_data['time_seconds']:>12.3f} "
                       f"{stage_data['memory_avg_mb']:>15.1f} {stage_data['memory_peak_mb']:>15.1f}")

            # Log sub-stages if any
            if 'sub_stages' in stage_data:
                for sub_name, sub_data in stage_data['sub_stages'].items():
                    logger.info(f"  ├─ {sub_name:<26} {sub_data['time_seconds']:>12.3f} "
                               f"{sub_data['memory_avg_mb']:>15.1f} {sub_data['memory_peak_mb']:>15.1f}")

        logger.info("=" * 80 + "\n")
