"""
基于机器学习的选择器 | ML-based selectors.

三个选择器均从预训练的 scikit-learn 模型文件加载，路径为：
All three selectors load a pre-trained scikit-learn model from:
  models/ml_selectors/<name>.pkl

使用以下脚本训练模型 | Train models with:
  python scripts/train_ml_selectors.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from .base import Selector
from .ml_features import extract_run_features, N_FEATURES

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "ml_selectors"


def _load_model(path: Path):
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ML selector model from {path}.\n"
            f"Run: python scripts/train_ml_selectors.py\nOriginal error: {e}"
        )


class LinearProbeSelector(Selector):
    """
    线性探针选择器：对组内归一化特征做 Ridge 回归，输出每个 run 的实数分数，取 argmax。
    Ridge regression on group-normalised features. Outputs a real-valued score per run; selects argmax.

    模型 | Model: sklearn Pipeline(StandardScaler, Ridge)，训练目标为预测 is_correct。
    Trained to predict is_correct.
    """
    def __init__(self, model_path: str | None = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "linear_probe.pkl"
        self._model = None          # lazy load
        self._context = None

    def bind(self, context):
        self._context = context

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats):
        feat = extract_run_features(D, run_stats, self._context)
        scores = self._get_model().predict(feat)
        return int(np.argmax(scores))


class LogisticSelector(Selector):
    """
    逻辑回归选择器：对组内归一化特征做逻辑回归，输出每个 run 的 P(正确)，取 argmax。
    Logistic regression on group-normalised features. Outputs P(correct) per run; selects argmax.

    模型 | Model: sklearn Pipeline(StandardScaler, LogisticRegression)，
    训练目标为 is_correct 标签 | trained on is_correct labels.
    """
    def __init__(self, model_path: str | None = None):
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / "logistic.pkl"
        self._model = None
        self._context = None

    def bind(self, context):
        self._context = context

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats):
        feat = extract_run_features(D, run_stats, self._context)
        probs = self._get_model().predict_proba(feat)[:, 1]   # P(correct)
        return int(np.argmax(probs))


class IsotonicCalibratedSelector(Selector):
    """
    等渗校准选择器：将某个基础分数（已归一化到 [0,1]）通过拟合好的等渗回归映射为
    P(正确)，取 argmax。
    Isotonic calibration on a single base score. Maps a base selector's rank score
    (in [0,1]) through a fitted isotonic regression to get calibrated P(correct).
    Selects argmax of calibrated probability.

    参数 | Parameters
    ----------
    base : "medoid" | "deepconf"
        使用哪个基础分数进行校准 | Which base score to calibrate.
        - "medoid"   : 组内平均距离的 rank（越大越居中）
                       rank of mean distance within group (higher = more central)
        - "deepconf" : DeepConf quality 分数的 rank（越大越自信）
                       rank of DeepConf quality score (higher = more confident)
    """
    _BASE_FEATURE_IDX = {"medoid": 1, "deepconf": 7}   # col index in feature matrix

    def __init__(self, base: str = "medoid", model_path: str | None = None):
        if base not in self._BASE_FEATURE_IDX:
            raise ValueError(f"base must be 'medoid' or 'deepconf', got '{base}'")
        self.base = base
        fname = f"isotonic_{base}.pkl"
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_DIR / fname
        self._model = None
        self._context = None

    def bind(self, context):
        self._context = context

    def _get_model(self):
        if self._model is None:
            self._model = _load_model(self._model_path)
        return self._model

    def select(self, D: np.ndarray, run_stats):
        feat = extract_run_features(D, run_stats, self._context)
        col = self._BASE_FEATURE_IDX[self.base]
        base_scores = feat[:, col]                         # shape (n,)
        calibrated = self._get_model().predict(base_scores)
        return int(np.argmax(calibrated))
