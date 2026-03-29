"""
ML-based selectors.

All three selectors load a pre-trained scikit-learn model from
  models/ml_selectors/<name>.pkl
Train models with:
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
    Ridge regression on group-normalised features.
    Outputs a real-valued score per run; selects argmax.

    Model: sklearn Pipeline(StandardScaler, Ridge) trained to predict is_correct.
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
    Logistic regression on group-normalised features.
    Outputs P(correct) per run via predict_proba; selects argmax.

    Model: sklearn Pipeline(StandardScaler, LogisticRegression) trained on is_correct labels.
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
    Isotonic calibration on a single base score.

    Takes a base selector's rank score (already in [0,1]) and maps it through
    a fitted isotonic regression to get calibrated P(correct).
    Selects argmax of calibrated probability.

    Parameters
    ----------
    base : "medoid" | "deepconf"
        Which base score to calibrate.
        - "medoid"   : rank of mean distance within group (higher = more central)
        - "deepconf" : rank of DeepConf quality score (higher = more confident)
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
