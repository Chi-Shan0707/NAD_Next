from __future__ import annotations

from typing import Any, Callable

import numpy as np

from nad.ops.earlystop_svd import (
    _build_representation,
    _predict_svd_lr,
    _rank_transform_matrix,
)


def _predict_positive_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    return np.asarray(model.predict(x), dtype=np.float64)


def collect_required_features(bundle: dict[str, Any]) -> set[str]:
    required: set[str] = set()
    for domain_bundle in bundle.get("domains", {}).values():
        for route in domain_bundle.get("routes", []):
            route_type = str(route.get("route_type", "svd"))
            if route_type == "baseline":
                required.add(str(route["signal_name"]))
            else:
                required.update(str(name) for name in route.get("feature_names", []))
    return required


def score_xraw_with_route(
    *,
    x_raw: np.ndarray,
    route: dict[str, Any],
    feature_to_idx: dict[str, int],
) -> np.ndarray:
    route_type = str(route.get("route_type", "svd"))
    if route_type == "baseline":
        score_col = feature_to_idx[str(route["signal_name"])]
        return np.asarray(x_raw[:, score_col], dtype=np.float64)

    x_rank = _rank_transform_matrix(x_raw)
    feature_indices = [int(v) for v in route.get("feature_indices", [])]
    representation = str(route.get("representation", route.get("feature_variant", "raw")))
    x_rep = _build_representation(
        x_raw=x_raw,
        x_rank=x_rank,
        feature_indices=feature_indices,
        representation=representation,
    )

    if route_type == "svd":
        return _predict_svd_lr(route["model"], x_rep)
    if route_type == "tree_ensemble":
        seed_scores = [
            _predict_positive_scores(model, x_rep)
            for model in list(route.get("models", []))
        ]
        if not seed_scores:
            return np.zeros((x_rep.shape[0],), dtype=np.float64)
        return np.mean(np.vstack(seed_scores), axis=0, dtype=np.float64)
    if route_type in {"pointwise", "ranksvm"}:
        return np.asarray(route["scorer"].score_group(x_rep), dtype=np.float64)
    raise ValueError(f"Unknown route type: {route_type}")


def make_mixed_bundle_score_fn(
    bundle: dict[str, Any],
) -> Callable[[str, int, np.ndarray], np.ndarray]:
    feature_to_idx = {str(name): idx for idx, name in enumerate(bundle["feature_names"])}

    def _score(domain: str, position_index: int, x_raw: np.ndarray) -> np.ndarray:
        route = bundle["domains"][str(domain)]["routes"][int(position_index)]
        return score_xraw_with_route(
            x_raw=x_raw,
            route=route,
            feature_to_idx=feature_to_idx,
        )

    return _score
