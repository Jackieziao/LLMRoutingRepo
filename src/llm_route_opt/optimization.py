"""Exact finite optimization of offline routes under resource constraints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

from .schemas import ModelProfile, QueryFeatures
from .scoring import QualityEstimator, default_quality


@dataclass(frozen=True, slots=True)
class RoutingOptimizationResult:
    assignments: dict[str, str]
    mean_quality: float
    total_cost: float
    mean_latency_ms: float
    feasible: bool


def maximize_quality(
    queries: Sequence[QueryFeatures],
    models: Sequence[ModelProfile],
    total_budget: float,
    max_latency_ms: float,
    estimator: QualityEstimator = default_quality,
) -> RoutingOptimizationResult:
    """Maximize total predicted quality with total-cost and per-query latency limits.

    Exact enumeration is intentionally transparent and appropriate for examples,
    regression baselines, and small candidate sets.
    """

    if not queries or not models:
        raise ValueError("queries and models must not be empty")
    if total_budget < 0 or max_latency_ms < 0:
        raise ValueError("constraints must be non-negative")
    eligible = [model for model in models if model.latency_ms <= max_latency_ms]
    if not eligible:
        return RoutingOptimizationResult({}, 0.0, 0.0, 0.0, False)
    best: RoutingOptimizationResult | None = None
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    for chosen in product(eligible, repeat=len(queries)):
        cost = sum(model.cost(query) for query, model in zip(queries, chosen, strict=True))
        if cost > total_budget + 1e-12:
            continue
        quality = sum(
            estimator(query, model) for query, model in zip(queries, chosen, strict=True)
        ) / len(queries)
        latency = sum(model.latency_ms for model in chosen) / len(chosen)
        # Deterministic tie-breaking prefers lower cost, then lower latency.
        key = (quality, -cost, -latency)
        if key > best_key:
            best_key = key
            best = RoutingOptimizationResult(
                {
                    query.query_id: model.model_id
                    for query, model in zip(queries, chosen, strict=True)
                },
                quality,
                cost,
                latency,
                True,
            )
    return best or RoutingOptimizationResult({}, 0.0, 0.0, 0.0, False)
