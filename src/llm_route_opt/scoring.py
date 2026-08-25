"""Deterministic default estimators; callers can inject learned estimators."""

from __future__ import annotations

from collections.abc import Callable

from .schemas import ModelProfile, QueryFeatures

QualityEstimator = Callable[[QueryFeatures, ModelProfile], float]


def default_quality(query: QueryFeatures, model: ModelProfile) -> float:
    return max(0.0, min(1.0, model.quality - query.difficulty * (1 - model.quality)))
