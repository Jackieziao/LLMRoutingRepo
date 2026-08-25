"""Small deterministic public dataset used by docs, CLI, and tests."""

from __future__ import annotations

from .benchmark import BenchmarkDataset
from .schemas import ModelProfile, QueryFeatures, RouteMeasurement
from .scoring import default_quality


def synthetic_dataset() -> BenchmarkDataset:
    models = {
        model.model_id: model
        for model in (
            ModelProfile("small", 0.72, 0.20, 0.80, 80.0, 8.0),
            ModelProfile("medium", 0.86, 1.00, 3.00, 210.0, 4.0),
            ModelProfile("large", 0.95, 4.00, 12.00, 580.0, 1.5),
        )
    }
    queries = (
        QueryFeatures("q1", "classification", 400, 20, 0.10),
        QueryFeatures("q2", "summarization", 1200, 180, 0.35),
        QueryFeatures("q3", "reasoning", 900, 350, 0.70),
        QueryFeatures("q4", "code", 1600, 500, 0.90),
    )
    measurements: dict[tuple[str, str], RouteMeasurement] = {}
    for query_index, query in enumerate(queries):
        for model_index, model in enumerate(models.values()):
            # Fixed perturbations mimic repeatable empirical outcomes.
            perturbation = ((query_index + 2 * model_index) % 3 - 1) * 0.01
            quality = max(0.0, min(1.0, default_quality(query, model) + perturbation))
            latency = model.latency_ms * (1 + 0.04 * query_index)
            measurement = RouteMeasurement(
                query.query_id, model.model_id, quality, model.cost(query), latency
            )
            measurements[(query.query_id, model.model_id)] = measurement
    return BenchmarkDataset(queries, models, measurements)
