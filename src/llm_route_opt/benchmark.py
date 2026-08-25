"""RouterBench-compatible JSONL loading and offline evaluation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .routers import Router
from .schemas import ModelProfile, QueryFeatures, RouteMeasurement


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    queries: tuple[QueryFeatures, ...]
    models: dict[str, ModelProfile]
    measurements: dict[tuple[str, str], RouteMeasurement]

    def measurement(self, query_id: str, model_id: str) -> RouteMeasurement:
        try:
            return self.measurements[(query_id, model_id)]
        except KeyError as error:
            raise ValueError(
                f"missing measurement for query={query_id}, model={model_id}"
            ) from error


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    queries: int
    mean_quality: float
    total_cost: float
    mean_latency_ms: float
    p95_latency_ms: float
    model_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_routerbench_jsonl(path: str | Path) -> BenchmarkDataset:
    """Load normalized JSONL records of types query, model, and measurement.

    This long-form format retains RouterBench's per-prompt/per-model outcomes
    while separating reusable query features and model metadata.
    """

    queries: list[QueryFeatures] = []
    models: dict[str, ModelProfile] = {}
    measurements: dict[tuple[str, str], RouteMeasurement] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.pop("type", None)
            try:
                if kind == "query":
                    queries.append(QueryFeatures(**record))
                elif kind == "model":
                    model = ModelProfile(**record)
                    models[model.model_id] = model
                elif kind == "measurement":
                    measurement = RouteMeasurement(**record)
                    measurements[(measurement.query_id, measurement.model_id)] = measurement
                else:
                    raise ValueError(f"unknown record type: {kind}")
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid record at line {line_number}: {error}") from error
    if not queries or not models:
        raise ValueError("dataset needs at least one query and model")
    return BenchmarkDataset(tuple(queries), models, measurements)


def write_routerbench_jsonl(dataset: BenchmarkDataset, path: str | Path) -> None:
    records: Iterable[tuple[str, dict[str, Any]]] = (
        [("query", query.to_dict()) for query in dataset.queries]
        + [("model", model.to_dict()) for model in dataset.models.values()]
        + [("measurement", item.to_dict()) for item in dataset.measurements.values()]
    )
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(
            json.dumps({"type": kind, **record}, sort_keys=True) + "\n" for kind, record in records
        )


def evaluate(dataset: BenchmarkDataset, router: Router) -> EvaluationResult:
    outcomes: list[RouteMeasurement] = []
    counts: Counter[str] = Counter()
    for query in dataset.queries:
        model_id = router.route(query).selected_model
        outcomes.append(dataset.measurement(query.query_id, model_id))
        counts[model_id] += 1
    latencies = sorted(item.latency_ms for item in outcomes)
    p95_index = max(0, int(0.95 * len(latencies) + 0.999999) - 1)
    size = len(outcomes)
    return EvaluationResult(
        queries=size,
        mean_quality=sum(item.quality for item in outcomes) / size,
        total_cost=sum(item.cost for item in outcomes),
        mean_latency_ms=sum(item.latency_ms for item in outcomes) / size,
        p95_latency_ms=latencies[p95_index],
        model_counts=dict(sorted(counts.items())),
    )
