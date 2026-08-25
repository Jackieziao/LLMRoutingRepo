"""Canonical, serializable schemas for offline routing experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class QueryFeatures:
    query_id: str
    task: str
    input_tokens: int
    expected_output_tokens: int
    difficulty: float = 0.5
    numeric: dict[str, float] = field(default_factory=dict)
    categorical: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id or not self.task:
            raise ValueError("query_id and task are required")
        if self.input_tokens < 0 or self.expected_output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if not 0 <= self.difficulty <= 1:
            raise ValueError("difficulty must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    quality: float
    input_cost_per_million: float
    output_cost_per_million: float
    latency_ms: float
    service_rate_rps: float
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if not 0 <= self.quality <= 1:
            raise ValueError("quality must be in [0, 1]")
        values = (
            self.input_cost_per_million,
            self.output_cost_per_million,
            self.latency_ms,
            self.service_rate_rps,
        )
        if any(value < 0 for value in values) or self.service_rate_rps == 0:
            raise ValueError("cost/latency must be non-negative and service rate positive")

    def cost(self, query: QueryFeatures) -> float:
        return (
            query.input_tokens * self.input_cost_per_million
            + query.expected_output_tokens * self.output_cost_per_million
        ) / 1_000_000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RouteMeasurement:
    query_id: str
    model_id: str
    quality: float
    cost: float
    latency_ms: float

    def __post_init__(self) -> None:
        if not 0 <= self.quality <= 1:
            raise ValueError("quality must be in [0, 1]")
        if self.cost < 0 or self.latency_ms < 0:
            raise ValueError("cost and latency must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Ordered candidates and deterministic executed model."""

    candidates: tuple[str, ...]
    selected_model: str
    path: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.candidates or self.selected_model not in self.candidates:
            raise ValueError("selected_model must occur in non-empty candidates")
