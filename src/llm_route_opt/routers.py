"""Single, top-k, cascade, and deterministic DAG routing policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .schemas import ModelProfile, QueryFeatures, RouteDecision
from .scoring import QualityEstimator, default_quality


class Router(Protocol):
    def route(self, query: QueryFeatures) -> RouteDecision: ...


class SingleModelRouter:
    def __init__(self, model: ModelProfile) -> None:
        self.model = model

    def route(self, query: QueryFeatures) -> RouteDecision:
        del query
        return RouteDecision(
            (self.model.model_id,), self.model.model_id, (self.model.model_id,), "fixed"
        )


class TopKRouter:
    """Rank models by predicted quality and execute the first candidate."""

    def __init__(
        self,
        models: Sequence[ModelProfile],
        k: int,
        estimator: QualityEstimator = default_quality,
    ) -> None:
        if not models or not 1 <= k <= len(models):
            raise ValueError("k must be between 1 and the model count")
        self.models = tuple(models)
        self.k = k
        self.estimator = estimator

    def route(self, query: QueryFeatures) -> RouteDecision:
        ranked = sorted(
            self.models,
            key=lambda model: (-self.estimator(query, model), model.cost(query), model.model_id),
        )[: self.k]
        candidates = tuple(model.model_id for model in ranked)
        return RouteDecision(candidates, candidates[0], (candidates[0],), f"top-{self.k}")


class CascadeRouter:
    """Traverse a fixed cheap-to-capable cascade until the quality gate passes."""

    def __init__(
        self,
        models: Sequence[ModelProfile],
        quality_threshold: float,
        estimator: QualityEstimator = default_quality,
    ) -> None:
        if not models or not 0 <= quality_threshold <= 1:
            raise ValueError("models and a threshold in [0, 1] are required")
        self.models = tuple(models)
        self.threshold = quality_threshold
        self.estimator = estimator

    def route(self, query: QueryFeatures) -> RouteDecision:
        path: list[str] = []
        selected = self.models[-1]
        for model in self.models:
            path.append(model.model_id)
            selected = model
            if self.estimator(query, model) >= self.threshold:
                break
        candidates = tuple(model.model_id for model in self.models)
        return RouteDecision(candidates, selected.model_id, tuple(path), "cascade quality gate")


@dataclass(frozen=True, slots=True)
class DAGEdge:
    source: str
    target: str
    predicate: Callable[[QueryFeatures], bool]
    label: str = "condition"


class DAGRouter:
    """Deterministic acyclic decision graph with ordered outgoing edges."""

    def __init__(
        self,
        models: Mapping[str, ModelProfile],
        root: str,
        edges: Sequence[DAGEdge],
    ) -> None:
        if root not in models:
            raise ValueError("root must name a model")
        self.models = dict(models)
        self.root = root
        self.edges: dict[str, list[DAGEdge]] = {}
        for edge in edges:
            if edge.source not in models or edge.target not in models:
                raise ValueError("edge endpoints must name models")
            self.edges.setdefault(edge.source, []).append(edge)
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("routing graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for edge in self.edges.get(node, []):
                visit(edge.target)
            visiting.remove(node)
            visited.add(node)

        visit(self.root)

    def route(self, query: QueryFeatures) -> RouteDecision:
        node = self.root
        path = [node]
        while True:
            match = next((edge for edge in self.edges.get(node, []) if edge.predicate(query)), None)
            if match is None:
                break
            node = match.target
            path.append(node)
        targets = (edge.target for edges in self.edges.values() for edge in edges)
        candidates = tuple(dict.fromkeys((self.root, *targets)))
        return RouteDecision(candidates, node, tuple(path), "DAG terminal node")
