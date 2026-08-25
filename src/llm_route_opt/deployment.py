"""Exact workload assignment and integer replica planning under FCFS queues."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from .queueing import fcfs_mm_c
from .schemas import ModelProfile


@dataclass(frozen=True, slots=True)
class Workload:
    workload_id: str
    arrival_rate_rps: float
    min_quality: float
    max_response_ms: float

    def __post_init__(self) -> None:
        if not self.workload_id or self.arrival_rate_rps < 0:
            raise ValueError("workload id and non-negative arrival rate are required")
        if not 0 <= self.min_quality <= 1 or self.max_response_ms < 0:
            raise ValueError("invalid quality or response constraint")


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    assignments: dict[str, str]
    replicas: dict[str, int]
    hourly_cost: float
    mean_quality: float
    mean_response_ms: float
    waiting_time_ms: dict[str, float]
    feasible: bool


class DeploymentOptimizer:
    def __init__(
        self,
        models: Sequence[ModelProfile],
        replica_hourly_cost: Mapping[str, float],
        max_replicas_per_model: int = 16,
    ) -> None:
        if not models or max_replicas_per_model < 1:
            raise ValueError("models and positive replica limit are required")
        missing = {model.model_id for model in models} - replica_hourly_cost.keys()
        if missing:
            raise ValueError(f"missing replica cost for {sorted(missing)}")
        self.models = tuple(models)
        self.costs = dict(replica_hourly_cost)
        self.max_replicas = max_replicas_per_model

    def optimize(self, workloads: Sequence[Workload], hourly_budget: float) -> DeploymentPlan:
        if not workloads or hourly_budget < 0:
            raise ValueError("workloads and non-negative budget are required")
        best: DeploymentPlan | None = None
        best_key = (float("-inf"), float("-inf"), float("-inf"))
        for indices in product(range(len(self.models)), repeat=len(workloads)):
            if any(
                self.models[index].quality < workload.min_quality
                for workload, index in zip(workloads, indices, strict=True)
            ):
                continue
            loads = [0.0] * len(self.models)
            for workload, index in zip(workloads, indices, strict=True):
                loads[index] += workload.arrival_rate_rps
            replicas: dict[str, int] = {}
            queue_results = []
            valid = True
            for model, load in zip(self.models, loads, strict=True):
                if load == 0:
                    replicas[model.model_id] = 0
                    queue_results.append(fcfs_mm_c(0, model.service_rate_rps, 1, model.latency_ms))
                    continue
                count = max(1, math.floor(load / model.service_rate_rps) + 1)
                replicas[model.model_id] = count
                valid &= count <= self.max_replicas
                queue_results.append(
                    fcfs_mm_c(load, model.service_rate_rps, count, model.latency_ms)
                )
            hourly_cost = sum(
                replicas[model.model_id] * self.costs[model.model_id] for model in self.models
            )
            if not valid or hourly_cost > hourly_budget + 1e-12:
                continue
            responses = [queue_results[index].response_time_ms for index in indices]
            if any(
                response > workload.max_response_ms
                for workload, response in zip(workloads, responses, strict=True)
            ):
                continue
            total_rate = sum(workload.arrival_rate_rps for workload in workloads) or 1.0
            mean_quality = (
                sum(
                    workload.arrival_rate_rps * self.models[index].quality
                    for workload, index in zip(workloads, indices, strict=True)
                )
                / total_rate
            )
            mean_response = (
                sum(
                    workload.arrival_rate_rps * response
                    for workload, response in zip(workloads, responses, strict=True)
                )
                / total_rate
            )
            key = (mean_quality, -hourly_cost, -mean_response)
            if key > best_key:
                best_key = key
                best = DeploymentPlan(
                    assignments={
                        workload.workload_id: self.models[index].model_id
                        for workload, index in zip(workloads, indices, strict=True)
                    },
                    replicas=replicas,
                    hourly_cost=hourly_cost,
                    mean_quality=mean_quality,
                    mean_response_ms=mean_response,
                    waiting_time_ms={
                        model.model_id: queue_results[index].waiting_time_ms
                        for index, model in enumerate(self.models)
                        if loads[index] > 0
                    },
                    feasible=True,
                )
        return best or DeploymentPlan({}, {}, 0.0, 0.0, math.inf, {}, False)
