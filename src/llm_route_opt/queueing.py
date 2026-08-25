"""FCFS M/M/c waiting-time calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FCFSResult:
    utilization: float
    probability_wait: float
    waiting_time_ms: float
    response_time_ms: float
    stable: bool


def fcfs_mm_c(
    arrival_rate_rps: float,
    service_rate_rps: float,
    servers: int,
    service_latency_ms: float,
) -> FCFSResult:
    """Erlang-C metrics for a stationary FCFS M/M/c queue."""

    if arrival_rate_rps < 0 or service_rate_rps <= 0 or servers <= 0:
        raise ValueError("arrival must be non-negative; service rate and servers positive")
    offered_load = arrival_rate_rps / service_rate_rps
    utilization = offered_load / servers
    if utilization >= 1:
        return FCFSResult(utilization, 1.0, math.inf, math.inf, False)
    if arrival_rate_rps == 0:
        return FCFSResult(0.0, 0.0, 0.0, service_latency_ms, True)
    finite_sum = sum(offered_load**n / math.factorial(n) for n in range(servers))
    tail = offered_load**servers / (math.factorial(servers) * (1 - utilization))
    probability_wait = tail / (finite_sum + tail)
    waiting_ms = probability_wait / (servers * service_rate_rps - arrival_rate_rps) * 1000
    return FCFSResult(
        utilization, probability_wait, waiting_ms, service_latency_ms + waiting_ms, True
    )
