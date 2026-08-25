import math

from llm_route_opt.data import synthetic_dataset
from llm_route_opt.deployment import DeploymentOptimizer, Workload
from llm_route_opt.queueing import fcfs_mm_c


def test_fcfs_wait_is_finite_only_when_stable() -> None:
    stable = fcfs_mm_c(3, 2, 2, 100)
    unstable = fcfs_mm_c(4, 2, 2, 100)
    assert stable.stable and stable.waiting_time_ms > 0
    assert not unstable.stable and math.isinf(unstable.waiting_time_ms)


def test_deployment_assigns_quality_and_reports_wait() -> None:
    dataset = synthetic_dataset()
    optimizer = DeploymentOptimizer(
        tuple(dataset.models.values()), {"small": 0.4, "medium": 1.1, "large": 3.5}
    )
    plan = optimizer.optimize([Workload("chat", 5, 0.7, 500), Workload("hard", 0.8, 0.9, 1500)], 8)
    assert plan.feasible
    assert plan.assignments["hard"] == "large"
    assert plan.hourly_cost <= 8
    assert set(plan.waiting_time_ms) == set(plan.assignments.values())
