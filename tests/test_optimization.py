from llm_route_opt.data import synthetic_dataset
from llm_route_opt.optimization import maximize_quality


def test_maximize_quality_honors_budget_and_latency() -> None:
    dataset = synthetic_dataset()
    result = maximize_quality(
        dataset.queries,
        tuple(dataset.models.values()),
        total_budget=0.005,
        max_latency_ms=250,
    )
    assert result.feasible
    assert result.total_cost <= 0.005
    assert result.mean_latency_ms <= 250
    assert set(result.assignments.values()) <= {"small", "medium"}


def test_maximize_quality_reports_infeasible() -> None:
    dataset = synthetic_dataset()
    result = maximize_quality(dataset.queries, tuple(dataset.models.values()), 0, 10)
    assert not result.feasible
