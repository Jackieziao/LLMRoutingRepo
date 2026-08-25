import pytest

from llm_route_opt.data import synthetic_dataset
from llm_route_opt.routers import CascadeRouter, DAGEdge, DAGRouter, TopKRouter


def test_top_k_is_ordered_and_bounded() -> None:
    dataset = synthetic_dataset()
    decision = TopKRouter(tuple(dataset.models.values()), 2).route(dataset.queries[0])
    assert len(decision.candidates) == 2
    assert decision.selected_model == decision.candidates[0]


def test_cascade_has_deterministic_path() -> None:
    dataset = synthetic_dataset()
    router = CascadeRouter(tuple(dataset.models.values()), 0.8)
    assert router.route(dataset.queries[0]).path == ("small", "medium")
    assert router.route(dataset.queries[-1]).path == ("small", "medium", "large")


def test_dag_uses_first_matching_edge() -> None:
    dataset = synthetic_dataset()
    router = DAGRouter(
        dataset.models,
        "small",
        [
            DAGEdge("small", "large", lambda query: query.difficulty >= 0.8),
            DAGEdge("small", "medium", lambda query: query.difficulty >= 0.4),
        ],
    )
    assert router.route(dataset.queries[-1]).selected_model == "large"
    assert router.route(dataset.queries[1]).selected_model == "small"


def test_dag_rejects_cycle() -> None:
    dataset = synthetic_dataset()
    with pytest.raises(ValueError, match="cycle"):
        DAGRouter(
            dataset.models,
            "small",
            [
                DAGEdge("small", "medium", lambda query: True),
                DAGEdge("medium", "small", lambda query: True),
            ],
        )
