import json

import pytest

from llm_route_opt.benchmark import evaluate, load_routerbench_jsonl, write_routerbench_jsonl
from llm_route_opt.data import synthetic_dataset
from llm_route_opt.routers import SingleModelRouter
from llm_route_opt.schemas import QueryFeatures


def test_query_validation() -> None:
    with pytest.raises(ValueError):
        QueryFeatures("bad", "task", -1, 2)


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    original = synthetic_dataset()
    write_routerbench_jsonl(original, path)
    loaded = load_routerbench_jsonl(path)
    assert loaded == original
    assert all(json.loads(line)["type"] for line in path.read_text().splitlines())


def test_evaluation_reports_quality_cost_latency() -> None:
    dataset = synthetic_dataset()
    result = evaluate(dataset, SingleModelRouter(dataset.models["small"]))
    assert result.queries == 4
    assert 0 < result.mean_quality < 1
    assert result.total_cost > 0
    assert result.mean_latency_ms > 0
    assert result.model_counts == {"small": 4}
