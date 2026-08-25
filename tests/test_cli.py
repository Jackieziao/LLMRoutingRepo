import json

from llm_route_opt.cli import main


def test_demo_reports_required_metrics(capsys) -> None:
    assert main(["demo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    evaluation = payload["evaluation"]
    assert evaluation["mean_quality"] > 0
    assert evaluation["total_cost"] > 0
    assert evaluation["mean_latency_ms"] > 0


def test_optimize_inverse(capsys) -> None:
    assert main(["optimize", "inverse"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pairwise_accuracy"] == 1
