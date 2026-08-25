"""CLI for reproducible evaluation and optimization runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkDataset, evaluate, load_routerbench_jsonl
from .data import synthetic_dataset
from .deployment import DeploymentOptimizer, Workload
from .inverse import inverse_example
from .optimization import maximize_quality
from .routers import CascadeRouter, Router, SingleModelRouter, TopKRouter


def _dataset(path: str | None) -> BenchmarkDataset:
    return load_routerbench_jsonl(Path(path)) if path else synthetic_dataset()


def _router(args: argparse.Namespace, dataset: BenchmarkDataset) -> Router:
    models = tuple(dataset.models.values())
    if args.router == "single":
        model_id = args.model or models[0].model_id
        if model_id not in dataset.models:
            raise ValueError(f"unknown model: {model_id}")
        return SingleModelRouter(dataset.models[model_id])
    if args.router == "top-k":
        return TopKRouter(models, args.k)
    return CascadeRouter(models, args.threshold)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-route-opt")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate", help="evaluate a router")
    evaluate_parser.add_argument(
        "--data", help="normalized RouterBench JSONL; synthetic by default"
    )
    evaluate_parser.add_argument(
        "--router", choices=("single", "top-k", "cascade"), default="cascade"
    )
    evaluate_parser.add_argument("--model", help="model id for the single router")
    evaluate_parser.add_argument("--k", type=int, default=2)
    evaluate_parser.add_argument("--threshold", type=float, default=0.80)

    optimize_parser = commands.add_parser("optimize", help="run an exact optimization example")
    optimize_parser.add_argument("kind", choices=("routing", "deployment", "inverse"))
    optimize_parser.add_argument(
        "--data", help="normalized RouterBench JSONL; synthetic by default"
    )
    optimize_parser.add_argument("--budget", type=float, default=0.03)
    optimize_parser.add_argument("--max-latency", type=float, default=600.0)

    commands.add_parser("demo", help="deterministic end-to-end report")
    return parser


def _optimize(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "inverse":
        return {"kind": "inverse", **asdict(inverse_example())}
    dataset = _dataset(args.data)
    if args.kind == "routing":
        result = maximize_quality(
            dataset.queries,
            tuple(dataset.models.values()),
            total_budget=args.budget,
            max_latency_ms=args.max_latency,
        )
        return {"kind": "routing", **asdict(result)}
    optimizer = DeploymentOptimizer(
        tuple(dataset.models.values()),
        {"small": 0.40, "medium": 1.10, "large": 3.50},
        max_replicas_per_model=8,
    )
    plan = optimizer.optimize(
        (
            Workload("interactive", 5.0, 0.70, 500.0),
            Workload("reasoning", 0.8, 0.90, 1500.0),
        ),
        hourly_budget=max(args.budget, 8.0),
    )
    return {"kind": "deployment", **asdict(plan)}


def deterministic_demo() -> dict[str, Any]:
    dataset = synthetic_dataset()
    router = CascadeRouter(tuple(dataset.models.values()), quality_threshold=0.80)
    result = evaluate(dataset, router)
    optimized = maximize_quality(
        dataset.queries,
        tuple(dataset.models.values()),
        total_budget=0.03,
        max_latency_ms=600.0,
    )
    return {
        "evaluation": result.to_dict(),
        "routing_optimization": asdict(optimized),
        "units": {"cost": "USD", "latency": "milliseconds", "quality": "0-1"},
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            dataset = _dataset(args.data)
            _print(evaluate(dataset, _router(args, dataset)).to_dict())
        elif args.command == "optimize":
            _print(_optimize(args))
        else:
            _print(deterministic_demo())
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
