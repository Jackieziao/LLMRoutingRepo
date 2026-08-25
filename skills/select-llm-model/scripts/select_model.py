#!/usr/bin/env python3
"""Deterministically select one model for a natural-language task."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

SOL = "gpt-5.6-sol"
TERRA = "gpt-5.6-terra"
LUNA = "luna"

ALIASES = {
    "sol": SOL,
    "5.6-sol": SOL,
    "5.6 sol": SOL,
    SOL: SOL,
    "terra": TERRA,
    "5.6-terra": TERRA,
    "5.6 terra": TERRA,
    TERRA: TERRA,
    "luna": LUNA,
    "gpt-5.6-luna": LUNA,
}

HIGH_SIGNALS = (
    "architecture",
    "distributed",
    "security",
    "vulnerability",
    "concurrency",
    "race condition",
    "production incident",
    "root cause",
    "formal proof",
    "hard algorithm",
    "deep research",
    "migration plan",
    "optimize performance",
)
LOW_SIGNALS = (
    "summarize",
    "classify",
    "extract",
    "format",
    "rename",
    "typo",
    "boilerplate",
    "short translation",
    "simple conversion",
)


@dataclass(frozen=True)
class Selection:
    model: str
    reason: str
    fallback: str | None


def normalize_available(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        key = value.strip().lower()
        model = ALIASES.get(key, value.strip())
        if model and model not in normalized:
            normalized.append(model)
    if not normalized:
        raise ValueError("at least one available model is required")
    return tuple(normalized)


def infer_complexity(task: str) -> str:
    lowered = task.lower()
    if any(signal in lowered for signal in HIGH_SIGNALS):
        return "high"
    if any(signal in lowered for signal in LOW_SIGNALS):
        return "low"
    return "medium"


def select_model(
    task: str,
    priority: str = "auto",
    complexity: str = "auto",
    available: Sequence[str] = (SOL, TERRA, LUNA),
) -> Selection:
    if not task.strip():
        raise ValueError("task must not be empty")
    catalog = normalize_available(available)
    resolved_complexity = infer_complexity(task) if complexity == "auto" else complexity

    if priority == "quality" or resolved_complexity == "high":
        desired = SOL
        reason = (
            "The task favors maximum reasoning depth and careful handling of complex constraints."
        )
    elif priority in {"speed", "cost"} or resolved_complexity == "low":
        desired = LUNA
        reason = "The task is low-complexity and favors fast, economical throughput."
    else:
        desired = TERRA
        reason = (
            "The task is standard engineering work with a balanced quality and latency profile."
        )

    fallback_order = {
        SOL: (TERRA, LUNA),
        TERRA: (SOL, LUNA),
        LUNA: (TERRA, SOL),
    }
    if desired in catalog:
        model = desired
    else:
        model = next(
            (candidate for candidate in fallback_order[desired] if candidate in catalog), catalog[0]
        )
        reason += f" {desired} is unavailable, so the nearest available route was selected."
    fallback = next(
        (candidate for candidate in fallback_order.get(model, ()) if candidate in catalog), None
    )
    return Selection(model, reason, fallback)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--priority", choices=("auto", "quality", "balanced", "speed", "cost"), default="auto"
    )
    parser.add_argument("--complexity", choices=("auto", "low", "medium", "high"), default="auto")
    parser.add_argument("--available", nargs="+", default=[SOL, TERRA, LUNA])
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = select_model(args.task, args.priority, args.complexity, args.available)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if args.format == "json":
        print(json.dumps(asdict(selection), sort_keys=True))
    else:
        print(f"Model: {selection.model}")
        print(f"Why: {selection.reason}")
        print(f"Fallback: {selection.fallback or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
