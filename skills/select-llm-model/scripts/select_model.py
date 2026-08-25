#!/usr/bin/env python3
"""Select Sol, Terra, or Luna from a credential config and optionally call it."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROUTES = ("sol", "terra", "luna")
EXPECTED_MODELS = {
    "sol": "gpt-5.6-sol",
    "terra": "gpt-5.6-terra",
    "luna": "gpt-5.6-luna",
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
class ModelConfig:
    route: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = field(default=None, repr=False)

    def resolve_api_key(self) -> str:
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if not value:
                raise ValueError(f"environment variable {self.api_key_env} is not set")
            return value
        if self.api_key:
            return self.api_key
        raise ValueError(f"no API key configured for {self.route}")


@dataclass(frozen=True)
class RouterConfig:
    models: Mapping[str, ModelConfig]


@dataclass(frozen=True)
class Selection:
    model: str
    reason: str
    fallback: str
    route: str


def load_config(path: str | Path) -> RouterConfig:
    """Load Sol, Terra, and Luna model/key mappings from JSON."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model config: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("models"), Mapping):
        raise TypeError("config must contain a 'models' object")

    records = payload["models"]
    if set(records) != set(ROUTES):
        raise ValueError("config models must contain exactly sol, terra, and luna")
    models: dict[str, ModelConfig] = {}
    for route in ROUTES:
        record = records[route]
        if not isinstance(record, Mapping):
            raise TypeError(f"models.{route} must be an object")
        model = record.get("model")
        if model != EXPECTED_MODELS[route]:
            raise ValueError(f"models.{route}.model must be {EXPECTED_MODELS[route]}")
        api_key_env = record.get("api_key_env")
        api_key = record.get("api_key")
        if api_key_env is not None and (
            not isinstance(api_key_env, str) or not api_key_env.strip()
        ):
            raise ValueError(f"models.{route}.api_key_env must be a non-empty string")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError(f"models.{route}.api_key must be a non-empty string")
        if bool(api_key_env) == bool(api_key):
            raise ValueError(f"models.{route} must contain exactly one of api_key_env or api_key")
        models[route] = ModelConfig(route, model, api_key_env, api_key)

    return RouterConfig(models)


def infer_complexity(task: str) -> str:
    lowered = task.lower()
    if any(signal in lowered for signal in HIGH_SIGNALS):
        return "high"
    if any(signal in lowered for signal in LOW_SIGNALS):
        return "low"
    return "medium"


def select_model(
    task: str,
    config: RouterConfig,
    priority: str = "auto",
    complexity: str = "auto",
) -> Selection:
    if not task.strip():
        raise ValueError("task must not be empty")
    resolved_complexity = infer_complexity(task) if complexity == "auto" else complexity

    if priority == "quality" or resolved_complexity == "high":
        route = "sol"
        reason = "The task needs the strongest reasoning and coding route."
    elif priority in {"speed", "cost"} or (priority == "auto" and resolved_complexity == "low"):
        route = "luna"
        reason = "The task is low-complexity or cost-sensitive, so it uses the economical route."
    else:
        route = "terra"
        reason = "The task is standard work that benefits from balanced capability and cost."

    fallback_routes = {
        "sol": ("terra", "luna"),
        "terra": ("sol", "luna"),
        "luna": ("terra", "sol"),
    }
    selected = config.models[route]
    fallback = config.models[fallback_routes[route][0]]
    return Selection(selected.model, reason, fallback.model, route)


def _response_text(payload: Mapping[str, Any]) -> str:
    for item in payload.get("output", []):
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("API response did not contain output text")


def execute_task(task: str, selection: Selection, config: RouterConfig) -> str:
    """Call the Responses API with the selected model and its configured key."""
    model_config = config.models[selection.route]
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps({"model": selection.model, "input": task, "store": False}).encode(),
        headers={
            "Authorization": f"Bearer {model_config.resolve_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise ValueError(f"model request failed: {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError("model request returned an invalid response")
    return _response_text(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON model/API-key mapping")
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--priority", choices=("auto", "quality", "balanced", "speed", "cost"), default="auto"
    )
    parser.add_argument("--complexity", choices=("auto", "low", "medium", "high"), default="auto")
    parser.add_argument(
        "--execute", action="store_true", help="send the task to the selected model"
    )
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        selection = select_model(args.task, config, args.priority, args.complexity)
        answer = execute_task(args.task, selection, config) if args.execute else None
    except (TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error

    public_selection = asdict(selection)
    public_selection.pop("route")
    if args.format == "json":
        output: dict[str, Any] = public_selection
        if answer is not None:
            output["output"] = answer
        print(json.dumps(output, sort_keys=True))
    else:
        print(f"Model: {selection.model}")
        print(f"Why: {selection.reason}")
        print(f"Fallback: {selection.fallback}")
        if answer is not None:
            print(f"Output: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
