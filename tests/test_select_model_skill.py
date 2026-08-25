import importlib.util
import json
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Self
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).parents[1] / "skills" / "select-llm-model" / "scripts" / "select_model.py"
SPEC = importlib.util.spec_from_file_location("select_model_skill", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_config(tmp_path: Path, *, direct_keys: bool = False) -> Path:
    models: dict[str, dict[str, str]] = {}
    for route, model in MODULE.EXPECTED_MODELS.items():
        credential = (
            {"api_key": f"secret-{route}"}
            if direct_keys
            else {"api_key_env": f"KEY_{route.upper()}"}
        )
        models[route] = {"model": model, **credential}
    path = tmp_path / "model-api-keys.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def test_config_routes_sol_terra_and_luna(tmp_path: Path) -> None:
    config = MODULE.load_config(write_config(tmp_path))

    assert MODULE.select_model("Review a security architecture", config).model == "gpt-5.6-sol"
    assert MODULE.select_model("Implement a typed endpoint", config).model == "gpt-5.6-terra"
    assert MODULE.select_model("Format this text", config).model == "gpt-5.6-luna"


def test_explicit_balanced_priority_selects_terra(tmp_path: Path) -> None:
    config = MODULE.load_config(write_config(tmp_path))

    selection = MODULE.select_model("Format this text", config, priority="balanced")

    assert selection.model == "gpt-5.6-terra"


def test_keys_are_resolved_only_for_selected_route(tmp_path: Path) -> None:
    config = MODULE.load_config(write_config(tmp_path))
    selection = MODULE.select_model("Format this text", config)

    with patch.dict("os.environ", {"KEY_LUNA": "luna-secret"}, clear=True):
        assert config.models[selection.route].resolve_api_key() == "luna-secret"
        with pytest.raises(ValueError, match="KEY_SOL"):
            config.models["sol"].resolve_api_key()


def test_direct_keys_are_not_exposed_by_selection(tmp_path: Path) -> None:
    config = MODULE.load_config(write_config(tmp_path, direct_keys=True))

    selection = MODULE.select_model("ordinary task", config)

    assert "secret" not in repr(config.models["terra"])
    assert "secret" not in json.dumps(
        {"model": selection.model, "reason": selection.reason, "fallback": selection.fallback}
    )


def test_execute_uses_selected_model_and_corresponding_key(tmp_path: Path) -> None:
    class FakeResponse(AbstractContextManager["FakeResponse"]):
        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"output": [{"content": [{"type": "output_text", "text": "done"}]}]}
            ).encode()

    config = MODULE.load_config(write_config(tmp_path))
    selection = MODULE.select_model("Format this text", config)
    captured_request = None

    def fake_urlopen(request: object, timeout: int) -> FakeResponse:
        nonlocal captured_request
        captured_request = request
        assert timeout == 120
        return FakeResponse()

    with (
        patch.dict("os.environ", {"KEY_LUNA": "luna-secret"}, clear=True),
        patch.object(MODULE.urllib.request, "urlopen", side_effect=fake_urlopen),
    ):
        assert MODULE.execute_task("Format this text", selection, config) == "done"

    assert captured_request is not None
    assert captured_request.get_header("Authorization") == "Bearer luna-secret"
    assert json.loads(captured_request.data)["model"] == "gpt-5.6-luna"


def test_config_requires_all_three_routes(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["models"]["luna"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly sol, terra, and luna"):
        MODULE.load_config(path)
