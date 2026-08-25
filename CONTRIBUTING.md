# Contributing

`llm-route-opt` welcomes focused additions that keep experiments reproducible and
provider-neutral.

1. Use Python 3.11+ and install with `python -m pip install -e ".[dev]"`.
2. Add tests for behavioral changes.
3. Run `pytest`, `ruff check .`, `ruff format --check .`, and `mypy src`.
4. Explain the serving assumption or paper result behind new optimization code.

Public APIs should use typed dataclasses where practical. Runtime dependencies
need a clear justification; optional solver or provider integrations should be
extras rather than mandatory dependencies.

Use conventional, focused commits. Documentation must state modeling
assumptions and units. Do not add provider credentials, proprietary benchmark
data, or network-dependent tests.

