# AGENTS.md

## Repository contract

- Support Python 3.11+ and keep production code under `src/llm_route_opt`.
- Preserve provider neutrality and a dependency-free runtime core.
- Run `pytest`, `ruff check .`, `ruff format --check .`, and `mypy src` after
  changes.
- Keep examples deterministic, offline, and free of credentials or proprietary
  data.
- Document units and mathematical assumptions for new metrics or constraints.
- Do not commit generated environments, caches, build products, or user traces.

Routers implement `route(QueryFeatures) -> RouteDecision`. New optimization
backends should return existing result schemas or introduce a documented typed
schema with tests. Avoid silently changing tie-breaking rules: reproducibility
depends on them.

