# llm-route-opt

`llm-route-opt` is a provider-neutral Python toolkit for reproducible LLM
routing, offline evaluation, discrete optimization, deployment assignment, and
queue-aware capacity planning. It uses only synthetic public data and requires
no API keys or paid services.

## MVP capabilities

- Typed schemas for query features, model profiles, empirical route outcomes,
  and route decisions.
- A normalized, long-form RouterBench-compatible JSONL loader and writer.
- Single-model, top-k, deterministic quality-gated cascade, and acyclic DAG
  routers.
- Exact quality maximization under total budget and per-query latency limits.
- Exact workload-to-model deployment assignment with integer replicas and FCFS
  M/M/c waiting times.
- Grid-exact discrete inverse optimization for learning normalized objective
  weights from observed choices.
- Deterministic `evaluate`, `optimize`, and end-to-end `demo` CLI workflows.

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Reproduce the MVP

```bash
llm-route-opt demo
llm-route-opt evaluate --data examples/synthetic_routerbench.jsonl --router cascade --threshold 0.8
llm-route-opt optimize routing --budget 0.03 --max-latency 600
llm-route-opt optimize deployment --budget 8
llm-route-opt optimize inverse
```

The demo always prints measured mean quality, total USD cost, mean/p95 latency,
model counts, and the constrained optimal assignment. Floating-point results
are deterministic for the included dataset.

## Python example

```python
from llm_route_opt import CascadeRouter, evaluate
from llm_route_opt.data import synthetic_dataset

dataset = synthetic_dataset()
router = CascadeRouter(tuple(dataset.models.values()), quality_threshold=0.80)
result = evaluate(dataset, router)
print(result.to_dict())
```

Learned estimators can replace the built-in deterministic quality estimator by
passing any typed callable `(QueryFeatures, ModelProfile) -> float`. Provider
adapters belong outside the core package, keeping experiments portable.

## Data format

The JSONL format contains `query`, `model`, and `measurement` records. A
measurement is one empirical `(query_id, model_id)` quality/cost/latency result,
which is the core information used by RouterBench-style offline evaluation.
See [the synthetic dataset](examples/synthetic_routerbench.jsonl) for a complete
public example.

## Documentation

- [Mathematical formulation](docs/math.md)
- [Routing and evaluation tutorial](docs/tutorial.md)
- [End-to-end experiment](docs/end_to_end.md)
- [Small Jupyter case demo](examples/quickstart.ipynb)

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy src
```

The package has a `src` layout and no runtime dependencies. Exact enumeration
is intentional: it creates auditable baselines for small research instances.
The roadmap is to add optional MILP/CP-SAT backends, trace converters, empirical
latency distributions, memory/accelerator constraints, and learned-router
calibration without changing the core schemas.

## License and citation

Apache-2.0. Cite the project using [`CITATION.cff`](CITATION.cff).
