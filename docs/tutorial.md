# Routing and evaluation tutorial

Start from the included offline dataset:

```python
from llm_route_opt.benchmark import load_routerbench_jsonl

dataset = load_routerbench_jsonl("examples/synthetic_routerbench.jsonl")
```

Evaluate a fixed baseline and a quality-gated cascade:

```python
from llm_route_opt import CascadeRouter, SingleModelRouter, evaluate

small = SingleModelRouter(dataset.models["small"])
cascade = CascadeRouter(tuple(dataset.models.values()), quality_threshold=0.8)
print(evaluate(dataset, small).to_dict())
print(evaluate(dataset, cascade).to_dict())
```

`TopKRouter` exposes an ordered candidate set while deterministically executing
the highest-ranked candidate. `DAGRouter` follows the first matching outgoing
edge from each node and stops when no predicate matches. Edges are evaluated in
insertion order, and cyclic graphs are rejected at construction.

For a learned router, provide a quality estimator:

```python
def estimator(query, model):
    task_bonus = 0.02 if model.metadata.get("specialty") == query.task else 0.0
    return min(1.0, model.quality + task_bonus)
```

An empirical estimator can instead index the dataset's measurement table. Keep
train/evaluation queries disjoint when fitting it to avoid leakage.

