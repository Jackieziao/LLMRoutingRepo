# End-to-end reproducible experiment

This experiment is completely local and deterministic.

1. Install the project with development tools.
2. Evaluate the cascade on the checked-in JSONL data.
3. Optimize routing under an aggregate request budget.
4. Plan model replicas for two FCFS workloads.
5. Infer a decision-maker's objective weights.

```bash
python -m pip install -e ".[dev]"
llm-route-opt evaluate --data examples/synthetic_routerbench.jsonl --router cascade --threshold 0.8
llm-route-opt optimize routing --data examples/synthetic_routerbench.jsonl --budget 0.03 --max-latency 600
llm-route-opt optimize deployment --budget 8
llm-route-opt optimize inverse
```

Or run the combined report:

```bash
llm-route-opt demo
```

The evaluation uses checked-in per-query/per-model measurements. The routing
optimizer uses the profiles' deterministic pre-inference estimator, making the
distinction between predicted optimization targets and measured offline
evaluation explicit. Change the JSONL records or inject a new estimator to test
other routing hypotheses.

