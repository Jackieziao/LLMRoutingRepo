---
name: select-llm-model
description: Select one LLM model for an input task using task complexity, coding depth, risk, latency, and cost preferences. Use when a user asks which model should handle a task, wants a model field for a router or workflow, compares Sol/Terra/Luna, or needs deterministic task-to-model routing output.
---

# Select LLM Model

Choose one model from the caller's available catalog. Return a recommendation;
do not claim to switch the active runtime model.

## Workflow

1. Extract the task, priority (`quality`, `balanced`, `speed`, or `cost`),
   complexity, risk, and available model identifiers.
2. Infer missing fields from the task. Ask only when a missing constraint could
   materially reverse the choice.
3. Run `scripts/select_model.py` for deterministic classification when Python is
   available. Pass the exact available catalog with `--available`.
4. Verify that the selected identifier is present in the available catalog.
   Never invent availability, pricing, context limits, or capabilities.
5. Return the selected model, a one-sentence reason, and one available fallback.

## Default routing policy

- Select `gpt-5.6-sol` for difficult architecture, deep debugging, security,
  complex algorithms, high-risk review, research synthesis, or maximum-quality
  work.
- Select `gpt-5.6-terra` for normal feature implementation, refactoring, tests,
  documentation, code review, and balanced daily engineering. Use it as the
  default when signals are mixed.
- Select `luna` for low-risk extraction, classification, formatting, short
  summaries, boilerplate, and other throughput- or cost-sensitive tasks.

Treat these as routing labels. `gpt-5.6-sol` and `gpt-5.6-terra` are the current
Codex identifiers in environments that expose them. Treat `luna` as an optional
catalog entry or user-defined alias; choose it only when the caller explicitly
lists it as available. If Luna is unavailable, fall back to Terra for simple or
balanced tasks.

Explicit user priorities override inferred complexity unless doing so would
violate a stated safety or availability constraint.

## Command

```bash
python scripts/select_model.py \
  --task "Implement a typed REST endpoint and tests" \
  --priority balanced \
  --available gpt-5.6-sol gpt-5.6-terra luna
```

Use `--format text` for a concise human-readable answer or `--format json` for a
router/config payload.

## Output contract

For normal answers, use exactly:

```text
Model: <available model identifier>
Why: <one sentence tied to the input task and priority>
Fallback: <available identifier or none>
```

When JSON is requested, return the script schema:

```json
{"model":"gpt-5.6-terra","reason":"...","fallback":"gpt-5.6-sol"}
```

Do not list every model unless the user asks for a comparison. Do not present
the optional Luna label as an official or installed OpenAI model without
current official documentation or an explicit caller-provided catalog.

