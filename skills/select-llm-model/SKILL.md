---
name: select-llm-model
description: Route a task to GPT-5.6 Sol, Terra, or Luna using a required JSON config that maps each model to API credentials. Use when a user wants config-driven model selection and optional execution that saves tokens by matching model capability to task complexity.
---

# Select LLM Model

Select one configured model and, when requested, send the task to that model.
Never print, return, log, or commit an API key.

## Workflow

1. First obtain the caller's JSON config containing the `sol`, `terra`, and
   `luna` entries shown in `model-api-keys.example.json`. Do not route or execute
   a task until the config is available and valid.
2. Prefer `api_key_env` references. If the config contains direct `api_key`
   values, use them without displaying them and warn the caller to keep the file
   untracked. Never ask the caller to paste a secret into chat.
3. Extract the task priority (`quality`, `balanced`, `speed`, or `cost`) and
   complexity. Infer missing values; ask only if a missing constraint could
   materially reverse the route.
4. Run `scripts/select_model.py --config <path> --task <task>` to select a model.
   Add `--execute` only when the caller asked the skill to run the task through
   the API; this sends data and incurs API usage.
5. Return the selected model and result. Do not claim that the current Codex
   session changed models: execution is a separate Responses API call.

## Routing policy

- `gpt-5.6-sol`: difficult architecture, deep debugging, security, complex
  algorithms, high-risk review, research synthesis, or maximum-quality work.
- `gpt-5.6-terra`: normal implementation, refactoring, tests, documentation,
  code review, and balanced daily engineering. This is the default.
- `gpt-5.6-luna`: extraction, classification, formatting, short summaries,
  boilerplate, and other low-risk or cost-sensitive work.

Explicit priority overrides inferred complexity, except that high-complexity
tasks continue to use Sol. The fallback order is Sol to Terra, Terra to Sol, and
Luna to Terra.

## Config and command

Use three environment variables so secrets stay outside the config:

```json
{
  "models": {
    "sol": {"model": "gpt-5.6-sol", "api_key_env": "OPENAI_SOL_API_KEY"},
    "terra": {"model": "gpt-5.6-terra", "api_key_env": "OPENAI_TERRA_API_KEY"},
    "luna": {"model": "gpt-5.6-luna", "api_key_env": "OPENAI_LUNA_API_KEY"}
  }
}
```

```bash
python scripts/select_model.py \
  --config model-api-keys.json \
  --task "Implement a typed REST endpoint and tests" \
  --priority balanced \
  --execute \
  --format text
```

Without `--execute`, the helper selects a route offline and does not resolve or
use the keys. The JSON selection schema is:

```json
{"model":"gpt-5.6-terra","reason":"...","fallback":"gpt-5.6-sol"}
```
