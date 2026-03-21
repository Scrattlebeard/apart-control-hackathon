# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GOAP (Goal-Oriented Action Planning) micro-protocol for AI control research. A deterministic A* planner selects which tools to call and with what structural parameters; an untrusted LLM only generates content within bounded action steps. Evaluated on AgentDojo's workspace suite via ControlArena.

**Threat model:** AI control — defending against a presumed malicious model (attack policy), not prompt injection. ControlArena plugs the eval model into an attack policy in attack mode. The planner is trusted; the eval model is untrusted.

## Commands

**Default model for testing: `openrouter/google/gemini-3-flash-preview`** — use this unless explicitly told otherwise. Do NOT use expensive models (Sonnet, Opus, GPT-4) for test runs.

```bash
# Run evaluation (honest mode, specific task)
uv run python -m goap.run_eval --mode honest --main-task user_task_13 --model openrouter/google/gemini-3-flash-preview --limit 1 -v

# Run all tasks
uv run python -m goap.run_eval --mode honest --model openrouter/google/gemini-3-flash-preview --limit 1 -v

# Run attack mode
uv run python -m goap.run_eval --mode attack --main-task user_task_13 --model openrouter/google/gemini-3-flash-preview --limit 1 -v

# Use different trusted vs untrusted models
uv run python -m goap.run_eval --mode honest --model openrouter/google/gemini-3-flash-preview --trusted-model openrouter/google/gemini-3-flash-preview --limit 1 -v

# Install/sync dependencies
uv sync
```

API keys via `.env` file (Inspect AI loads `.env` automatically):
```
OPENROUTER_API_KEY=sk-or-...
```

## Architecture

### Core flow

The protocol runs inside ControlArena's react scaffold loop. Each call to the agent advances the plan by one action:

1. **Planning phase** — Trusted model (`get_model(role="trusted")`) generates action schemas from task description + tool signatures. A* planner orders them by dependencies.
2. **Re-planning** — If the initial plan was retrieval-only (no `llm_args`), the protocol re-plans with tool results as context, allowing the planner to decompose the full task.
3. **Execution phase** — Each call constructs a `ToolCall` for the next action. Actions with `llm_args` use `get_model()` (the eval/untrusted model) to generate content. The react scaffold executes the tool call and returns results.
4. **Submit phase** — Trusted model formulates the answer from task + tool results, then submits.

### Key invariant

The planner (not the LLM) controls: which tools are called, in what order, and with what `fixed_args` (recipients, file IDs, subjects). The LLM only fills declared `llm_args` (email body, activity content). Even a malicious model cannot expand the action space or override structural parameters.

### Source layout (`src/goap/`)

- `types.py` — `GoapAction`, `WorldState`, `GoalState`, `Plan` dataclasses
- `planner.py` — A* search (heuristic: unsatisfied goal count)
- `trusted_planner.py` — Trusted LLM planner: extracts tool signatures, calls trusted model to generate `PlannedAction`s with semantic preconditions/effects + world state + goal; A* derives ordering
- `actions.py` — Hand-crafted action schemas for UserTask13 (kept as reference / few-shot example source)
- `protocol.py` — `@control_agent` implementation, planning/re-planning/execution/submit state machine
- `run_eval.py` — CLI entry point wrapping ControlArena's `get_control_task()`

### Vendored dependencies

- `control-arena/` — ControlArena evaluation framework (Inspect AI + AgentDojo integration). Defines `@control_agent`, `control_solver`, `AgentDojoSetting`, scorers.
- `camel/` — CaMeL framework for prompt injection defense via taint tracking. Not yet integrated with GOAP.

Both are gitignored and installed as path dependencies via `uv`.

### Integration points

- `goap_protocol()` implements ControlArena's `ControlAgent` interface via `@control_agent` decorator
- Tool calls use `inspect_ai.tool.ToolCall(id, function, arguments)` — not dicts
- After appending a `ChatMessageAssistant` with tool calls, set `state._output = None` so the react scaffold synthesizes output from the new message
- `get_model()` returns the eval/untrusted model — in attack mode this is wrapped by the attack policy
- `get_model(role="trusted")` returns the trusted model for planning and answer formulation
- `control_task_metadata().main_task_description` provides the task text at runtime
- `parse_tool_info(tool)` from `inspect_ai.tool._tool_info` extracts parameter schemas; `registry_info(tool).name` gives the tool name

## Working style

- **Command/query separation.** If Daniel asks a question or raises a concern, answer it. Don't start changing code until explicitly asked. A question is not an instruction.
