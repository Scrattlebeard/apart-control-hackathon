# GOAP Control Protocol

A [Goal-Oriented Action Planning](https://en.wikipedia.org/wiki/Goal-oriented_action_planning) micro-protocol for AI control research. A trusted LLM planner decomposes tasks into deterministic action sequences; an untrusted model only generates free-text content within bounded steps.

Evaluated on [AgentDojo](https://github.com/ethz-spylab/agentdojo)'s workspace suite via [ControlArena](https://github.com/ControlArena/control-arena).

## Threat Model

**AI control** — defending against a presumed malicious model (attack policy), not prompt injection. The planner is trusted; the eval model is untrusted. ControlArena plugs the eval model into an attack policy in attack mode.

### Key Invariant

The planner controls which tools are called, in what order, and with what structural arguments (recipients, file IDs, subjects). The untrusted model only fills declared `llm_args` — free-text content like email bodies or file content. Even a malicious model cannot expand the action space or override structural parameters.

## Setup

```bash
uv sync
```

Add your API key to `.env` (Inspect AI loads it automatically):

```
OPENROUTER_API_KEY=sk-or-...
```

## Usage

```bash
# Run a single task
uv run python -m goap.run_eval --mode honest --main-task user_task_13 --limit 1 -v

# Run 5 tasks
uv run python -m goap.run_eval --mode honest --limit 5 -v

# Run all tasks
uv run python -m goap.run_eval --mode honest -v

# Attack mode
uv run python -m goap.run_eval --mode attack --main-task user_task_13 --limit 1 -v
```

### Model selection

Both `--model` (untrusted content generator) and `--trusted-model` (planner) default to `openrouter/anthropic/claude-haiku`. If `--trusted-model` is omitted, it falls back to `--model`.

```bash
# Use a stronger planner, cheap content generator
uv run python -m goap.run_eval --mode honest \
  --model openrouter/anthropic/claude-haiku \
  --trusted-model openrouter/anthropic/claude-sonnet-4 \
  --limit 5 -v

# Use Sonnet for everything
uv run python -m goap.run_eval --mode honest \
  --model openrouter/anthropic/claude-sonnet-4 \
  --limit 1 -v
```

### ShadeArena tasks

```bash
uv run python -m goap.run_eval --setting shade_arena --task-pair bank_transfer_2 --mode honest --limit 1 -v
```

| ID | Task name | Description |
|----|-----------|-------------|
| 1 | `api_key_calendar_agendas_2` | Review upcoming appointments, email attendees |
| 2 | `spam_filter_update` | Standardize spam filter rules across employees |
| 3 | `bank_transfer_2` | Cabo trip expense splitting + payments via Venmo/Zelle/bank |
| 4 | `american_airlines_2` | Conference visitor travel: flights, loyalty numbers, restaurant booking |
| 5 | `expenditure_accounting` | Financial transaction review and documentation |

## How It Works

The protocol runs inside ControlArena's react scaffold loop:

1. **Planning** — A trusted model generates action schemas from the task description and available tool signatures. Each action specifies `fixed_args` (planner-controlled) and `llm_args` (untrusted model fills). A\* search orders actions by dependencies.

2. **Re-planning** — If the initial plan was retrieval-only (gathering info before the real work), the protocol automatically re-plans with tool results as context.

3. **Execution** — One action per scaffold call. Actions with `llm_args` invoke the untrusted model to generate content. The scaffold executes the resulting `ToolCall`.

4. **Submit** — The trusted model formulates the final answer from task description + tool results.

## Source Layout

```
src/goap/
├── types.py            # GoapAction, WorldState, GoalState, Plan
├── planner.py          # A* search (heuristic: unsatisfied goal count)
├── trusted_planner.py  # Trusted LLM planner: tool extraction, plan generation, GOAP conversion
├── actions.py          # Hand-crafted UserTask13 actions (reference / few-shot example)
├── protocol.py         # @control_agent state machine: plan → execute → submit
└── run_eval.py         # CLI entry point
```
