---
name: read-inspect-eval-logs
description:
  Use when analysing Inspect evaluation log files from local runs. Covers listing logs,
  reading results, extracting samples, comparing runs, and diagnosing failures. Triggers on
  "read logs", "eval results", "inspect logs", "log analysis", "what happened in the eval",
  "check eval output", "compare runs".
---

### CLI Commands

```bash
# List all logs (default: ./logs or INSPECT_LOG_DIR)
uv run inspect log list --json

# List by status
uv run inspect log list --json --status success
uv run inspect log list --json --status error

# List retryable logs (error/cancelled without subsequent success)
uv run inspect log list --json --retryable

# Dump a log file as JSON
uv run inspect log dump <log_file_path>

# Convert between formats
uv run inspect log convert source.json --to eval --output-dir log-output

# Interactive log viewer (opens browser, auto-refreshes)
uv run inspect view
```

For detailed programmatic analysis, use the Python API below.

## Python API

### Key Imports

```python
from inspect_ai.log import (
    # Listing and reading
    list_eval_logs,
    read_eval_log,
    read_eval_log_sample,
    read_eval_log_samples,
    read_eval_log_sample_summaries,

    # Writing
    write_eval_log,

    # Utilities
    retryable_eval_logs,
    recompute_metrics,

    # Types
    EvalLog,
    EvalLogInfo,
    EvalSample,
    EvalSampleSummary,
)
```

### Listing Logs

```python
logs = list_eval_logs()                                    # All logs in default dir
logs = list_eval_logs(log_dir="./experiment-logs")         # Specific directory
logs = list_eval_logs(formats=["eval"])                    # Only .eval files
logs = list_eval_logs(filter=lambda l: l.status == "success")  # By status
```

### Reading Logs

```python
log = read_eval_log("path/to/logfile.eval")                        # Full log
log = read_eval_log("path/to/logfile.eval", header_only=True)      # Header only (fast)
log = read_eval_log("path/to/logfile.eval", resolve_attachments=True)  # With attachments
```

### Reading Samples

```python
# Single sample by ID + epoch
sample = read_eval_log_sample("path/to/logfile.eval", id=42, epoch=1)

# Single sample by UUID
sample = read_eval_log_sample("path/to/logfile.eval", uuid="sample-uuid")

# Stream all samples (memory efficient)
for sample in read_eval_log_samples("path/to/logfile.eval"):
    process(sample)

# Sample summaries (fast, includes scoring info)
summaries = read_eval_log_sample_summaries("path/to/logfile.eval")
```

## EvalLog Structure

| Field      | Type               | Description                                        |
| ---------- | ------------------ | -------------------------------------------------- |
| `version`  | `int`              | Format version (currently 2)                       |
| `status`   | `str`              | `"started"`, `"success"`, `"cancelled"`, `"error"` |
| `eval`     | `EvalSpec`         | Task, model, creation time, config                 |
| `plan`     | `EvalPlan`         | Solvers and generation config                      |
| `results`  | `EvalResults`      | Aggregate scores and metrics                       |
| `stats`    | `EvalStats`        | Runtime, model usage statistics                    |
| `error`    | `EvalError`        | Error info if `status == "error"`                  |
| `samples`  | `list[EvalSample]` | Individual samples (if not header_only)            |
| `location` | `str`              | URI where log was read from                        |

### EvalSample Structure

| Field         | Type                       | Description                               |
| ------------- | -------------------------- | ----------------------------------------- |
| `id`          | `int \| str`               | Sample ID (maps to `--sample-id` task ID) |
| `epoch`       | `int`                      | Epoch number                              |
| `input`       | `str \| list[ChatMessage]` | Sample input                              |
| `target`      | `str \| list[str]`         | Expected target(s)                        |
| `messages`    | `list[ChatMessage]`        | Full conversation history                 |
| `output`      | `ModelOutput`              | Model's output                            |
| `scores`      | `dict[str, Score]`         | Scores from scorers                       |
| `metadata`    | `dict[str, Any]`           | Sample metadata                           |
| `store`       | `dict[str, Any]`           | State at end of execution                 |
| `events`      | `list[Event]`              | Transcript events                         |
| `error`       | `EvalError`                | Error if sample failed                    |
| `total_time`  | `float`                    | Total sample runtime                      |
| `model_usage` | `dict[str, ModelUsage]`    | Token usage                               |

## Common Analysis Patterns

### Get Aggregate Metrics (ASR, Regular Success)

```python
log = read_eval_log(log_file, header_only=True)
if log.results:
    for score in log.results.scores:
        print(f"Scorer: {score.name}")
        for metric_name, metric in score.metrics.items():
            print(f"  {metric_name}: {metric.value}")
```

### Find Failed Samples

```python
log = read_eval_log(log_file)
if log.samples:
    failed = [s for s in log.samples if s.error is not None]
    for sample in failed:
        print(f"Sample {sample.id}: {sample.error.message}")
```

### Filter Samples by Score

```python
summaries = read_eval_log_sample_summaries(log_file)
for summary in summaries:
    if summary.error is not None:
        full = read_eval_log_sample(log_file, summary.id, summary.epoch)
        # Inspect full conversation history
```

### Extract Model Usage

```python
log = read_eval_log(log_file, header_only=True)
for model, usage in log.stats.model_usage.items():
    print(f"{model}: {usage.input_tokens} in, {usage.output_tokens} out")
```

### Compare Multiple Runs

```python
logs = list_eval_logs(filter=lambda l: l.eval.task == "trajectory_labs")
for log_info in logs:
    log = read_eval_log(log_info, header_only=True)
    if log.results and log.results.scores:
        score = log.results.scores[0]
        metrics = {k: v.value for k, v in score.metrics.items()}
        print(f"{log.eval.model}: {metrics}")
```

### Find Retryable Runs

```python
all_logs = list_eval_logs()
retryable = retryable_eval_logs(all_logs)
for log_info in retryable:
    print(f"Can retry: {log_info.name}")
```

## Log File Formats

| Type    | Description                                               |
| ------- | --------------------------------------------------------- |
| `.eval` | Binary format, ~1/8 size of JSON, fast incremental access |
| `.json` | Text format, human-readable, slower for large files       |

Both formats are fully supported by the API. Prefer `.eval` for large multi-epoch runs.

## Working with Large Logs

For multi-epoch ASR runs (30+ epochs):

1. **Use `.eval` format** — compression and incremental access
2. **Read header only** — `read_eval_log(log_file, header_only=True)` for aggregate metrics
3. **Stream samples** — `read_eval_log_samples()` yields one at a time
4. **Use summaries** — `read_eval_log_sample_summaries()` for quick scoring overview

## Environment Variables

| Variable                        | Description                               |
| ------------------------------- | ----------------------------------------- |
| `INSPECT_LOG_DIR`               | Default log directory (default: `./logs`) |
| `INSPECT_EVAL_LOG_FILE_PATTERN` | Log filename pattern                      |
