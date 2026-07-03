cd tr# trainerd

`trainerd` is a domain-neutral job runner.

It accepts caller-provided job payloads, resolves them through a config file,
queues work, runs subprocess steps, stores status, streams logs, and optionally
runs validation and promotion hooks.

It does not know sports, betting, research metrics, or model semantics.

## Safety Note

`trainerd` runs caller-provided commands from config and payload substitutions.
Treat it like a remote execution service for trusted repos and trusted configs.
Do not expose it to untrusted users or unreviewed command templates.

## Quickstart

Install:

```bash
pip install -e .[dev]
```

Run the server:

```bash
TRAINING_CONFIG=./examples/trainerd/sleep_job_config.yaml trainerd serve
```

Submit a job:

```bash
trainerd submit --server-url http://127.0.0.1:7860 --version demo-v1
```

Watch a job:

```bash
trainerd watch --server-url http://127.0.0.1:7860 --job-id <job-id> --logs
```

## API Overview

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | health and queue status |
| `POST /api/jobs` | submit a job |
| `GET /api/jobs` | list recent jobs |
| `GET /api/jobs/{job_id}` | inspect one job |
| `GET /api/jobs/{job_id}/logs` | stream or tail logs |
| `DELETE /api/jobs/{job_id}` | cancel pending/running job |
| `POST /api/jobs/{job_id}/promote` | manually trigger promotion |
| `GET /api/models` | list promoted model directories from the configured repo |

## Job Payload Example

```json
{
  "version": "v42",
  "steps": ["pull", "train"],
  "branch": "main",
  "markets": "task_a,task_b",
  "extra_args": "--flag value",
  "force": false,
  "triggered_by": "cli"
}
```

All fields are optional. `version`, `markets`, and `extra_args` are opaque
strings to `trainerd`.

## Artifact Manifest Example

```json
{
  "run_label": "v42",
  "job_id": "abcd1234",
  "produced_at": "2026-07-03T06:00:00Z",
  "artifacts": [
    {
      "path": "artifacts/output.txt",
      "sha256": "abc123",
      "bytes": 42
    }
  ]
}
```

## Local Dev

Run tests:

```bash
python -m pytest -q tests/test_trainerd_contracts.py tests/test_training_server.py tests/test_trainerd_runner.py tests/test_training_pipeline_smoke.py
```

Compile-check:

```bash
python -m py_compile trainerd/*.py training_server/*.py
```

## Package Contents

- `trainerd/`: package code
- `training_server/`: temporary compatibility shim for legacy import paths
- `examples/trainerd/`: minimal configs and scripts
- `.github/workflows/trainerd.yml`: draft CI for the extracted repo

