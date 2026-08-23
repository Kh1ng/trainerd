# trainerd

`trainerd` is a standalone HTTP daemon for trusted, queued subprocess jobs. It
can run in zero-configuration LAN mode or load an immutable project allowlist,
persists jobs and logs per project, and enforces daemon-wide and per-project
concurrency limits.

Clients submit a project ID and bounded job parameters. They cannot submit
filesystem paths or command templates. No SSH access is needed for normal job
submission, status, logs, cancellation, or promotion.

## Install

Python 3.10 or newer is required.

```bash
python -m pip install \
  "https://github.com/Kh1ng/trainerd/releases/download/v0.3.0/trainerd-0.3.0-py3-none-any.whl"
trainerd --version
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Zero-configuration LAN mode

LAN mode is for a trusted private network where convenience matters more than
authentication. Start the installed package with no registry, API key, project
path, or SSH configuration:

```powershell
trainerd serve --lan
```

For a persistent network listener, constrain LAN mode to reviewed repositories
and require the existing API-key header. Keep the secret in the environment,
not the command line:

```powershell
$env:TRAINERD_API_KEY = "replace-with-a-long-random-secret"
trainerd serve --lan `
  --allow-repo http://git.local/Khing/sportsball-bets.git
```

`--allow-repo` is repeatable and fails startup unless `TRAINERD_API_KEY` is
set. Submitting, cancelling, and promoting jobs require `X-API-Key`. Health,
job status, logs, and model metadata remain readable on the trusted LAN, and
persisted job status and logs remain readable after daemon restarts.

It listens on `0.0.0.0:7860`. On Windows, managed checkouts and job state
default to `%PROGRAMDATA%\trainerd\state`; `--state-dir` can override this.

One stable Windows account should own the state directory. If the daemon moves
between an elevated shell, Task Scheduler, and a normal user session, Git
reflogs in a managed checkout can end up owned by the earlier account. The next
submit then fails during `git fetch` with `Permission denied`. trainerd probes
the checkout's Git metadata before syncing and fails with an actionable message
that names the checkout and the current service identity instead of failing
mid-job. Grant the trainerd service account recursive control of the affected
checkout, then resubmit.

Submit one anonymous HTTP request containing only the Git HTTP URL and the
repository-owned task name:

```powershell
$body = @{
  repo = "http://192.168.5.150/Khing/sportsball-bets.git"
  task = "nfl-train"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:7860/api/jobs `
  -Headers @{ "X-API-Key" = $env:TRAINERD_API_KEY } `
  -ContentType application/json `
  -Body $body
```

The daemon clones or fast-forwards its own managed checkout and loads
`.trainerd.yaml` from the repository root:

```powershell
trainerd submit `
  --server-url http://127.0.0.1:7860 `
  --repo http://git.local/Khing/sportsball-bets.git `
  --task nfl-train `
  --branch feature/training
```

The branch is optional. If omitted, trainerd uses the managed checkout's
current branch or the repository's default branch on the first clone. Branch
names must pass Git's native branch-name validation.

```yaml
version: 1
tasks:
  nfl-train:
    required_env:
      - NFL_DATABASE_URL
    # Set this to at least 2 to overlap stages from independent jobs.
    max_concurrent_jobs: 2
    steps:
      - id: prepare
        name: Prepare training data
        queue: cpu
        cmd: 'py -3.12 -u scripts/prepare.py --work-dir "{work_dir}"'
        cwd: "."
      - id: train
        name: Train NFL models
        queue: gpu
        # This stage claims both units from a daemon started with --gpu-capacity 2.
        units: 2
        cmd: 'py -3.12 -u scripts/trainerd_nfl_task.py --work-dir "{work_dir}"'
        cwd: "."
        timeout_seconds: 14400
```

Persistent job environment is configured once on the worker. Values are stored
under trainerd's managed state, are never printed by `env list`, and are
injected only into tasks that opt in with `required_env`:

```powershell
$env:NFL_DATABASE_URL = "postgresql://user:password@database/nfl"
trainerd env set NFL_DATABASE_URL --from-env NFL_DATABASE_URL
Remove-Item Env:NFL_DATABASE_URL
trainerd env list
```

Changing a stored value does not require restarting the daemon; it is loaded
when the next LAN task is prepared.

Only anonymous `http://` and `https://` Git URLs are accepted. SSH/file URLs,
URL credentials, client commands, and client filesystem paths are rejected.
Task manifests are bounded and working directories cannot escape the daemon's
managed checkout or work directory. The commands in `.trainerd.yaml` are still
executable code from the repository.

LAN tasks default to one queued, running, or validating job per repository.
Set a task's `max_concurrent_jobs` above one to allow jobs from that repository
to overlap. The daemon-wide limit still caps total active jobs.

When every task step sets `queue: cpu` or `queue: gpu`, trainerd persists each
stage handoff and admits it through a daemon-wide queue. Each stage claims one
`unit` by default. Set CPU capacity with `--cpu-concurrency` and GPU capacity
with `--gpu-capacity`; both default to one. A stage can set `units` up to its
queue's configured capacity. Job status and `/api/health` report total,
claimed, and available capacity. Stages in one job share
`TRAINERD_ARTIFACT_DIR`.

GPU units are operator-defined slots, not measured VRAM. Leave VRAM headroom,
measure each workload, and raise `--gpu-capacity` gradually. A request that
fits the configured units can still exhaust device memory if the estimates are
wrong. Set `--max-concurrent-jobs 1` to disable job overlap.

At submission, trainerd verifies that the managed checkout has no tracked
changes, fast-forwards it, and saves the exact revision. Each job runs in a
detached Git worktree at that revision with its own work directory. Trainerd
exports the revision as `TRAINERD_REPO_SHA`. It removes the worktree when the
job ends but retains the work directory and declared artifacts. The CPU queue
serializes preparation when shared environment setup must not overlap.

**LAN mode has no authentication. Anyone who can reach the port can run tasks
from an HTTP Git repository. Keep it behind the host firewall on a trusted LAN.
Use registry mode for any less-trusted network.**

## Run one daemon for multiple projects

Create a server-owned registry:

```yaml
# projects.yaml
default_project: project-a
api_key: "${TRAINING_SERVER_API_KEY}"
max_concurrent_jobs: 2
server:
  port: 7860
projects:
  project-a:
    config: "./project_a.yaml"
  project-b:
    config: "./project_b.yaml"
```

Each allowlisted project has its own command configuration:

```yaml
# project_a.yaml
project: project-a
repo:
  local_path: "${PROJECT_A_REPO_PATH}"
  # Optional outside LAN mode. Safely fast-forward immediately before each job.
  sync_before_job: true
work_dir: "${PROJECT_A_WORK_PATH}"
log_dir: "${PROJECT_A_LOG_PATH}"
max_concurrent_jobs: 1
steps:
  - id: run
    name: Run workload
    cmd: ".venv/Scripts/python.exe scripts/run_job.py --version {version}"
    timeout_seconds: 14400
```

Start the daemon:

```bash
export TRAINING_SERVER_API_KEY='replace-with-a-long-random-secret'
export PROJECT_A_REPO_PATH='/srv/project-a'
export PROJECT_A_WORK_PATH='/var/lib/trainerd/project-a'
export PROJECT_A_LOG_PATH='/var/log/trainerd/project-a'

trainerd serve \
  --projects-config ./projects.yaml \
  --host 0.0.0.0 \
  --port 7860
```

Registry mode fails closed if an environment variable, API key, config path, or
project identity is invalid. Each project must use a distinct `log_dir`, which
owns that project's SQLite database and job logs.

## Submit work over HTTP

The CLI reads `TRAINERD_API_KEY`, so the secret does not need to appear in the
command line:

```bash
export TRAINERD_API_KEY="$TRAINING_SERVER_API_KEY"

trainerd submit \
  --server-url http://training-node:7860 \
  --project project-a \
  --steps run \
  --version v42 \
  --wait \
  --logs
```

The equivalent request is:

```bash
curl --fail-with-body \
  -H "X-API-Key: $TRAINING_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"project":"project-a","steps":["run"],"version":"v42"}' \
  http://training-node:7860/api/jobs
```

Registry-mode requests must include `project`. They may select only configured
step IDs. `branch` and arbitrary `extra_args` are rejected in registry mode;
commands and paths remain entirely server-owned.

## Job artifacts

A repository-owned task can publish files from its managed work directory.
trainerd exports `TRAINERD_JOB_ID` and `TRAINERD_ARTIFACT_DIR` to each task step.

Write each file under `TRAINERD_ARTIFACT_DIR`. Then write
`artifact_manifest.json` in the same directory:

```json
{
  "run_label": "v42",
  "job_id": "abcd1234",
  "produced_at": "2026-08-21T12:00:00Z",
  "artifacts": [
    {
      "path": "result.json",
      "bytes": 72,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  ]
}
```

Artifact paths are relative to the job directory. trainerd rejects absolute
paths, directory escapes, size mismatches, and SHA-256 mismatches.

The manifest limit is 1 MiB and 256 entries. The total declared artifact size
must not exceed 2 GiB. Artifact endpoints are unavailable without an API key.

## API

| Endpoint | Authentication | Purpose |
|---|---:|---|
| `GET /api/health` | No | Version, allowlist, queue, and capacity |
| `POST /api/jobs` | API key; none in LAN mode | Submit a job |
| `GET /api/jobs` | API key; none in LAN mode | List recent jobs |
| `GET /api/jobs/{job_id}` | API key; none in LAN mode | Read job status |
| `GET /api/queue` | API key; none in LAN mode | Ordered active queue for monitors |
| `GET /api/jobs/{job_id}/logs` | API key; none in LAN mode | Tail or stream logs |
| `GET /api/jobs/{job_id}/artifacts` | API key required | List validated job artifacts |
| `GET /api/jobs/{job_id}/artifacts/{index}` | API key required | Download one validated artifact |
| `DELETE /api/jobs/{job_id}` | API key; none in LAN mode | Cancel a queued/running job |
| `POST /api/jobs/{job_id}/promote` | API key; none in LAN mode | Run a configured promotion hook |
| `GET /api/models?project=...` | API key; none in LAN mode | Compatibility artifact listing |

Interactive OpenAPI documentation is available at `/docs`.

## Windows daemon

Install `trainerd` in a dedicated daemon virtual environment, separate from all
project virtual environments:

```powershell
py -3.12 -m venv C:\ProgramData\trainerd\venvs\0.3.4
C:\ProgramData\trainerd\venvs\0.3.4\Scripts\python.exe -m pip install --upgrade pip
C:\ProgramData\trainerd\venvs\0.3.4\Scripts\python.exe -m pip install `
  https://github.com/Kh1ng/trainerd/releases/download/v0.3.4/trainerd-0.3.4-py3-none-any.whl
C:\ProgramData\trainerd\venvs\0.3.4\Scripts\trainerd.exe --version
```

Run this command from a Windows service wrapper or Scheduled Task:

```powershell
C:\ProgramData\trainerd\venvs\0.3.4\Scripts\trainerd.exe serve `
  --projects-config C:\ProgramData\trainerd\projects.yaml `
  --host 0.0.0.0 `
  --port 7860
```

Scheduled Tasks must run indefinitely and recover if the daemon exits. After
registering a task named `trainerd-lan`, apply these native Task Scheduler
settings. The repeating trigger is ignored while the daemon is already
running and starts it within one minute if it is not:

```powershell
$settings = New-ScheduledTaskSettingsSet `
  -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1) `
  -StartWhenAvailable `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -MultipleInstances IgnoreNew
$boot = New-ScheduledTaskTrigger -AtStartup
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$watchdog = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
  -RepetitionInterval (New-TimeSpan -Minutes 1)
Set-ScheduledTask -TaskName trainerd-lan -Settings $settings `
  -Trigger $boot,$logon,$watchdog
```

Project step commands may invoke each project's own virtual environment. Only
the daemon itself belongs in the dedicated `trainerd` environment. A normal
upgrade installs a new versioned daemon environment, validates it on an
alternate port, then switches the service action; the old environment remains
available for rollback.

## Legacy single-project mode

For an existing trusted config, run:

```bash
trainerd serve --config ./training_config.yaml
```

This compatibility mode permits the older optional `branch` and `extra_args`
payload fields. Prefer registry mode for any network-accessible daemon.

## Security boundary

Project configs contain executable command templates. Treat them as code and
review them before deployment. Bind to loopback unless remote access is
required; when exposed on a network, use registry mode, a strong API key, host
firewall rules, and TLS at a reverse proxy or private overlay network.

`trainerd` does not accept config paths or raw commands over HTTP. It
constant-time compares API keys, authenticates logs, validates project and
step identifiers, and starts with CORS disabled.

## Build and test

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

CI runs the suite on Linux and Windows, builds both wheel and source
distribution, installs the wheel into a clean environment, and smoke-tests the
CLI and import path.

## License

`trainerd` is released under the [MIT License](LICENSE).
