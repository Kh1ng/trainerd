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
  "https://github.com/Kh1ng/trainerd/releases/download/v0.2.1/trainerd-0.2.1-py3-none-any.whl"
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

It listens on `0.0.0.0:7860`. On Windows, managed checkouts and job state
default to `%PROGRAMDATA%\trainerd\state`; `--state-dir` can override this.

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
  -ContentType application/json `
  -Body $body
```

The daemon clones or fast-forwards its own managed checkout and loads
`.trainerd.yaml` from the repository root:

```yaml
version: 1
tasks:
  nfl-train:
    max_concurrent_jobs: 1
    steps:
      - id: train
        name: Train NFL models
        cmd: 'py -3.12 -u scripts/trainerd_nfl_task.py --work-dir "{work_dir}"'
        cwd: "."
        timeout_seconds: 14400
```

Only anonymous `http://` and `https://` Git URLs are accepted. SSH/file URLs,
URL credentials, client commands, and client filesystem paths are rejected.
Task manifests are bounded and working directories cannot escape the daemon's
managed checkout or work directory. The commands in `.trainerd.yaml` are still
executable code from the repository.

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
work_dir: "${PROJECT_A_WORK_PATH}"
log_dir: "${PROJECT_A_LOG_PATH}"
max_concurrent_jobs: 1
steps:
  - id: pull
    name: Update checkout
    cmd: "git pull --ff-only origin {branch}"
    timeout_seconds: 300
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
  --steps pull,run \
  --version v42 \
  --wait \
  --logs
```

The equivalent request is:

```bash
curl --fail-with-body \
  -H "X-API-Key: $TRAINING_SERVER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"project":"project-a","steps":["pull","run"],"version":"v42"}' \
  http://training-node:7860/api/jobs
```

Registry-mode requests must include `project`. They may select only configured
step IDs. `branch` and arbitrary `extra_args` are rejected in registry mode;
commands and paths remain entirely server-owned.

## API

| Endpoint | Authentication | Purpose |
|---|---:|---|
| `GET /api/health` | No | Version, allowlist, queue, and capacity |
| `POST /api/jobs` | API key; none in LAN mode | Submit a job |
| `GET /api/jobs` | API key; none in LAN mode | List recent jobs |
| `GET /api/jobs/{job_id}` | API key; none in LAN mode | Read job status |
| `GET /api/jobs/{job_id}/logs` | API key; none in LAN mode | Tail or stream logs |
| `DELETE /api/jobs/{job_id}` | API key; none in LAN mode | Cancel a queued/running job |
| `POST /api/jobs/{job_id}/promote` | API key; none in LAN mode | Run a configured promotion hook |
| `GET /api/models?project=...` | API key; none in LAN mode | Compatibility artifact listing |

Interactive OpenAPI documentation is available at `/docs`.

## Windows daemon

Install `trainerd` in a dedicated daemon virtual environment, separate from all
project virtual environments:

```powershell
py -3.12 -m venv C:\ProgramData\trainerd\venvs\0.2.1
C:\ProgramData\trainerd\venvs\0.2.1\Scripts\python.exe -m pip install --upgrade pip
C:\ProgramData\trainerd\venvs\0.2.1\Scripts\python.exe -m pip install `
  https://github.com/Kh1ng/trainerd/releases/download/v0.2.1/trainerd-0.2.1-py3-none-any.whl
C:\ProgramData\trainerd\venvs\0.2.1\Scripts\trainerd.exe --version
```

Run this command from a Windows service wrapper or Scheduled Task:

```powershell
C:\ProgramData\trainerd\venvs\0.2.1\Scripts\trainerd.exe serve `
  --projects-config C:\ProgramData\trainerd\projects.yaml `
  --host 0.0.0.0 `
  --port 7860
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

No license has been granted for redistribution or modification. The repository
is source-available pending an explicit license decision.
