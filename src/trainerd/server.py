"""trainerd API server.

A minimal REST server for triggering model training on a GPU machine,
validating results, and promoting validated artifacts back to git.

Designed to run as a persistent service on a Windows or Linux GPU machine.
Any project can configure it via training_config.yaml in the repo root.

Usage:
    uvicorn trainerd.server:app --host 0.0.0.0 --port 7860
    # or: python -m trainerd serve
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from fastapi.security import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr

from . import __version__
from .config import ServerConfig, TrainingConfig, load_config, load_server_config
from .lan import (
    LanConfigError,
    LanPreparedProject,
    default_state_dir,
    normalize_repo_url,
    prepare_lan_project,
    repo_key,
)
from .runner import JobRunner
from .storage import JobStore, JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_store: JobStore | None = None
_runner: JobRunner | None = None
_config: TrainingConfig | None = None
_config_path: Path | None = None


@dataclass
class ProjectRuntime:
    project: str
    config_path: Path | None
    config: TrainingConfig
    store: JobStore
    runner: JobRunner
    lan_repo_key: str | None = None


_server_config: ServerConfig | None = None
_projects: dict[str, ProjectRuntime] = {}
_default_project: str | None = None
_lan_mode_active = False
_lan_state_dir: Path | None = None
_lan_max_concurrent_jobs = 1
_lan_prepare_lock: asyncio.Lock | None = None

# Queue worker state
_queue_worker_task: asyncio.Task | None = None
_running_tasks: dict[str, asyncio.Task] = {}
_queue_poll_interval: float = 5.0


class JobRequest(BaseModel):
    """Strict HTTP shape; project config paths/commands are never accepted."""

    model_config = ConfigDict(extra="forbid")

    project: StrictStr | None = None
    repo: StrictStr | None = None
    repo_url: StrictStr | None = None
    task: StrictStr | None = None
    version: StrictStr | None = None
    steps: list[StrictStr] | None = None
    branch: StrictStr | None = None
    markets: StrictStr | None = None
    extra_args: StrictStr | None = None
    force: StrictBool = False
    triggered_by: StrictStr = "api"


_SAFE_JOB_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def _api_key_auth(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> None:
    api_key = _server_config.api_key if _server_config else (_config.api_key if _config else "")
    if not api_key:
        return
    key = header_key or query_key
    if key is None or not hmac.compare_digest(key, api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _runtime_map() -> dict[str, ProjectRuntime]:
    if _projects:
        return _projects
    if _store is None or _runner is None or _config is None or _config_path is None:
        return {}
    return {
        _config.project: ProjectRuntime(
            project=_config.project,
            config_path=_config_path,
            config=_config,
            store=_store,
            runner=_runner,
        )
    }


def _default_runtime() -> ProjectRuntime:
    runtimes = _runtime_map()
    project = _default_project or (_config.project if _config else None)
    runtime = runtimes.get(project or "")
    if runtime is None:
        raise RuntimeError("trainerd has no configured default project")
    return runtime


def _is_registry_mode() -> bool:
    return bool(_server_config and _server_config.registry_mode)


def _is_lan_mode() -> bool:
    return _lan_mode_active


def _select_runtime(project: Any = None) -> ProjectRuntime:
    requested = str(project) if project is not None else ""
    if not requested:
        if _is_registry_mode():
            raise HTTPException(
                status_code=400,
                detail="project is required in trainerd registry mode",
            )
        return _default_runtime()
    runtime = _runtime_map().get(requested)
    if runtime is None:
        allowed = ", ".join(sorted(_runtime_map()))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown project {requested!r}. Allowed projects: {allowed}",
        )
    return runtime


def _validate_multi_project_request(payload: dict[str, Any]) -> None:
    if not _is_lan_mode() and any(field in payload for field in ("repo", "repo_url", "task")):
        raise HTTPException(
            status_code=400,
            detail="repo and task are accepted only when trainerd starts with --lan",
        )
    if not _is_registry_mode():
        return
    if "branch" in payload:
        raise HTTPException(status_code=400, detail="branch is not accepted in registry mode")
    if "extra_args" in payload:
        raise HTTPException(status_code=400, detail="extra_args is not accepted in registry mode")

    version = payload.get("version")
    if version is not None and not _SAFE_JOB_TOKEN.fullmatch(version):
        raise HTTPException(status_code=400, detail="version contains unsupported characters")

    steps = payload.get("steps")
    if steps is not None:
        if not steps or any(not _SAFE_JOB_TOKEN.fullmatch(step) for step in steps):
            raise HTTPException(status_code=400, detail="steps must be non-empty safe step ids")

    markets = payload.get("markets")
    if markets is not None:
        market_ids = markets.split(",")
        if (
            not market_ids
            or len(market_ids) > 64
            or any(not _SAFE_JOB_TOKEN.fullmatch(market) for market in market_ids)
        ):
            raise HTTPException(
                status_code=400,
                detail="markets must be a comma-separated list of safe ids",
            )


def _refresh_project_runtime(runtime: ProjectRuntime) -> TrainingConfig:
    if runtime.config_path is None:
        return runtime.config
    refreshed = load_config(runtime.config_path)
    if refreshed.project != runtime.project:
        raise RuntimeError(
            f"Configured project changed from {runtime.project!r} "
            f"to {refreshed.project!r}"
        )
    runtime.config = refreshed
    runtime.runner.update_config(refreshed, config_path=runtime.config_path)
    if runtime.project == (_default_project or runtime.project):
        global _config
        _config = refreshed
    return refreshed


def _find_job_runtime(job_id: str) -> tuple[ProjectRuntime, dict] | None:
    found: list[tuple[ProjectRuntime, dict]] = []
    for runtime in _runtime_map().values():
        job = runtime.store.get_job(job_id)
        if job:
            found.append((runtime, job))
    if len(found) > 1:
        projects = ", ".join(item[0].project for item in found)
        raise RuntimeError(f"Ambiguous job id {job_id!r} exists in projects: {projects}")
    return found[0] if found else None


def _validate_unique_job_ids(runtimes: dict[str, ProjectRuntime]) -> None:
    owners: dict[str, str] = {}
    for runtime in runtimes.values():
        for job_id in runtime.store.list_job_ids():
            prior = owners.setdefault(job_id, runtime.project)
            if prior != runtime.project:
                raise RuntimeError(
                    f"Duplicate historical job id {job_id!r} exists in "
                    f"projects {prior!r} and {runtime.project!r}"
                )


def _with_project(runtime: ProjectRuntime, job: dict) -> dict:
    return {**job, "project": runtime.project}


def _active_job_ids_by_project(
    runtimes: dict[str, ProjectRuntime] | None = None,
) -> dict[str, set[str]]:
    runtimes = runtimes or _runtime_map()
    active_by_project: dict[str, set[str]] = {}
    for runtime in runtimes.values():
        running_ids = set(runtime.store.list_job_ids(status=JobStatus.RUNNING))
        reserved_ids = {
            job_id
            for job_id in _running_tasks
            if (found := _find_job_runtime(job_id)) and found[0].project == runtime.project
        }
        active_by_project[runtime.project] = running_ids | reserved_ids
    return active_by_project


def _pending_candidates(max_jobs: int) -> list[tuple[ProjectRuntime, dict]]:
    """Return globally oldest claimable jobs with running-task reservations."""
    runtimes = _runtime_map()
    active_by_project = _active_job_ids_by_project(runtimes)
    available = max(max_jobs - len(set().union(*active_by_project.values())), 0)
    candidates: list[tuple[str, ProjectRuntime, dict]] = []
    for runtime in runtimes.values():
        project_available = max(
            runtime.config.max_concurrent_jobs - len(active_by_project[runtime.project]),
            0,
        )
        if project_available <= 0:
            continue
        pending = runtime.store.list_jobs(
            status=JobStatus.PENDING,
            limit=min(project_available, available),
            oldest_first=True,
        )
        candidates.extend((job.get("created_at") or "", runtime, job) for job in pending)
    return [
        (runtime, job)
        for _, runtime, job in sorted(candidates, key=lambda item: item[0])[:available]
    ]


def _recover_stale_jobs(store: JobStore) -> None:
    """On startup, mark any running jobs as failed (interrupted by shutdown)."""
    for job_id in store.list_job_ids(status=JobStatus.RUNNING):
        log.warning("Recovering stale running job %s — marking as failed (interrupted)", job_id)
        store.set_failed(job_id, "Interrupted — server restarted")


async def _queue_worker() -> None:
    """Poll all allowlisted queues without exceeding the daemon-wide cap."""
    max_jobs = (
        _lan_max_concurrent_jobs
        if _is_lan_mode()
        else _server_config.max_concurrent_jobs
        if _server_config
        else (_config.max_concurrent_jobs if _config else 1)
    )
    log.info("Queue worker started (max_concurrent_jobs=%s)", max_jobs)
    while True:
        try:
            runtimes = _runtime_map()
            if not runtimes:
                await asyncio.sleep(1)
                continue
            for runtime in runtimes.values():
                _refresh_project_runtime(runtime)

            for runtime, job in _pending_candidates(max_jobs):
                jid = job["job_id"]
                if jid in _running_tasks:
                    continue
                log.info("Queue worker claiming job %s for project %s", jid, runtime.project)
                task = asyncio.create_task(_run_job_wrapper(jid, runtime.project))
                _running_tasks[jid] = task
                task.add_done_callback(lambda t, jid=jid: _running_tasks.pop(jid, None))

            await asyncio.sleep(_queue_poll_interval)
        except asyncio.CancelledError:
            log.info("Queue worker cancelled")
            break
        except Exception:
            log.exception("Queue worker error")
            await asyncio.sleep(1)


async def _run_job_wrapper(job_id: str, project: str | None = None) -> None:
    """Wrap runner.run_job to handle exceptions."""
    found = _find_job_runtime(job_id)
    runtime = _select_runtime(project) if project else (found[0] if found else _default_runtime())
    try:
        await runtime.runner.run_job(job_id)
    except asyncio.CancelledError:
        log.info("Job %s was cancelled", job_id)
        job = runtime.store.get_job(job_id)
        if job and job["status"] in (JobStatus.PENDING, JobStatus.RUNNING):
            runtime.store.set_failed(job_id, "Cancelled via API")
    except Exception:
        log.exception("Unexpected error in job %s", job_id)
        runtime.store.set_failed(job_id, "Internal error — see server logs")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _store, _runner, _config, _config_path, _queue_worker_task
    global _server_config, _projects, _default_project
    global _lan_mode_active, _lan_state_dir, _lan_max_concurrent_jobs, _lan_prepare_lock
    _lan_mode_active = os.environ.get("TRAINERD_LAN_MODE") == "1"
    _projects = {}
    if _lan_mode_active:
        _server_config = None
        _default_project = None
        _store = None
        _runner = None
        _config = None
        _config_path = None
        configured_state = os.environ.get("TRAINERD_STATE_DIR", "").strip()
        _lan_state_dir = (
            Path(configured_state).expanduser().resolve()
            if configured_state
            else default_state_dir().expanduser().resolve()
        )
        _lan_state_dir.mkdir(parents=True, exist_ok=True)
        try:
            _lan_max_concurrent_jobs = int(
                os.environ.get("TRAINERD_MAX_CONCURRENT_JOBS", "1")
            )
        except ValueError as exc:
            raise ValueError("TRAINERD_MAX_CONCURRENT_JOBS must be an integer") from exc
        if not 1 <= _lan_max_concurrent_jobs <= 64:
            raise ValueError("TRAINERD_MAX_CONCURRENT_JOBS must be from 1 to 64")
        _lan_prepare_lock = asyncio.Lock()
    else:
        _server_config = load_server_config()
        _default_project = _server_config.default_project
        for configured in _server_config.projects.values():
            store = JobStore(configured.config.log_dir / "jobs.db")
            runner = JobRunner(store, configured.config, config_path=configured.config_path)
            runtime = ProjectRuntime(
                configured.project,
                configured.config_path,
                configured.config,
                store,
                runner,
            )
            _projects[configured.project] = runtime
        _validate_unique_job_ids(_projects)
        for runtime in _projects.values():
            _recover_stale_jobs(runtime.store)

        default_runtime = _projects[_default_project]
        _config_path = default_runtime.config_path
        _config = default_runtime.config
        _store = default_runtime.store
        _runner = default_runtime.runner

    # Start background queue worker
    _queue_worker_task = asyncio.create_task(_queue_worker())

    if _lan_mode_active:
        log.warning(
            "INSECURE LAN MODE enabled on managed state %s; authentication is disabled",
            _lan_state_dir,
        )
    else:
        log.info(
            "Training server ready. Projects: %s  default=%s  max_concurrent_jobs=%s",
            sorted(_projects),
            _default_project,
            _server_config.max_concurrent_jobs,
        )
    yield
    log.info("Training server shutting down.")
    if _queue_worker_task:
        _queue_worker_task.cancel()
        try:
            await _queue_worker_task
        except asyncio.CancelledError:
            pass
    # Cancel any running training tasks
    for jid, task in list(_running_tasks.items()):
        task.cancel()
    _running_tasks.clear()
    _projects = {}
    _server_config = None
    _default_project = None
    _lan_mode_active = False
    _lan_state_dir = None
    _lan_prepare_lock = None


app = FastAPI(title="trainerd", version=__version__, lifespan=_lifespan)


def _refresh_runtime_config() -> TrainingConfig:
    global _config, _runner
    runtime = _default_runtime()
    _config = _refresh_project_runtime(runtime)
    _runner = runtime.runner
    return _config


@app.get("/api/health")
async def health() -> dict:
    runtimes = _runtime_map()
    for runtime in runtimes.values():
        _refresh_project_runtime(runtime)
    pending = sum(len(runtime.store.list_job_ids(status=JobStatus.PENDING)) for runtime in runtimes.values())
    running = sum(len(runtime.store.list_job_ids(status=JobStatus.RUNNING)) for runtime in runtimes.values())
    active = len(set().union(*_active_job_ids_by_project(runtimes).values()))
    max_jobs = (
        _lan_max_concurrent_jobs
        if _is_lan_mode()
        else _server_config.max_concurrent_jobs
        if _server_config
        else (_config.max_concurrent_jobs if _config else 1)
    )
    default = _default_project or (_config.project if _config else None)
    return {
        "status": "ok",
        "version": __version__,
        "build_commit": os.environ.get("TRAINERD_BUILD_COMMIT"),
        "project": default,
        "projects": sorted(runtimes),
        "default_project": default,
        "mode": "lan" if _is_lan_mode() else "registry" if _is_registry_mode() else "single",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pending_jobs": pending,
        "running_jobs": running,
        "max_concurrent_jobs": max_jobs,
        "queue_capacity": max(max_jobs - active, 0),
    }


@app.post("/api/jobs", dependencies=[Depends(_api_key_auth)])
async def submit_job(body: JobRequest = JobRequest()) -> dict:
    """Submit a training job. The job is queued and executed when a slot is available.

    In registry mode, project is required and branch/extra_args are rejected.
    In legacy singleton mode, project may be omitted and branch/extra_args retain
    their existing behavior. Other body fields:
      - project: startup-allowlisted project id
      - steps: list of step IDs (default: all configured steps)
      - version: version string (default: auto-incremented)
      - markets: market filter string
      - force: if true, skip dedupe check (default: false)
    """
    payload = body.model_dump(exclude_none=True)
    _validate_multi_project_request(payload)
    if _is_lan_mode():
        runtime, payload = await _prepare_lan_runtime(payload)
    else:
        runtime = _select_runtime(payload.get("project"))
    config = _refresh_project_runtime(runtime)
    store = runtime.store
    requested_steps = payload.get("steps")
    configured_step_ids = [s.id for s in config.steps]
    if requested_steps:
        steps = [str(step).strip() for step in requested_steps if str(step).strip()]
        unknown_steps = sorted(set(steps) - set(configured_step_ids))
        if unknown_steps:
            raise HTTPException(status_code=400, detail=f"Unknown step ids: {', '.join(unknown_steps)}")
        if not steps:
            raise HTTPException(status_code=400, detail="No valid step ids requested")
    else:
        steps = configured_step_ids
    version = _normalize_version(payload.get("version")) or _next_version(config)
    branch = payload.get("branch")
    markets = payload.get("markets")
    extra_args = payload.get("extra_args")
    force = payload.get("force", False)

    # Dedupe guard: reject duplicate pending/running jobs for same parameters
    if not force:
        dup = store.find_pending_or_running(version=version, branch=branch, markets=markets, extra_args=extra_args, steps=steps)
        if dup is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate job {dup['job_id']} already {'pending' if dup['status'] == JobStatus.PENDING else 'running'} "
                       f"for version={version} branch={branch} markets={markets}. Set force=true to override.",
            )

    job_id = str(uuid.uuid4())[:8]
    while _find_job_runtime(job_id) is not None:
        job_id = str(uuid.uuid4())[:8]
    job = store.create_job(
        job_id,
        steps=steps,
        version=version,
        triggered_by=payload.get("triggered_by", "api"),
        branch=branch,
        markets=markets,
        extra_args=extra_args,
    )
    log.info(
        "Job %s queued: project=%s steps=%s version=%s force=%s",
        job_id,
        runtime.project,
        steps,
        version,
        force,
    )
    return {
        "job_id": job_id,
        "project": runtime.project,
        "status": job["status"],
        "version": version,
        "steps": steps,
        "queued": True,
    }


async def _prepare_lan_runtime(
    payload: dict[str, Any],
) -> tuple[ProjectRuntime, dict[str, Any]]:
    """Resolve a bounded LAN request to a daemon-owned runtime."""
    allowed = {"repo", "repo_url", "task", "version", "force", "triggered_by"}
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"LAN mode does not accept field(s): {', '.join(unsupported)}",
        )
    repo = payload.get("repo")
    repo_url = payload.get("repo_url")
    if repo is not None and repo_url is not None:
        raise HTTPException(status_code=400, detail="Use repo; do not send both repo and repo_url")
    selected_repo = repo if repo is not None else repo_url
    if selected_repo is None:
        raise HTTPException(status_code=400, detail="repo is required in LAN mode")
    task = payload.get("task")
    if task is None:
        raise HTTPException(status_code=400, detail="task is required in LAN mode")
    version = payload.get("version")
    if version is not None and not _SAFE_JOB_TOKEN.fullmatch(version):
        raise HTTPException(status_code=400, detail="version contains unsupported characters")
    if _lan_state_dir is None:
        raise HTTPException(status_code=503, detail="LAN state is not initialized")

    lock = _lan_prepare_lock
    if lock is None:
        raise HTTPException(status_code=503, detail="LAN checkout manager is not initialized")
    async with lock:
        try:
            normalized_repo = normalize_repo_url(selected_repo)
            selected_key = repo_key(normalized_repo)
            for existing_runtime in _projects.values():
                if existing_runtime.lan_repo_key != selected_key:
                    continue
                active = (
                    existing_runtime.store.list_job_ids(status=JobStatus.PENDING)
                    + existing_runtime.store.list_job_ids(status=JobStatus.RUNNING)
                )
                if active:
                    raise HTTPException(
                        status_code=409,
                        detail="This repository already has pending or running work",
                    )
            prepared = await asyncio.to_thread(
                prepare_lan_project,
                _lan_state_dir,
                normalized_repo,
                task,
            )
        except LanConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime = _install_lan_runtime(prepared)

    sanitized = {
        key: value
        for key, value in payload.items()
        if key in {"version", "force", "triggered_by"}
    }
    sanitized["project"] = runtime.project
    return runtime, sanitized


def _install_lan_runtime(prepared: LanPreparedProject) -> ProjectRuntime:
    """Install or refresh a fully prepared dynamic runtime."""
    for runtime in _projects.values():
        if runtime.lan_repo_key != prepared.repo_key:
            continue
        active = (
            runtime.store.list_job_ids(status=JobStatus.PENDING)
            + runtime.store.list_job_ids(status=JobStatus.RUNNING)
        )
        if active:
            raise HTTPException(
                status_code=409,
                detail="This repository already has pending or running work",
            )

    existing = _projects.get(prepared.project)
    if existing is not None:
        existing.config = prepared.config
        existing.runner.update_config(prepared.config)
        return existing

    store = JobStore(prepared.config.log_dir / "jobs.db")
    runner = JobRunner(store, prepared.config)
    runtime = ProjectRuntime(
        prepared.project,
        None,
        prepared.config,
        store,
        runner,
        lan_repo_key=prepared.repo_key,
    )
    _recover_stale_jobs(store)
    for job_id in store.list_job_ids():
        found = _find_job_runtime(job_id)
        if found is not None:
            raise RuntimeError(f"Duplicate historical job id {job_id!r} in LAN state")
    _projects[prepared.project] = runtime
    global _default_project, _store, _runner, _config, _config_path
    if _default_project is None:
        _default_project = prepared.project
        _store = store
        _runner = runner
        _config = prepared.config
        _config_path = None
    return runtime


@app.get("/api/jobs", dependencies=[Depends(_api_key_auth)])
async def list_jobs(limit: int = 20) -> list[dict]:
    jobs = [
        _with_project(runtime, job)
        for runtime in _runtime_map().values()
        for job in runtime.store.list_jobs(limit=limit)
    ]
    return sorted(jobs, key=lambda job: job.get("created_at") or "", reverse=True)[:limit]


@app.get("/api/jobs/{job_id}", dependencies=[Depends(_api_key_auth)])
async def get_job(job_id: str) -> dict:
    found = _find_job_runtime(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Job not found")
    runtime, job = found
    return _with_project(runtime, job)


@app.get("/api/jobs/{job_id}/logs", dependencies=[Depends(_api_key_auth)])
async def stream_logs(job_id: str, request: Request, tail: int | None = None) -> Response:
    """Stream job logs as plain text. Accepts ?tail=N to return last N lines."""
    found = _find_job_runtime(job_id)
    if not found:
        return PlainTextResponse("Log not available yet.\n", status_code=404)
    runtime, _ = found
    log_path = runtime.config.log_dir / f"{job_id}.log"
    if not log_path.exists():
        return PlainTextResponse("Log not available yet.\n")

    if tail is not None:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        text = "\n".join(lines[-max(tail, 0):])
        if text:
            text += "\n"
        return PlainTextResponse(text)

    async def _generate():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(4096)
                if chunk:
                    yield chunk
                else:
                    if runtime.store.get_job(job_id, field="status") in (
                        JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.PROMOTED
                    ):
                        break
                    await asyncio.sleep(0.5)
                    if await request.is_disconnected():
                        break

    return StreamingResponse(_generate(), media_type="text/plain")


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(_api_key_auth)])
async def cancel_job(job_id: str) -> dict:
    """Cancel a pending or running job.

    Pending jobs are marked failed and will never start.
    Running jobs are terminated (subprocess killed) if possible.
    """
    found = _find_job_runtime(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Job not found")
    runtime, job = found
    if job["status"] in (JobStatus.PROMOTED, JobStatus.FAILED):
        raise HTTPException(status_code=400, detail=f"Job already terminal: {job['status']}")

    if job["status"] == JobStatus.PENDING:
        # A freshly claimed task may still have a pending DB row. Cancel that
        # reservation as well so it cannot start after this response.
        task = _running_tasks.pop(job_id, None)
        if task is not None:
            task.cancel()
        runtime.store.set_failed(job_id, "Cancelled via API")
        log.info("Pending job %s cancelled", job_id)
        return {
            "job_id": job_id,
            "project": runtime.project,
            "status": "failed",
            "error": "Cancelled via API",
        }

    # Running — kill subprocess and cancel task
    killed = await runtime.runner.cancel_job(job_id)
    task = _running_tasks.pop(job_id, None)
    if task is not None:
        task.cancel()
    runtime.store.set_failed(job_id, "Cancelled via API" + ("" if killed else " (subprocess could not be terminated)"))
    log.info("Running job %s cancelled (killed=%s)", job_id, killed)
    return {
        "job_id": job_id,
        "project": runtime.project,
        "status": "failed",
        "error": "Cancelled via API",
        "subprocess_killed": killed,
    }


@app.post("/api/jobs/{job_id}/promote", dependencies=[Depends(_api_key_auth)])
async def promote_job(job_id: str) -> dict:
    """Manually promote a validated job's models to git."""
    found = _find_job_runtime(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Job not found")
    runtime, job = found
    if job["status"] not in (JobStatus.COMPLETED, JobStatus.VALIDATED):
        raise HTTPException(status_code=400, detail=f"Job status {job['status']} cannot be promoted")
    asyncio.create_task(runtime.runner.promote_job(job_id))
    return {"job_id": job_id, "project": runtime.project, "status": "promoting"}


@app.get("/api/models", dependencies=[Depends(_api_key_auth)])
async def list_models(project: str | None = None) -> list[dict]:
    """List promoted model versions in the git repo."""
    runtime = _select_runtime(project)
    models_dir = Path(runtime.config.repo.local_path) / "models"
    if not models_dir.exists():
        return []
    return [
        {
            "name": d.name,
            "path": str(d),
            "mtime": datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        for d in sorted(models_dir.iterdir())
        if d.is_dir() and d.name.startswith("cv_")
    ]


def _next_version(config: TrainingConfig) -> str:
    """Auto-increment the numeric vN suffix from existing cv_vN model dirs."""
    models_dir = Path(config.repo.local_path) / "models"
    if not models_dir.exists():
        return "v1"
    existing = [
        int(d.name.removeprefix("cv_v"))
        for d in models_dir.iterdir()
        if d.is_dir() and d.name.startswith("cv_v") and d.name.removeprefix("cv_v").isdigit()
    ]
    next_n = max(existing, default=0) + 1
    return f"v{next_n}"


def _normalize_version(version: Any) -> str:
    if version is None:
        return ""
    text = str(version).strip()
    if not text:
        return ""
    if text.startswith("cv_"):
        return text[3:]
    return text


def main(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    projects_config: str | None = None,
    config: str | None = None,
    lan: bool = False,
    state_dir: str | None = None,
    max_concurrent_jobs: int | None = None,
) -> None:
    if not lan and (state_dir is not None or max_concurrent_jobs is not None):
        raise ValueError("--state-dir and --max-concurrent-jobs require --lan")
    if lan:
        os.environ["TRAINERD_LAN_MODE"] = "1"
        os.environ.pop("TRAINERD_PROJECTS_CONFIG", None)
        os.environ.pop("TRAINING_CONFIG", None)
        if state_dir:
            os.environ["TRAINERD_STATE_DIR"] = str(Path(state_dir).expanduser().resolve())
        else:
            os.environ.pop("TRAINERD_STATE_DIR", None)
        if max_concurrent_jobs is not None:
            if not 1 <= max_concurrent_jobs <= 64:
                raise ValueError("--max-concurrent-jobs must be from 1 to 64")
            os.environ["TRAINERD_MAX_CONCURRENT_JOBS"] = str(max_concurrent_jobs)
        else:
            os.environ.pop("TRAINERD_MAX_CONCURRENT_JOBS", None)
        server_port = 7860
    elif projects_config:
        os.environ.pop("TRAINERD_LAN_MODE", None)
        os.environ["TRAINERD_PROJECTS_CONFIG"] = str(Path(projects_config).resolve())
        os.environ.pop("TRAINING_CONFIG", None)
    elif config:
        os.environ.pop("TRAINERD_LAN_MODE", None)
        os.environ["TRAINING_CONFIG"] = str(Path(config).resolve())
        os.environ.pop("TRAINERD_PROJECTS_CONFIG", None)
    else:
        os.environ.pop("TRAINERD_LAN_MODE", None)
    if not lan:
        cfg = load_server_config()
        server_port = cfg.server_port
    uvicorn.run(
        "trainerd.server:app",
        host=host,
        port=port if port is not None else server_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
