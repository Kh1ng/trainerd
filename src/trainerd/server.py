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
import hashlib
import hmac
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import (
    FileResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.security import APIKeyHeader, APIKeyQuery

from . import __version__
from .config import TrainingConfig, load_server_config
from .contracts import (
    ARTIFACT_MANIFEST_SCHEMA,
    JobRequest,
    is_safe_identifier,
    validate_payload,
)
from .lan import (
    LanConfigError,
    LanRepositoryPolicy,
    default_state_dir,
    load_lan_server_config,
    normalize_repo_url,
    prepare_lan_project,
    prepare_persisted_lan_project,
)
from .runner import StageQueuePool
from .runtime import (
    DaemonRuntime,
    ProjectRuntime,
    RepositoryCapacityError,
)
from .storage import JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_runtime = DaemonRuntime()
_LAN_API_KEY_ENV = "TRAINERD_API_KEY"
_LAN_ALLOWED_REPOS_ENV = "TRAINERD_ALLOWED_REPOS"
_MAX_ARTIFACT_MANIFEST_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_ARTIFACTS = 256


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def _configured_api_key() -> str:
    return os.environ.get(_LAN_API_KEY_ENV, "").strip() if _runtime.lan_mode else _runtime.api_key


def _api_key_auth(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> None:
    api_key = _configured_api_key()
    if not api_key:
        return
    key = header_key or query_key
    if key is None or not hmac.compare_digest(key, api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


def _required_api_key_auth(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> None:
    if not _configured_api_key():
        raise HTTPException(status_code=503, detail="Artifact retrieval requires an API key")
    _api_key_auth(header_key, query_key)


def _read_api_key_auth(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> None:
    if not _runtime.lan_mode:
        _api_key_auth(header_key, query_key)


def _lan_allowed_repo_urls() -> frozenset[str]:
    if _runtime.lan_repositories:
        return frozenset(_runtime.lan_repositories)
    values = [
        value.strip()
        for value in os.environ.get(_LAN_ALLOWED_REPOS_ENV, "").splitlines()
        if value.strip()
    ]
    try:
        return frozenset(normalize_repo_url(value) for value in values)
    except LanConfigError as exc:
        raise ValueError(f"{_LAN_ALLOWED_REPOS_ENV} is invalid: {exc}") from exc


def _load_lan_repositories() -> dict[str, LanRepositoryPolicy]:
    if _runtime.lan_config_path is not None:
        try:
            return load_lan_server_config(_runtime.lan_config_path)
        except LanConfigError as exc:
            raise ValueError(f"LAN configuration is invalid: {exc}") from exc
    values = [
        value.strip()
        for value in os.environ.get(_LAN_ALLOWED_REPOS_ENV, "").splitlines()
        if value.strip()
    ]
    try:
        urls = {normalize_repo_url(value) for value in values}
    except LanConfigError as exc:
        raise ValueError(f"{_LAN_ALLOWED_REPOS_ENV} is invalid: {exc}") from exc
    return {repo_url: LanRepositoryPolicy(repo_url, None) for repo_url in urls}


def _validate_lan_security(
    allowed_repos: frozenset[str] | None = None,
) -> frozenset[str]:
    allowed_repos = allowed_repos if allowed_repos is not None else _lan_allowed_repo_urls()
    if allowed_repos and not os.environ.get(_LAN_API_KEY_ENV, "").strip():
        raise ValueError(
            f"{_LAN_API_KEY_ENV} is required when LAN repositories are allowlisted"
        )
    return allowed_repos


def _select_runtime(project: Any = None) -> ProjectRuntime:
    try:
        return _runtime.select(project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_multi_project_request(payload: dict[str, Any]) -> None:
    if not _runtime.lan_mode and any(field in payload for field in ("repo", "repo_url", "task")):
        raise HTTPException(
            status_code=400,
            detail="repo and task are accepted only when trainerd starts with --lan",
        )
    if not _runtime.registry_mode:
        return
    if "branch" in payload:
        raise HTTPException(status_code=400, detail="branch is not accepted in registry mode")
    if "extra_args" in payload:
        raise HTTPException(status_code=400, detail="extra_args is not accepted in registry mode")

    version = payload.get("version")
    if version is not None and not is_safe_identifier(version):
        raise HTTPException(status_code=400, detail="version contains unsupported characters")

    steps = payload.get("steps")
    if steps is not None:
        if not steps or any(not is_safe_identifier(step) for step in steps):
            raise HTTPException(status_code=400, detail="steps must be non-empty safe step ids")

    markets = payload.get("markets")
    if markets is not None:
        market_ids = markets.split(",")
        if (
            not market_ids
            or len(market_ids) > 64
            or any(not is_safe_identifier(market) for market in market_ids)
        ):
            raise HTTPException(
                status_code=400,
                detail="markets must be a comma-separated list of safe ids",
            )


def _with_project(runtime: ProjectRuntime, job: dict) -> dict:
    return {**job, "project": runtime.project}


async def _queue_worker() -> None:
    """Dispatch queued jobs within daemon-wide and per-project limits."""
    max_jobs = _runtime.max_concurrent_jobs
    log.info("Queue worker started (max_concurrent_jobs=%s)", max_jobs)
    while True:
        try:
            runtimes = _runtime.projects
            if not runtimes:
                await _runtime.wait_for_queue()
                continue
            for runtime in runtimes.values():
                _runtime.refresh(runtime)

            for runtime, job in _runtime.pending_candidates(max_jobs):
                jid = job["job_id"]
                if jid in _runtime.running_tasks:
                    continue
                log.info("Queue worker claiming job %s for project %s", jid, runtime.project)
                task = asyncio.create_task(_run_job_wrapper(jid, runtime.project))
                _runtime.running_tasks[jid] = task

            await _runtime.wait_for_queue()
        except asyncio.CancelledError:
            log.info("Queue worker cancelled")
            break
        except Exception:
            log.exception("Queue worker error")
            await asyncio.sleep(1)


async def _run_job_wrapper(job_id: str, project: str | None = None) -> None:
    """Wrap runner.run_job to handle exceptions."""
    try:
        found = _runtime.find_job(job_id)
        runtime = (
            _select_runtime(project)
            if project
            else (found[0] if found else _runtime.default())
        )
        try:
            await runtime.runner.run_job(job_id)
        except asyncio.CancelledError:
            log.info("Job %s was cancelled", job_id)
            job = runtime.store.get_job(job_id)
            if job and job["status"] in (JobStatus.PENDING, JobStatus.RUNNING):
                stages = job.get("stages") or {}
                if stages and not any(
                    stage.get("status") in {"running", "failed"}
                    for stage in stages.values()
                ):
                    runtime.store.set_pending(job_id)
                else:
                    runtime.store.set_failed(job_id, "Cancelled via API")
            raise
        except Exception:
            log.exception("Unexpected error in job %s", job_id)
            runtime.store.set_failed(job_id, "Internal error — see server logs")
    finally:
        _runtime.running_tasks.pop(job_id, None)
        _runtime.wake_queue()


async def _recover_lan_runtimes() -> None:
    """Load persisted LAN runtimes that still own active jobs."""
    state_dir = _runtime.lan_state_dir
    if state_dir is None:
        return
    for project, store, _ in _runtime.persisted_lan_stores():
        active = store.list_job_ids(status=JobStatus.PENDING)
        active += store.list_job_ids(status=JobStatus.RUNNING)
        if not active:
            continue
        try:
            prepared = await asyncio.to_thread(
                prepare_persisted_lan_project,
                state_dir,
                project,
                _runtime.lan_repositories,
            )
            _runtime.install_lan(prepared)
            log.info("Recovered LAN project %s with %s active job(s)", project, len(active))
        except (LanConfigError, RuntimeError):
            log.exception("Could not recover active LAN project %s", project)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        cpu_limit = int(os.environ.get("TRAINERD_CPU_CONCURRENCY", "1"))
    except ValueError as exc:
        raise ValueError("TRAINERD_CPU_CONCURRENCY must be an integer") from exc
    try:
        gpu_capacity = int(os.environ.get("TRAINERD_GPU_CAPACITY", "1"))
    except ValueError as exc:
        raise ValueError("TRAINERD_GPU_CAPACITY must be an integer") from exc
    stage_queues = StageQueuePool(cpu=cpu_limit, gpu=gpu_capacity)
    lan_mode = os.environ.get("TRAINERD_LAN_MODE") == "1"
    allowed_repos: frozenset[str] = frozenset()
    if lan_mode:
        configured_state = os.environ.get("TRAINERD_STATE_DIR", "").strip()
        lan_state_dir = (
            Path(configured_state).expanduser().resolve()
            if configured_state
            else default_state_dir().expanduser().resolve()
        )
        lan_state_dir.mkdir(parents=True, exist_ok=True)
        try:
            max_concurrent_jobs = int(
                os.environ.get("TRAINERD_MAX_CONCURRENT_JOBS", "1")
            )
        except ValueError as exc:
            raise ValueError("TRAINERD_MAX_CONCURRENT_JOBS must be an integer") from exc
        if not 1 <= max_concurrent_jobs <= 64:
            raise ValueError("TRAINERD_MAX_CONCURRENT_JOBS must be from 1 to 64")
        lan_config_path = _runtime.lan_config_path
        repositories = _load_lan_repositories()
        allowed_repos = _validate_lan_security(frozenset(repositories))
        _runtime.configure_lan(
            lan_state_dir,
            max_concurrent_jobs,
            stage_queues,
            repositories=repositories,
            config_path=lan_config_path,
        )
        await _recover_lan_runtimes()
    else:
        _runtime.configure_projects(load_server_config(), stage_queues)

    _runtime.queue_wake = asyncio.Event()
    _runtime.queue_worker_task = asyncio.create_task(_queue_worker())

    if _runtime.lan_mode and not allowed_repos:
        log.warning(
            "LAN mode accepts arbitrary repositories on managed state %s; "
            "configure --allow-repo for a repository boundary",
            _runtime.lan_state_dir,
        )
    elif _runtime.lan_mode:
        log.info(
            "Constrained LAN mode enabled on %s with %s allowlisted repository(s)",
            _runtime.lan_state_dir,
            len(allowed_repos),
        )
    else:
        log.info(
            "Training server ready. Projects: %s  default=%s  max_concurrent_jobs=%s",
            sorted(_runtime.projects),
            _runtime.default_project,
            _runtime.max_concurrent_jobs,
        )
    yield
    log.info("Training server shutting down.")
    if _runtime.queue_worker_task:
        _runtime.queue_worker_task.cancel()
        try:
            await _runtime.queue_worker_task
        except asyncio.CancelledError:
            # Cancellation is the expected queue-worker shutdown signal.
            pass
    # Cancel any running training tasks
    tasks = list(_runtime.running_tasks.items())
    for jid, task in tasks:
        found = _runtime.find_job(jid)
        if found:
            await found[0].runner.cancel_job(jid)
        task.cancel()
    if tasks:
        await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
    _runtime.reset()


app = FastAPI(title="trainerd", version=__version__, lifespan=_lifespan)


@app.get("/api/health")
async def health() -> dict:
    runtimes = _runtime.projects
    for runtime in runtimes.values():
        _runtime.refresh(runtime)
    pending = sum(len(runtime.store.list_job_ids(status=JobStatus.PENDING)) for runtime in runtimes.values())
    running = sum(len(runtime.store.list_job_ids(status=JobStatus.RUNNING)) for runtime in runtimes.values())
    active = len(set().union(*_runtime.active_job_ids(runtimes).values()))
    max_jobs = _runtime.max_concurrent_jobs
    default = _runtime.default_project
    return {
        "status": "ok",
        "version": __version__,
        "build_commit": os.environ.get("TRAINERD_BUILD_COMMIT"),
        "project": default,
        "projects": sorted(runtimes),
        "default_project": default,
        "mode": _runtime.mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pending_jobs": pending,
        "running_jobs": running,
        "max_concurrent_jobs": max_jobs,
        "queue_capacity": max(max_jobs - active, 0),
        "stage_queues": _runtime.stage_queues.snapshot() if _runtime.stage_queues else None,
        "authentication_required": bool(_configured_api_key()),
        "allowed_repository_count": (
            len(_lan_allowed_repo_urls()) if _runtime.lan_mode else None
        ),
        "lan_policy_hash": _runtime.lan_policy_hash() if _runtime.lan_mode else None,
    }


@app.get("/api/lan/config", dependencies=[Depends(_required_api_key_auth)])
async def lan_config() -> dict:
    """List normalized LAN repositories and task sources without executable fields."""
    if not _runtime.lan_mode:
        raise HTTPException(status_code=404, detail="LAN configuration is not active")
    return _runtime.lan_config_view()


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
    if _runtime.lan_mode:
        lock = _runtime.lan_prepare_lock
        if lock is None:
            raise HTTPException(status_code=503, detail="LAN checkout manager is not initialized")
        async with lock:
            runtime, payload = await _prepare_lan_runtime(payload)
            return _queue_job(runtime, payload)
    else:
        runtime = _select_runtime(payload.get("project"))
    return _queue_job(runtime, payload)


def _queue_job(runtime: ProjectRuntime, payload: dict[str, Any]) -> dict:
    """Validate and persist one job against its selected runtime."""
    if _runtime.lan_mode:
        try:
            _runtime.ensure_lan_capacity(runtime)
        except RepositoryCapacityError as exc:
            raise HTTPException(
                status_code=409,
                detail="This repository has reached its configured concurrent job limit",
            ) from exc
    config = _runtime.refresh(runtime)
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
    if _runtime.stage_queues:
        try:
            for step in config.steps:
                if step.id in steps and step.queue:
                    _runtime.stage_queues.validate(step.queue, step.units)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    version = _normalize_version(payload.get("version")) or _next_version(config)
    branch = payload.get("branch")
    markets = payload.get("markets")
    extra_args = payload.get("extra_args")
    repo_sha = payload.get("repo_sha")
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
    while _runtime.find_job(job_id) is not None:
        job_id = str(uuid.uuid4())[:8]
    job = store.create_job(
        job_id,
        steps=steps,
        version=version,
        triggered_by=payload.get("triggered_by", "api"),
        branch=branch,
        markets=markets,
        extra_args=extra_args,
        repo_sha=repo_sha,
        task_definition_hash=config.lan_task_definition_hash,
        stage_queues={step.id: step.queue for step in config.steps if step.id in steps}
        if config.steps and all(step.queue for step in config.steps)
        else None,
        stage_units={step.id: step.units for step in config.steps if step.id in steps},
    )
    _runtime.wake_queue()
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
    allowed = {"repo", "repo_url", "task", "branch", "version", "steps", "force", "triggered_by"}
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
    if version is not None and not is_safe_identifier(version):
        raise HTTPException(status_code=400, detail="version contains unsupported characters")
    if _runtime.lan_state_dir is None:
        raise HTTPException(status_code=503, detail="LAN state is not initialized")

    try:
        normalized_repo = normalize_repo_url(selected_repo)
        try:
            allowed_repos = _lan_allowed_repo_urls()
        except ValueError as exc:
            raise LanConfigError(str(exc)) from exc
        if allowed_repos and normalized_repo not in allowed_repos:
            raise HTTPException(
                status_code=403,
                detail="Repository is not allowlisted on this trainerd host",
            )
        policy = _runtime.lan_repositories.get(normalized_repo)
        prepared = await asyncio.to_thread(
            prepare_lan_project,
            _runtime.lan_state_dir,
            normalized_repo,
            task,
            branch=payload.get("branch"),
            task_definitions=policy.tasks if policy else None,
        )
    except LanConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    runtime = _runtime.install_lan(prepared)

    sanitized = {
        key: value
        for key, value in payload.items()
        if key in {"branch", "version", "steps", "force", "triggered_by"}
    }
    sanitized["project"] = runtime.project
    sanitized["repo_sha"] = prepared.revision
    return runtime, sanitized


@app.get("/api/jobs", dependencies=[Depends(_read_api_key_auth)])
async def list_jobs(limit: int = 20) -> list[dict]:
    jobs = [
        _with_project(runtime, job)
        for runtime in _runtime.projects.values()
        for job in runtime.store.list_jobs(limit=limit)
    ]
    jobs.extend(
        {**job, "project": project}
        for project, store, _ in _runtime.persisted_lan_stores()
        for job in store.list_jobs(limit=limit)
    )
    return sorted(jobs, key=lambda job: job.get("created_at") or "", reverse=True)[:limit]


def _job_queue_entry(project: str, job: dict) -> dict:
    """Compact active-queue fields for one job, without secrets or commands."""
    stages = job.get("stages") or {}
    steps = job.get("steps") or []
    status = job.get("status")
    current_stage = None
    next_stage = None
    if stages:
        for step, stage in stages.items():
            if stage.get("status") == "running":
                current_stage = step
            elif stage.get("status") == "pending" and next_stage is None:
                next_stage = step
    elif status == JobStatus.RUNNING:
        current_stage = job.get("current_step")
        if current_stage and current_stage in steps:
            following = steps[steps.index(current_stage) + 1:]
            if following:
                next_stage = following[0]
    elif status == JobStatus.PENDING and steps:
        next_stage = steps[0]
    if current_stage is None and status == JobStatus.RUNNING:
        current_stage = job.get("current_step")
    stage_id = current_stage or next_stage
    queue = (stages.get(stage_id) or {}).get("queue") if stage_id else None
    return {
        "job_id": job.get("job_id"),
        "project": project,
        "version": job.get("version"),
        "status": status,
        "current_stage": current_stage,
        "next_stage": next_stage,
        "queue": queue,
        "queue_position": None,
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "steps": job.get("steps") or [],
    }


def _active_queue_entries() -> tuple[list[dict], list[dict]]:
    """Return (running, pending) entries matching health counts.

    Only loaded runtimes contribute. Dormant LAN stores are historical; a job
    still marked pending or running there is stale, so counting it would make
    the queue disagree with /api/health.
    """
    runtimes = _runtime.projects
    running: list[dict] = []
    for runtime in runtimes.values():
        for job in runtime.store.list_jobs(
            status=JobStatus.RUNNING, limit=None, oldest_first=True
        ):
            running.append(_job_queue_entry(runtime.project, job))
    running.sort(key=lambda entry: entry.get("started_at") or entry.get("created_at") or "")
    pending = [
        _job_queue_entry(runtime.project, job)
        for runtime, job in _runtime.pending_order()
    ]
    for position, entry in enumerate(pending, start=1):
        entry["queue_position"] = position
    return running, pending


@app.get("/api/queue", dependencies=[Depends(_read_api_key_auth)])
async def active_queue() -> dict:
    """Read-only ordered view of running and pending work for external monitors.

    Running jobs are listed first, oldest start first, followed by pending jobs
    in scheduler order. Each entry exposes only identifiers, status, stage/queue
    position, and times - never commands, environment values, or secrets.
    """
    runtimes = _runtime.projects
    for runtime in runtimes.values():
        _runtime.refresh(runtime)
    running, pending = _active_queue_entries()
    active = len(set().union(*_runtime.active_job_ids(runtimes).values()))
    max_jobs = _runtime.max_concurrent_jobs
    return {
        "jobs": running + pending,
        "pending_jobs": len(pending),
        "running_jobs": len(running),
        "max_concurrent_jobs": max_jobs,
        "queue_capacity": max(max_jobs - active, 0),
        "stage_queues": _runtime.stage_queues.snapshot() if _runtime.stage_queues else None,
        "authentication_required": bool(_configured_api_key()),
    }


@app.get("/api/jobs/{job_id}", dependencies=[Depends(_read_api_key_auth)])
async def get_job(job_id: str) -> dict:
    found = _runtime.find_job(job_id)
    if found:
        runtime, job = found
        return {
            **_with_project(runtime, job),
            "stage_queues": _runtime.stage_queues.snapshot() if _runtime.stage_queues else None,
        }
    persisted = _runtime.find_persisted_lan_job(job_id)
    if persisted:
        project, _, _, job = persisted
        return {
            **job,
            "project": project,
            "stage_queues": _runtime.stage_queues.snapshot() if _runtime.stage_queues else None,
        }
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/jobs/{job_id}/logs", dependencies=[Depends(_read_api_key_auth)])
async def stream_logs(job_id: str, request: Request, tail: int | None = None) -> Response:
    """Stream job logs as plain text. Accepts ?tail=N to return last N lines."""
    found = _runtime.find_job(job_id)
    if found:
        runtime, _ = found
        store = runtime.store
        log_dir = runtime.config.log_dir
    elif persisted := _runtime.find_persisted_lan_job(job_id):
        _, store, log_dir, _ = persisted
    else:
        return PlainTextResponse("Log not available yet.\n", status_code=404)
    log_path = log_dir / f"{job_id}.log"
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
                    if JobStatus.is_terminal(store.get_job(job_id, field="status")):
                        break
                    await asyncio.sleep(0.5)
                    if await request.is_disconnected():
                        break

    return StreamingResponse(_generate(), media_type="text/plain")


def _find_job_artifact_context(job_id: str) -> tuple[Path, dict] | None:
    """Return the daemon-owned work directory and job, including after restart."""
    found = _runtime.find_job(job_id)
    if found:
        runtime, job = found
        return runtime.config.work_dir, job
    persisted = _runtime.find_persisted_lan_job(job_id)
    if not persisted or _runtime.lan_state_dir is None:
        return None
    project, _, _, job = persisted
    match = re.fullmatch(
        r"lan-([0-9a-f]{20})-([A-Za-z0-9][A-Za-z0-9._-]{0,63})",
        project,
    )
    if not match:
        return None
    return _runtime.lan_state_dir / "work" / match[1] / match[2], job


def _validated_job_artifacts(
    work_dir: Path,
    job: dict,
    artifact_index: int | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    work_root = work_dir.resolve()
    job_root = (work_root / job["job_id"]).resolve()
    if not job_root.is_relative_to(work_root):
        raise HTTPException(status_code=422, detail="Artifact directory has an invalid path")
    manifest_path = (job_root / "artifact_manifest.json").resolve()
    if not manifest_path.is_relative_to(job_root):
        raise HTTPException(status_code=422, detail="Artifact manifest has an invalid path")
    try:
        if manifest_path.stat().st_size > _MAX_ARTIFACT_MANIFEST_BYTES:
            raise HTTPException(status_code=413, detail="Artifact manifest is too large")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact manifest not found") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Artifact manifest could not be read") from exc

    problems = validate_payload(manifest, ARTIFACT_MANIFEST_SCHEMA)
    if problems:
        raise HTTPException(status_code=422, detail="Invalid artifact manifest: " + "; ".join(problems))
    if manifest.get("job_id") != job["job_id"] or manifest.get("run_label") != job["version"]:
        raise HTTPException(status_code=422, detail="Artifact manifest does not match this job")
    entries = manifest["artifacts"]
    if len(entries) > _MAX_ARTIFACTS:
        raise HTTPException(status_code=413, detail="Artifact manifest contains too many files")
    declared_bytes = [entry.get("bytes") for entry in entries]
    if any(type(size) is not int or size < 0 for size in declared_bytes):
        raise HTTPException(status_code=422, detail="Artifact manifest has invalid byte counts")
    if sum(declared_bytes) > _MAX_ARTIFACT_BYTES:
        raise HTTPException(status_code=413, detail="Declared artifacts are too large")

    if artifact_index is not None and not 0 <= artifact_index < len(entries):
        raise HTTPException(status_code=404, detail="Artifact not found")
    selected = range(len(entries)) if artifact_index is None else (artifact_index,)
    paths: list[Path] = []
    for index in selected:
        entry = entries[index]
        try:
            relative = Path(entry["path"])
            path = (job_root / relative).resolve()
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Artifact {index} has an invalid path") from exc
        if relative.is_absolute() or not path.is_relative_to(job_root) or not path.is_file():
            raise HTTPException(status_code=422, detail=f"Artifact {index} has an invalid path")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise HTTPException(status_code=422, detail=f"Artifact {index} has invalid integrity metadata")
        try:
            if path.stat().st_size != expected_bytes:
                raise HTTPException(status_code=422, detail=f"Artifact {index} size does not match its manifest")
            digest = hashlib.sha256()
            with path.open("rb") as artifact_file:
                for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"Artifact {index} could not be read") from exc
        if not hmac.compare_digest(digest.hexdigest(), expected_sha256.lower()):
            raise HTTPException(status_code=422, detail=f"Artifact {index} hash does not match its manifest")
        paths.append(path)
    return manifest, paths


@app.get("/api/jobs/{job_id}/artifacts", dependencies=[Depends(_required_api_key_auth)])
async def list_job_artifacts(job_id: str) -> dict:
    context = _find_job_artifact_context(job_id)
    if not context:
        raise HTTPException(status_code=404, detail="Job not found")
    work_dir, job = context
    manifest, _ = await asyncio.to_thread(_validated_job_artifacts, work_dir, job)
    return {
        **manifest,
        "artifacts": [
            {
                **entry,
                "download_url": f"/api/jobs/{job_id}/artifacts/{index}",
            }
            for index, entry in enumerate(manifest["artifacts"])
        ],
    }


@app.get("/api/jobs/{job_id}/artifacts/{artifact_index}", dependencies=[Depends(_required_api_key_auth)])
async def download_job_artifact(job_id: str, artifact_index: int) -> Response:
    context = _find_job_artifact_context(job_id)
    if not context:
        raise HTTPException(status_code=404, detail="Job not found")
    work_dir, job = context
    _, paths = await asyncio.to_thread(
        _validated_job_artifacts, work_dir, job, artifact_index
    )
    return FileResponse(paths[0], filename=paths[0].name)


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(_api_key_auth)])
async def cancel_job(job_id: str) -> dict:
    """Cancel a pending or running job.

    Pending jobs are marked failed and will never start.
    Running jobs are terminated (subprocess killed) if possible.
    """
    found = _runtime.find_job(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Job not found")
    runtime, job = found
    if JobStatus.is_terminal(job["status"]):
        raise HTTPException(status_code=400, detail=f"Job already terminal: {job['status']}")

    if job["status"] == JobStatus.PENDING:
        # A freshly claimed task may still have a pending DB row. Cancel that
        # reservation as well so it cannot start after this response.
        task = _runtime.running_tasks.pop(job_id, None)
        if task is not None:
            task.cancel()
        runtime.store.set_failed(job_id, "Cancelled via API")
        _runtime.wake_queue()
        log.info("Pending job %s cancelled", job_id)
        return {
            "job_id": job_id,
            "project": runtime.project,
            "status": "failed",
            "error": "Cancelled via API",
        }

    # Running — kill subprocess and cancel task
    killed = await runtime.runner.cancel_job(job_id)
    task = _runtime.running_tasks.pop(job_id, None)
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
    found = _runtime.find_job(job_id)
    if not found:
        raise HTTPException(status_code=404, detail="Job not found")
    runtime, job = found
    if job["status"] not in (JobStatus.COMPLETED, JobStatus.VALIDATED):
        raise HTTPException(status_code=400, detail=f"Job status {job['status']} cannot be promoted")
    asyncio.create_task(runtime.runner.promote_job(job_id))
    return {"job_id": job_id, "project": runtime.project, "status": "promoting"}


@app.get("/api/models", dependencies=[Depends(_read_api_key_auth)])
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
    cpu_concurrency: int | None = None,
    gpu_capacity: int | None = None,
    allowed_repos: list[str] | None = None,
    lan_config: str | None = None,
) -> None:
    if not lan and (
        state_dir is not None
        or max_concurrent_jobs is not None
        or allowed_repos is not None
        or lan_config is not None
    ):
        raise ValueError(
            "--state-dir, --max-concurrent-jobs, --allow-repo, and --lan-config require --lan"
        )
    if not lan:
        os.environ.pop(_LAN_ALLOWED_REPOS_ENV, None)
        _runtime.lan_config_path = None
    if cpu_concurrency is not None:
        if not 1 <= cpu_concurrency <= 64:
            raise ValueError("--cpu-concurrency must be from 1 to 64")
        os.environ["TRAINERD_CPU_CONCURRENCY"] = str(cpu_concurrency)
    if gpu_capacity is not None:
        if not 1 <= gpu_capacity <= 64:
            raise ValueError("--gpu-capacity must be from 1 to 64")
        os.environ["TRAINERD_GPU_CAPACITY"] = str(gpu_capacity)
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
        if allowed_repos:
            if lan_config:
                raise ValueError("Use --lan-config or --allow-repo, not both")
            os.environ[_LAN_ALLOWED_REPOS_ENV] = "\n".join(
                normalize_repo_url(value) for value in allowed_repos
            )
        else:
            os.environ.pop(_LAN_ALLOWED_REPOS_ENV, None)
        if lan_config:
            _runtime.lan_config_path = Path(lan_config).expanduser().resolve()
        else:
            _runtime.lan_config_path = None
        _validate_lan_security(frozenset(_load_lan_repositories()))
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
