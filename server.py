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
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from .config import load_config, TrainingConfig
from .contracts import validate_payload
from .runner import JobRunner
from .storage import JobStore, JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_store: JobStore | None = None
_runner: JobRunner | None = None
_config: TrainingConfig | None = None
_config_path: Path | None = None

# Queue worker state
_queue_worker_task: asyncio.Task | None = None
_running_tasks: dict[str, asyncio.Task] = {}
_queue_poll_interval: float = 5.0


def _api_key_auth(request: Request) -> None:
    if not _config or not _config.api_key:
        return
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != _config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _recover_stale_jobs(store: JobStore) -> None:
    """On startup, mark any running jobs as failed (interrupted by shutdown)."""
    stale = store.list_jobs(status=JobStatus.RUNNING)
    for job in stale:
        log.warning("Recovering stale running job %s — marking as failed (interrupted)", job["job_id"])
        store.set_failed(job["job_id"], "Interrupted — server restarted")


async def _queue_worker() -> None:
    """Background task that polls for pending jobs and starts them up to max_concurrent_jobs."""
    global _store, _runner, _config
    log.info("Queue worker started (max_concurrent_jobs=%s)", _config.max_concurrent_jobs if _config else 1)
    while True:
        try:
            if _store is None or _runner is None or _config is None:
                await asyncio.sleep(1)
                continue

            running = _store.list_jobs(status=JobStatus.RUNNING)
            available = _config.max_concurrent_jobs - len(running)

            if available > 0:
                pending = _store.list_jobs(status=JobStatus.PENDING, limit=available, oldest_first=True)
                for job in pending:
                    jid = job["job_id"]
                    if jid in _running_tasks:
                        continue
                    log.info("Queue worker claiming job %s", jid)
                    task = asyncio.create_task(_run_job_wrapper(jid))
                    _running_tasks[jid] = task
                    task.add_done_callback(lambda t, jid=jid: _running_tasks.pop(jid, None))

            await asyncio.sleep(_queue_poll_interval)
        except asyncio.CancelledError:
            log.info("Queue worker cancelled")
            break
        except Exception:
            log.exception("Queue worker error")
            await asyncio.sleep(1)


async def _run_job_wrapper(job_id: str) -> None:
    """Wrap runner.run_job to handle exceptions."""
    global _runner
    try:
        await _runner.run_job(job_id)
    except asyncio.CancelledError:
        log.info("Job %s was cancelled", job_id)
        if _store:
            job = _store.get_job(job_id)
            if job and job["status"] in (JobStatus.PENDING, JobStatus.RUNNING):
                _store.set_failed(job_id, "Cancelled via API")
    except Exception:
        log.exception("Unexpected error in job %s", job_id)
        if _store:
            _store.set_failed(job_id, "Internal error — see server logs")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _store, _runner, _config, _config_path, _queue_worker_task
    _config_path = Path(os.environ.get("TRAINING_CONFIG", "training_config.yaml"))
    _config = load_config(_config_path)
    _store = JobStore(_config.log_dir / "jobs.db")
    _runner = JobRunner(_store, _config, config_path=_config_path)

    # Recover stale running jobs from previous run
    _recover_stale_jobs(_store)

    # Start background queue worker
    _queue_worker_task = asyncio.create_task(_queue_worker())

    log.info("Training server ready. Project: %s  max_concurrent_jobs=%s", _config.project, _config.max_concurrent_jobs)
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


app = FastAPI(title="trainerd", version="0.1.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _refresh_runtime_config() -> TrainingConfig:
    global _config, _runner
    assert _config_path is not None
    assert _store is not None
    _config = load_config(_config_path)
    if _runner is None:
        _runner = JobRunner(_store, _config, config_path=_config_path)
    else:
        _runner.update_config(_config, config_path=_config_path)
    return _config


@app.get("/api/health")
async def health() -> dict:
    if _store and _config_path:
        _refresh_runtime_config()
    pending = len(_store.list_jobs(status=JobStatus.PENDING)) if _store else 0
    running = len(_store.list_jobs(status=JobStatus.RUNNING)) if _store else 0
    max_jobs = _config.max_concurrent_jobs if _config else 1
    return {
        "status": "ok",
        "project": _config.project if _config else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pending_jobs": pending,
        "running_jobs": running,
        "max_concurrent_jobs": max_jobs,
        "queue_capacity": max_jobs - running,
    }


@app.post("/api/jobs", dependencies=[Depends(_api_key_auth)])
async def submit_job(body: dict = {}) -> dict:
    """Submit a training job. The job is queued and executed when a slot is available.

    Optional body fields:
      - steps: list of step IDs (default: all configured steps)
      - version: version string (default: auto-incremented)
      - branch: git branch
      - markets: market filter string
      - extra_args: extra CLI args appended to the training command
      - force: if true, skip dedupe check (default: false)
    """
    assert _store and _runner and _config
    if _config_path:
        _refresh_runtime_config()
    problems = validate_payload(body)
    if problems:
        raise HTTPException(status_code=400, detail={"problems": problems})
    requested_steps = body.get("steps")
    configured_step_ids = [s.id for s in _config.steps]
    if requested_steps:
        steps = [str(step).strip() for step in requested_steps if str(step).strip()]
        unknown_steps = sorted(set(steps) - set(configured_step_ids))
        if unknown_steps:
            raise HTTPException(status_code=400, detail=f"Unknown step ids: {', '.join(unknown_steps)}")
        if not steps:
            raise HTTPException(status_code=400, detail="No valid step ids requested")
    else:
        steps = configured_step_ids
    version = _normalize_version(body.get("version")) or _next_version(_config)
    branch = body.get("branch")
    markets = body.get("markets")
    extra_args = body.get("extra_args")
    force = body.get("force", False)

    # Dedupe guard: reject duplicate pending/running jobs for same parameters
    if not force:
        dup = _store.find_pending_or_running(version=version, branch=branch, markets=markets, extra_args=extra_args, steps=steps)
        if dup is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Duplicate job {dup['job_id']} already {'pending' if dup['status'] == JobStatus.PENDING else 'running'} "
                       f"for version={version} branch={branch} markets={markets}. Set force=true to override.",
            )

    job_id = str(uuid.uuid4())[:8]
    job = _store.create_job(
        job_id,
        steps=steps,
        version=version,
        triggered_by=body.get("triggered_by", "api"),
        branch=branch,
        markets=markets,
        extra_args=extra_args,
    )
    log.info("Job %s queued: steps=%s version=%s force=%s", job_id, steps, version, force)
    return {"job_id": job_id, "status": job["status"], "version": version, "steps": steps, "queued": True}


@app.get("/api/jobs", dependencies=[Depends(_api_key_auth)])
async def list_jobs(limit: int = 20) -> list[dict]:
    assert _store
    return _store.list_jobs(limit=limit)


@app.get("/api/jobs/{job_id}", dependencies=[Depends(_api_key_auth)])
async def get_job(job_id: str) -> dict:
    assert _store
    job = _store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str, request: Request, tail: int | None = None) -> Response:
    """Stream job logs as plain text. Accepts ?tail=N to return last N lines."""
    assert _config
    log_path = _config.log_dir / f"{job_id}.log"
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
                    if _store and _store.get_job(job_id, field="status") in (
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
    assert _store and _runner
    job = _store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in (JobStatus.PROMOTED, JobStatus.FAILED):
        raise HTTPException(status_code=400, detail=f"Job already terminal: {job['status']}")

    if job["status"] == JobStatus.PENDING:
        # Pending — just mark as failed; it will be skipped by queue worker
        _store.set_failed(job_id, "Cancelled via API")
        log.info("Pending job %s cancelled", job_id)
        return {"job_id": job_id, "status": "failed", "error": "Cancelled via API"}

    # Running — kill subprocess and cancel task
    killed = await _runner.cancel_job(job_id)
    task = _running_tasks.pop(job_id, None)
    if task is not None:
        task.cancel()
    _store.set_failed(job_id, "Cancelled via API" + ("" if killed else " (subprocess could not be terminated)"))
    log.info("Running job %s cancelled (killed=%s)", job_id, killed)
    return {
        "job_id": job_id,
        "status": "failed",
        "error": "Cancelled via API",
        "subprocess_killed": killed,
    }


@app.post("/api/jobs/{job_id}/promote", dependencies=[Depends(_api_key_auth)])
async def promote_job(job_id: str) -> dict:
    """Manually promote a validated job's models to git."""
    assert _store and _runner
    job = _store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in (JobStatus.COMPLETED, JobStatus.VALIDATED):
        raise HTTPException(status_code=400, detail=f"Job status {job['status']} cannot be promoted")
    asyncio.create_task(_runner.promote_job(job_id))
    return {"job_id": job_id, "status": "promoting"}


@app.get("/api/models", dependencies=[Depends(_api_key_auth)])
async def list_models() -> list[dict]:
    """List promoted model versions in the git repo."""
    assert _config
    models_dir = Path(_config.repo.local_path) / "models"
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


def main() -> None:
    config_path = Path(os.environ.get("TRAINING_CONFIG", "training_config.yaml"))
    cfg = load_config(config_path)
    uvicorn.run("trainerd.server:app", host="0.0.0.0", port=cfg.server_port, reload=False)


if __name__ == "__main__":
    main()
