"""SQLite-backed job store for the training server."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    FAILED = "failed"


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._db = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    version TEXT,
                    steps TEXT,
                    triggered_by TEXT,
                    created_at TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    current_step TEXT,
                    error TEXT,
                    validation_result TEXT,
                    promotion_ref TEXT,
                    branch TEXT,
                    markets TEXT,
                    extra_args TEXT
                )
            """)
            for col, ctype in [
                ("branch", "TEXT"),
                ("markets", "TEXT"),
                ("extra_args", "TEXT"),
                ("stages", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ctype}")
                except Exception:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS jobs_created_at ON jobs(created_at)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_status_created_at ON jobs(status, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS jobs_version_status_created_at "
                "ON jobs(version, status, created_at)"
            )

    def create_job(
        self,
        job_id: str,
        steps: list[str],
        version: str,
        triggered_by: str = "api",
        branch: str | None = None,
        markets: str | None = None,
        extra_args: str | None = None,
        stage_queues: dict[str, str] | None = None,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        stages = (
            {
                step: {
                    "queue": queue,
                    "status": "pending",
                    **({"queued_at": now} if index == 0 else {}),
                }
                for index, (step, queue) in enumerate(stage_queues.items())
            }
            if stage_queues
            else None
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, status, version, steps, triggered_by, created_at, branch, markets, extra_args, stages)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    JobStatus.PENDING,
                    version,
                    json.dumps(steps),
                    triggered_by,
                    now,
                    branch,
                    markets,
                    extra_args,
                    json.dumps(stages) if stages else None,
                ),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str, field: str | None = None) -> dict | Any | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["steps"] = json.loads(data["steps"] or "[]")
        data["stages"] = json.loads(data.get("stages") or "null")
        if data.get("validation_result"):
            try:
                data["validation_result"] = json.loads(data["validation_result"])
            except Exception:
                pass
        if field:
            return data.get(field)
        return data

    def list_jobs(self, limit: int = 20, status: str | None = None, *, oldest_first: bool = False) -> list[dict]:
        order = "ASC" if oldest_first else "DESC"
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    f"SELECT * FROM jobs WHERE status = ? ORDER BY created_at {order} LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT * FROM jobs ORDER BY created_at {order} LIMIT ?", (limit,)
                ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            d["steps"] = json.loads(d["steps"] or "[]")
            d["stages"] = json.loads(d.get("stages") or "null")
            out.append(d)
        return out

    def list_job_ids(self, status: str | None = None) -> list[str]:
        """Return persisted job ids, optionally filtered by status."""
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT job_id FROM jobs WHERE status = ?",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT job_id FROM jobs").fetchall()
        return [str(row["job_id"]) for row in rows]

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        cols = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [job_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", vals)

    def set_running(self, job_id: str, step: str) -> None:
        existing = self.get_job(job_id) or {}
        self.update_job(
            job_id,
            status=JobStatus.RUNNING,
            started_at=existing.get("started_at") or datetime.now(timezone.utc).isoformat(),
            current_step=step,
        )

    def set_failed(self, job_id: str, error: str) -> None:
        self.update_job(
            job_id,
            status=JobStatus.FAILED,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=error[:2000],
        )

    def set_completed(self, job_id: str) -> None:
        self.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            finished_at=datetime.now(timezone.utc).isoformat(),
            current_step=None,
        )

    def set_stage_running(self, job_id: str, step: str) -> None:
        stages = self.get_job(job_id, "stages") or {}
        stage = stages.get(step)
        if stage is None:
            return
        now = datetime.now(timezone.utc)
        queued_at = datetime.fromisoformat(stage.get("queued_at") or now.isoformat())
        stage.update(
            status="running",
            started_at=now.isoformat(),
            queue_wait_seconds=max((now - queued_at).total_seconds(), 0),
        )
        self.update_job(job_id, stages=json.dumps(stages))

    def set_stage_completed(self, job_id: str, step: str, duration_seconds: float) -> None:
        stages = self.get_job(job_id, "stages") or {}
        stage = stages.get(step)
        if stage is None:
            return
        stage.update(
            status="completed",
            finished_at=(now := datetime.now(timezone.utc)).isoformat(),
            duration_seconds=duration_seconds,
        )
        stage_ids = list(stages)
        next_index = stage_ids.index(step) + 1
        if next_index < len(stage_ids):
            stages[stage_ids[next_index]]["queued_at"] = now.isoformat()
        self.update_job(job_id, stages=json.dumps(stages))

    def set_stage_failed(self, job_id: str, step: str, error: str) -> None:
        stages = self.get_job(job_id, "stages") or {}
        stage = stages.get(step)
        if stage is None:
            return
        stage.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=error[:2000],
        )
        self.update_job(job_id, stages=json.dumps(stages))

    def fail_running_stage(self, job_id: str, error: str) -> None:
        stages = self.get_job(job_id, "stages") or {}
        for step, stage in stages.items():
            if stage.get("status") == "running":
                self.set_stage_failed(job_id, step, error)
                return

    def recover_interrupted_jobs(self) -> list[str]:
        """Persist failed job and stage state after daemon interruption."""
        interrupted = self.list_job_ids(status=JobStatus.RUNNING)
        for job_id in interrupted:
            error = "Interrupted — server restarted"
            stages = self.get_job(job_id, "stages") or {}
            if stages and not any(
                stage.get("status") in {"running", "failed"}
                for stage in stages.values()
            ):
                self.set_pending(job_id)
            else:
                self.fail_running_stage(job_id, error)
                self.set_failed(job_id, error)
        return interrupted

    def set_validated(self, job_id: str, result: dict) -> None:
        self.update_job(
            job_id,
            status=JobStatus.VALIDATED,
            validation_result=json.dumps(result),
        )

    def set_promoted(self, job_id: str, ref: str) -> None:
        self.update_job(job_id, status=JobStatus.PROMOTED, promotion_ref=ref)

    def set_pending(self, job_id: str) -> None:
        """Re-queue a job back to pending (used for stale-job recovery)."""
        self.update_job(
            job_id,
            status=JobStatus.PENDING,
            current_step=None,
            started_at=None,
            finished_at=None,
            error=None,
        )

    def find_pending_or_running(
        self,
        version: str,
        branch: str | None,
        markets: str | None,
        steps: list[str],
        extra_args: str | None = None,
    ) -> dict | None:
        """Return the first pending/running job matching the same parameters, or None."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM jobs
                   WHERE status IN ('pending', 'running')
                   AND version = ?
                   ORDER BY created_at ASC""",
                (version,),
            ).fetchall()
        for row in rows:
            job = dict(row)
            job["steps"] = json.loads(job["steps"] or "[]")
            if job.get("branch") != branch:
                continue
            if job.get("markets") != markets:
                continue
            if job.get("extra_args") != extra_args:
                continue
            if set(job.get("steps", [])) != set(steps):
                continue
            return job
        return None
