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
            for col, ctype in [("branch", "TEXT"), ("markets", "TEXT"), ("extra_args", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {ctype}")
                except Exception:
                    pass

    def create_job(
        self,
        job_id: str,
        steps: list[str],
        version: str,
        triggered_by: str = "api",
        branch: str | None = None,
        markets: str | None = None,
        extra_args: str | None = None,
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (job_id, status, version, steps, triggered_by, created_at, branch, markets, extra_args)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, JobStatus.PENDING, version, json.dumps(steps), triggered_by, now, branch, markets, extra_args),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str, field: str | None = None) -> dict | Any | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["steps"] = json.loads(data["steps"] or "[]")
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
            out.append(d)
        return out

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
