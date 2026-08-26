"""MCP tools backed by a running trainerd HTTP server."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field

from . import __version__
from .cli import _headers, _request_json
from .storage import JobStatus

_MAX_LOG_BYTES = 100_000
_JOB_FIELDS = (
    "job_id",
    "project",
    "status",
    "version",
    "steps",
    "created_at",
    "started_at",
    "finished_at",
    "current_step",
    "current_stage",
    "next_stage",
    "queue",
    "queue_position",
    "repo_sha",
    "promotion_ref",
)


def _job_summary(job: dict[str, Any]) -> dict[str, Any]:
    summary = {key: job[key] for key in _JOB_FIELDS if job.get(key) is not None}
    summary["terminal"] = JobStatus.is_terminal(job.get("status"))
    return summary


def _request_bounded_text(
    url: str, api_key: str, limit: int = _MAX_LOG_BYTES
) -> tuple[str, bool]:
    request = urllib.request.Request(url, headers=_headers(api_key), method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read(limit + 1)
    return body[:limit].decode("utf-8", errors="replace"), len(body) > limit


def _http_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> Any:
    try:
        return _request_json(method, url, api_key, payload)
    except urllib.error.HTTPError as exc:
        body = exc.read(65_537)[:65_536].decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        raise ValueError(f"Trainerd returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Could not reach Trainerd: {exc.reason}") from None


def create_server(server_url: str, api_key: str = "") -> MCPServer:
    """Create a stdio MCP server whose tools call one trainerd HTTP server."""
    base = server_url.rstrip("/")
    server = MCPServer(
        "trainerd",
        description="Manage allowlisted jobs on a trainerd daemon.",
        version=__version__,
    )

    @server.tool()
    def list_jobs(
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """List active jobs in queue order and recent jobs."""
        queue = _http_json("GET", f"{base}/api/queue", api_key)
        recent = _http_json(
            "GET", f"{base}/api/jobs?{urllib.parse.urlencode({'limit': limit})}", api_key
        )
        active = [_job_summary(job) for job in queue.get("jobs", [])]
        active_ids = {job.get("job_id") for job in active}
        return {
            "active": active,
            "recent": [
                _job_summary(job)
                for job in recent
                if job.get("job_id") not in active_ids
            ],
            **{
                key: queue.get(key)
                for key in (
                    "pending_jobs",
                    "running_jobs",
                    "queue_capacity",
                    "max_concurrent_jobs",
                    "stage_queues",
                )
                if key in queue
            },
        }

    @server.tool()
    def get_job(job_id: str) -> dict[str, Any]:
        """Get one job's status and active queue position."""
        path_id = urllib.parse.quote(job_id, safe="")
        job = _job_summary(_http_json("GET", f"{base}/api/jobs/{path_id}", api_key))
        queue = _http_json("GET", f"{base}/api/queue", api_key)
        active = next(
            (entry for entry in queue.get("jobs", []) if entry.get("job_id") == job_id),
            None,
        )
        if active:
            job.update(_job_summary(active))
        job.setdefault("queue_position", None)
        return job

    @server.tool()
    def tail_job_logs(
        job_id: str,
        lines: Annotated[int, Field(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        """Read 1 to 500 final log lines, capped at 100,000 bytes."""
        path_id = urllib.parse.quote(job_id, safe="")
        query = urllib.parse.urlencode({"tail": lines})
        text, truncated = _request_bounded_text(
            f"{base}/api/jobs/{path_id}/logs?{query}", api_key, _MAX_LOG_BYTES
        )
        return {
            "job_id": job_id,
            "lines": lines,
            "text": text,
            "truncated": truncated,
        }

    @server.tool()
    def list_job_artifacts(job_id: str) -> dict[str, Any]:
        """List validated artifact paths, sizes, and SHA-256 hashes."""
        path_id = urllib.parse.quote(job_id, safe="")
        manifest = _http_json(
            "GET", f"{base}/api/jobs/{path_id}/artifacts", api_key
        )
        return {
            key: manifest[key]
            for key in ("job_id", "run_label", "produced_at")
            if key in manifest
        } | {
            "artifacts": [
                {
                    key: artifact[key]
                    for key in ("path", "sha256", "bytes")
                    if key in artifact
                }
                for artifact in manifest.get("artifacts", [])
            ]
        }

    @server.tool()
    def submit_job(
        project: str | None = None,
        repo: str | None = None,
        task: str | None = None,
        steps: list[str] | None = None,
        version: str | None = None,
        branch: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Submit an allowlisted LAN repo task or registry project step set."""
        if repo is not None or task is not None:
            if not repo or not task or project is not None:
                raise ValueError("LAN submission requires repo and task, without project")
        elif not project:
            raise ValueError("Registry submission requires project")
        payload = {
            key: value
            for key, value in {
                "project": project,
                "repo": repo,
                "task": task,
                "steps": steps,
                "version": version,
                "branch": branch,
                "force": force,
                "triggered_by": "mcp",
            }.items()
            if value is not None
        }
        result = _http_json("POST", f"{base}/api/jobs", api_key, payload)
        return {**result, "terminal": JobStatus.is_terminal(result.get("status"))}

    @server.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel one pending or running job."""
        path_id = urllib.parse.quote(job_id, safe="")
        result = _http_json("DELETE", f"{base}/api/jobs/{path_id}", api_key)
        return {**result, "terminal": JobStatus.is_terminal(result.get("status"))}

    @server.tool()
    def promote_job(job_id: str) -> dict[str, Any]:
        """Promote one eligible completed or validated job."""
        path_id = urllib.parse.quote(job_id, safe="")
        result = _http_json(
            "POST", f"{base}/api/jobs/{path_id}/promote", api_key, {}
        )
        return {**result, "terminal": JobStatus.is_terminal(result.get("status"))}

    return server


def run(server_url: str, api_key: str = "") -> None:
    create_server(server_url, api_key).run()
