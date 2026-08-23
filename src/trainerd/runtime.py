"""Daemon runtime state and scheduling policy behind one interface."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import ServerConfig, TrainingConfig, load_config
from .lan import LanPreparedProject, LanRepositoryPolicy
from .runner import JobRunner, StageQueuePool
from .storage import JobStatus, JobStore

log = logging.getLogger(__name__)


@dataclass
class ProjectRuntime:
    project: str
    config_path: Path | None
    config: TrainingConfig
    store: JobStore
    runner: JobRunner
    lan_repo_key: str | None = None
    workspace_lock: asyncio.Lock | None = None


class RepositoryCapacityError(RuntimeError):
    """A LAN repository already holds every configured job slot."""


class DaemonRuntime:
    """Own daemon mode, projects, scheduling state, and LAN lifecycle."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Return runtime-owned state to its unconfigured defaults."""
        self.projects: dict[str, ProjectRuntime] = {}
        self.default_project: str | None = None
        self.server_config: ServerConfig | None = None
        self.lan_mode = False
        self.lan_state_dir: Path | None = None
        self.lan_config_path: Path | None = None
        self.max_concurrent_jobs = 1
        self.lan_prepare_lock: asyncio.Lock | None = None
        self.lan_repositories: dict[str, LanRepositoryPolicy] = {}
        self.stage_queues: StageQueuePool | None = None
        self.queue_worker_task: asyncio.Task | None = None
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.queue_wake: asyncio.Event | None = None
        self.queue_poll_interval = 5.0

    @property
    def registry_mode(self) -> bool:
        return bool(self.server_config and self.server_config.registry_mode)

    @property
    def mode(self) -> str:
        return "lan" if self.lan_mode else "registry" if self.registry_mode else "single"

    @property
    def api_key(self) -> str:
        if self.registry_mode and self.server_config:
            return self.server_config.api_key
        default = self.projects.get(self.default_project or "")
        return default.config.api_key if default else ""

    def configure_lan(
        self,
        state_dir: Path,
        max_concurrent_jobs: int,
        stage_queues: StageQueuePool,
        *,
        repositories: dict[str, LanRepositoryPolicy] | None = None,
        config_path: Path | None = None,
    ) -> None:
        """Configure empty dynamic LAN state before accepting requests."""
        self.reset()
        self.lan_mode = True
        self.lan_state_dir = state_dir
        self.lan_config_path = config_path
        self.max_concurrent_jobs = max_concurrent_jobs
        self.lan_prepare_lock = asyncio.Lock()
        self.lan_repositories = dict(repositories or {})
        self.stage_queues = stage_queues

    def lan_config_view(self) -> dict[str, list[dict[str, object]]]:
        """Return inspectable LAN policy without task commands or environment."""
        return {
            "repositories": [
                {
                    "repo": repo_url,
                    "task_source": (
                        "server_config" if policy.tasks is not None else "repository_manifest"
                    ),
                    "tasks": sorted(policy.tasks or {}),
                }
                for repo_url, policy in sorted(self.lan_repositories.items())
            ]
        }

    def configure_projects(
        self,
        config: ServerConfig,
        stage_queues: StageQueuePool,
        *,
        projects: dict[str, ProjectRuntime] | None = None,
    ) -> None:
        """Install a fixed project registry and recover interrupted jobs."""
        self.reset()
        self.server_config = config
        self.default_project = config.default_project
        self.max_concurrent_jobs = config.max_concurrent_jobs
        self.stage_queues = stage_queues
        if projects is None:
            for configured in config.projects.values():
                store = JobStore(configured.config.log_dir / "jobs.db")
                self.projects[configured.project] = ProjectRuntime(
                    configured.project,
                    configured.config_path,
                    configured.config,
                    store,
                    JobRunner(
                        store,
                        configured.config,
                        config_path=configured.config_path,
                        queues=stage_queues,
                    ),
                )
        else:
            self.projects.update(projects)
        self.validate_unique_job_ids()
        for project in self.projects.values():
            self.recover_stale_jobs(project)

    def default(self) -> ProjectRuntime:
        """Return the default project or fail when runtime is unconfigured."""
        project = self.projects.get(self.default_project or "")
        if project is None:
            raise RuntimeError("trainerd has no configured default project")
        return project

    def select(self, project: object = None) -> ProjectRuntime:
        """Select an explicit project, or the default outside registry mode."""
        requested = str(project) if project is not None else ""
        if not requested:
            if self.registry_mode:
                raise ValueError("project is required in trainerd registry mode")
            return self.default()
        selected = self.projects.get(requested)
        if selected is None:
            allowed = ", ".join(sorted(self.projects))
            raise ValueError(f"Unknown project {requested!r}. Allowed projects: {allowed}")
        return selected

    def refresh(self, project: ProjectRuntime) -> TrainingConfig:
        """Reload a file-backed project without allowing identity changes."""
        if project.config_path is None:
            return project.config
        refreshed = load_config(project.config_path)
        if refreshed.project != project.project:
            raise RuntimeError(
                f"Configured project changed from {project.project!r} "
                f"to {refreshed.project!r}"
            )
        project.config = refreshed
        project.runner.update_config(refreshed, config_path=project.config_path)
        return refreshed

    def find_job(self, job_id: str) -> tuple[ProjectRuntime, dict] | None:
        """Find one loaded job and reject duplicate job identifiers."""
        found = [
            (project, job)
            for project in self.projects.values()
            if (job := project.store.get_job(job_id)) is not None
        ]
        if len(found) > 1:
            owners = ", ".join(item[0].project for item in found)
            raise RuntimeError(
                f"Ambiguous job id {job_id!r} exists in projects: {owners}"
            )
        return found[0] if found else None

    def persisted_lan_stores(self) -> list[tuple[str, JobStore, Path]]:
        """Open unloaded LAN job stores read-only for history requests."""
        if not self.lan_mode or self.lan_state_dir is None:
            return []
        loaded_logs = {
            project.config.log_dir.resolve() for project in self.projects.values()
        }
        root = self.lan_state_dir / "jobs"
        if not root.is_dir():
            return []
        return [
            (path.name, JobStore(path / "jobs.db", read_only=True), path)
            for path in root.iterdir()
            if path.is_dir()
            and (path / "jobs.db").is_file()
            and path.resolve() not in loaded_logs
        ]

    def find_persisted_lan_job(
        self, job_id: str
    ) -> tuple[str, JobStore, Path, dict] | None:
        """Find one unloaded persisted LAN job without mutating its database."""
        found = [
            (project, store, log_dir, job)
            for project, store, log_dir in self.persisted_lan_stores()
            if (job := store.get_job(job_id)) is not None
        ]
        if len(found) > 1:
            owners = ", ".join(item[0] for item in found)
            raise RuntimeError(
                f"Ambiguous persisted job id {job_id!r} exists in projects: {owners}"
            )
        return found[0] if found else None

    def validate_unique_job_ids(self) -> None:
        """Reject project stores that share a historical job identifier."""
        owners: dict[str, str] = {}
        for project in self.projects.values():
            for job_id in project.store.list_job_ids():
                prior = owners.setdefault(job_id, project.project)
                if prior != project.project:
                    raise RuntimeError(
                        f"Duplicate historical job id {job_id!r} exists in "
                        f"projects {prior!r} and {project.project!r}"
                    )

    def active_job_ids(
        self, projects: dict[str, ProjectRuntime] | None = None
    ) -> dict[str, set[str]]:
        """Return running and scheduler-reserved job ids by project."""
        selected = projects or self.projects
        active: dict[str, set[str]] = {}
        for project in selected.values():
            running = set(project.store.list_job_ids(status=JobStatus.RUNNING))
            reserved = {
                job_id
                for job_id in self.running_tasks
                if (found := self.find_job(job_id))
                and found[0].project == project.project
            }
            active[project.project] = running | reserved
        return active

    def pending_candidates(self, limit: int | None = None) -> list[tuple[ProjectRuntime, dict]]:
        """Return globally oldest jobs that current capacity can claim."""
        maximum = self.max_concurrent_jobs if limit is None else limit
        active = self.active_job_ids()
        available = max(maximum - len(set().union(*active.values())), 0)
        candidates: list[tuple[str, ProjectRuntime, dict]] = []
        for project in self.projects.values():
            project_available = max(
                project.config.max_concurrent_jobs - len(active[project.project]), 0
            )
            if project_available <= 0:
                continue
            pending = project.store.list_jobs(
                status=JobStatus.PENDING,
                limit=min(project_available, available),
                oldest_first=True,
            )
            candidates.extend(
                (job.get("created_at") or "", project, job) for job in pending
            )
        return [
            (project, job)
            for _, project, job in sorted(candidates, key=lambda item: item[0])[:available]
        ]

    def pending_order(self) -> list[tuple[ProjectRuntime, dict]]:
        """Order claimable jobs first, then capacity-blocked jobs by age."""
        claimable = self.pending_candidates()
        claimable_ids = {
            (project.project, job["job_id"]) for project, job in claimable
        }
        deferred: list[tuple[str, ProjectRuntime, dict]] = []
        for project in self.projects.values():
            for job in project.store.list_jobs(
                status=JobStatus.PENDING, limit=None, oldest_first=True
            ):
                if (project.project, job["job_id"]) not in claimable_ids:
                    deferred.append((job.get("created_at") or "", project, job))
        deferred.sort(key=lambda item: item[0])
        return claimable + [(project, job) for _, project, job in deferred]

    def recover_stale_jobs(self, project: ProjectRuntime) -> None:
        """Persist interruption state for jobs left running by a restart."""
        for job_id in project.runner.recover_interrupted_jobs():
            status = project.store.get_job(job_id, "status")
            log.warning("Recovered stale running job %s as %s", job_id, status)

    def install_lan(self, prepared: LanPreparedProject) -> ProjectRuntime:
        """Install or refresh a prepared LAN project within repository limits."""
        repository_load = sum(
            self._lan_project_load(project)
            for project in self.projects.values()
            if project.lan_repo_key == prepared.repo_key
        )
        if repository_load >= prepared.config.max_concurrent_jobs:
            raise RepositoryCapacityError

        existing = self.projects.get(prepared.project)
        if existing is not None:
            existing.config = prepared.config
            existing.runner.update_config(prepared.config)
            return existing

        store = JobStore(prepared.config.log_dir / "jobs.db")
        workspace_lock = next(
            (
                project.workspace_lock
                for project in self.projects.values()
                if project.lan_repo_key == prepared.repo_key
                and project.workspace_lock is not None
            ),
            None,
        ) or asyncio.Lock()
        project = ProjectRuntime(
            prepared.project,
            None,
            prepared.config,
            store,
            JobRunner(
                store,
                prepared.config,
                queues=self.stage_queues,
                workspace_lock=workspace_lock,
            ),
            lan_repo_key=prepared.repo_key,
            workspace_lock=workspace_lock,
        )
        self.recover_stale_jobs(project)
        if repository_load + self._lan_project_load(project) >= prepared.config.max_concurrent_jobs:
            raise RepositoryCapacityError
        for job_id in store.list_job_ids():
            if self.find_job(job_id) is not None:
                raise RuntimeError(f"Duplicate historical job id {job_id!r} in LAN state")
        self.projects[prepared.project] = project
        if self.default_project is None:
            self.default_project = prepared.project
        return project

    def _lan_project_load(self, project: ProjectRuntime) -> int:
        pending = set(project.store.list_job_ids(status=JobStatus.PENDING))
        active = self.active_job_ids({project.project: project})[project.project]
        return len(pending | active)

    def wake_queue(self) -> None:
        """Notify the scheduler that work or capacity changed."""
        if self.queue_wake is not None:
            self.queue_wake.set()

    async def wait_for_queue(self) -> None:
        """Wait for a scheduler notification or the recovery poll interval."""
        if self.queue_wake is None:
            await asyncio.sleep(self.queue_poll_interval)
            return
        try:
            await asyncio.wait_for(
                self.queue_wake.wait(), timeout=self.queue_poll_interval
            )
        except asyncio.TimeoutError:
            return
        self.queue_wake.clear()
