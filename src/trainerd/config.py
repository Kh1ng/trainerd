"""Config loading for training server."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class RepoConfig:
    url: str
    branch: str
    local_path: str


@dataclass
class StepConfig:
    id: str
    name: str
    cmd: str
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 7200  # 2 hours default


@dataclass
class ValidationConfig:
    cmd: str
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    success_key: str = "status"
    success_value: str = "pass"
    output_is_json: bool = True


@dataclass
class PromotionConfig:
    source_dir: str
    repo_subdir: str
    branch: str
    commit_message: str = "chore: auto-promote {version} [{timestamp}]"
    push: bool = True
    excludes: list[str] = field(default_factory=lambda: ["cv_predictions.csv", "*.log"])


@dataclass
class TrainingConfig:
    project: str
    repo: RepoConfig
    work_dir: Path
    steps: list[StepConfig]
    validation: ValidationConfig | None
    promotion: PromotionConfig | None
    api_key: str
    server_port: int
    log_dir: Path
    max_concurrent_jobs: int = 1


@dataclass(frozen=True)
class ConfiguredProject:
    """One startup-allowlisted project and its server-owned config path."""

    project: str
    config_path: Path
    config: TrainingConfig


@dataclass(frozen=True)
class ServerConfig:
    """Immutable project allowlist selected when trainerd starts."""

    projects: dict[str, ConfiguredProject]
    default_project: str
    api_key: str
    server_port: int
    max_concurrent_jobs: int
    registry_mode: bool


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _expand_env(value: str) -> str:
    """Expand required ${NAME} values and fail instead of guessing host paths."""
    missing = sorted({name for name in _ENV_PATTERN.findall(value) if name not in os.environ})
    if missing:
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")
    return _ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)


def _resolve(value: str, work_dir: Path, repo_path: str) -> str:
    return (
        _expand_env(value)
        .replace("{work_dir}", str(work_dir))
        .replace("{repo_path}", repo_path)
        # NOTE: {version} is intentionally NOT resolved here — the runner
        # substitutes it at job-dispatch time via _resolve_cmd().
    )


def load_config(path: Path) -> TrainingConfig:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text())

    repo_raw = raw.get("repo", {})
    project = _expand_env(str(raw.get("project", "unknown")))
    repo = RepoConfig(
        url=_expand_env(str(repo_raw.get("url", ""))),
        branch=_expand_env(str(repo_raw.get("branch", "main"))),
        local_path=_expand_env(
            str(repo_raw.get("local_path", str(Path.home() / "repos" / project)))
        ),
    )

    work_dir = Path(_expand_env(str(raw.get("work_dir", str(Path.home() / "training-work")))))
    api_key = raw.get("api_key", "")
    # Expand env vars in api_key
    if api_key.startswith("${") and api_key.endswith("}"):
        api_key = os.environ.get(api_key[2:-1], "")
    if raw.get("require_api_key", False) and not api_key:
        raise ValueError("api_key is required by this training config")

    steps = [
        StepConfig(
            id=s["id"],
            name=s.get("name", s["id"]),
            cmd=_resolve(s["cmd"], work_dir, repo.local_path),
            cwd=_resolve(s.get("cwd", repo.local_path), work_dir, repo.local_path),
            env={k: _resolve(v, work_dir, repo.local_path) for k, v in s.get("env", {}).items()},
            timeout_seconds=s.get("timeout_seconds", 7200),
        )
        for s in raw.get("steps", [])
    ]

    val_raw = raw.get("validation")
    validation = None
    if val_raw:
        validation = ValidationConfig(
            cmd=_resolve(val_raw["cmd"], work_dir, repo.local_path),
            cwd=_resolve(val_raw.get("cwd", repo.local_path), work_dir, repo.local_path),
            env={k: _resolve(v, work_dir, repo.local_path) for k, v in val_raw.get("env", {}).items()},
            success_key=val_raw.get("success_key", "status"),
            success_value=val_raw.get("success_value", "pass"),
            output_is_json=val_raw.get("output_is_json", True),
        )

    prom_raw = raw.get("promotion")
    promotion = None
    if prom_raw:
        promotion = PromotionConfig(
            source_dir=_resolve(prom_raw["source_dir"], work_dir, repo.local_path),
            repo_subdir=prom_raw["repo_subdir"],
            branch=prom_raw.get("branch", repo.branch),
            commit_message=prom_raw.get("commit_message", "chore: auto-promote {version} [{timestamp}]"),
            push=prom_raw.get("push", True),
            excludes=prom_raw.get("excludes", ["cv_predictions.csv", "*.log"]),
        )

    log_dir = Path(_expand_env(str(raw.get("log_dir", str(work_dir / "training-server-logs")))))
    log_dir.mkdir(parents=True, exist_ok=True)

    return TrainingConfig(
        project=project,
        repo=repo,
        work_dir=work_dir,
        steps=steps,
        validation=validation,
        promotion=promotion,
        api_key=api_key,
        server_port=raw.get("server", {}).get("port", 7860),
        log_dir=log_dir,
        max_concurrent_jobs=int(raw.get("max_concurrent_jobs", 1)),
    )


def _manifest_path(value: Any, manifest_path: Path) -> Path:
    expanded = Path(_expand_env(str(value)))
    if not expanded.is_absolute():
        expanded = manifest_path.parent / expanded
    return expanded.resolve()


def load_server_config(
    *,
    projects_path: Path | None = None,
    single_config_path: Path | None = None,
) -> ServerConfig:
    """Load the immutable startup project allowlist.

    `TRAINERD_PROJECTS_CONFIG` enables strict registry mode. Without it, the
    legacy `TRAINING_CONFIG` path remains the sole/default project.
    """
    if projects_path is None:
        configured_manifest = os.environ.get("TRAINERD_PROJECTS_CONFIG", "").strip()
        projects_path = Path(configured_manifest) if configured_manifest else None

    if projects_path is None:
        path = single_config_path or Path(os.environ.get("TRAINING_CONFIG", "training_config.yaml"))
        path = path.resolve()
        config = load_config(path)
        project = ConfiguredProject(config.project, path, config)
        return ServerConfig(
            projects={config.project: project},
            default_project=config.project,
            api_key=config.api_key,
            server_port=config.server_port,
            max_concurrent_jobs=config.max_concurrent_jobs,
            registry_mode=False,
        )

    projects_path = projects_path.resolve()
    if not projects_path.exists():
        raise FileNotFoundError(f"Project registry config not found: {projects_path}")
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")

    raw = yaml.safe_load(projects_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("trainerd project registry must be a mapping")
    raw_projects = raw.get("projects")
    if not isinstance(raw_projects, dict) or not raw_projects:
        raise ValueError("trainerd project registry requires a non-empty projects mapping")

    projects: dict[str, ConfiguredProject] = {}
    for project_id, entry in raw_projects.items():
        project_id = str(project_id).strip()
        if not _PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ValueError(f"Invalid trainerd project id: {project_id!r}")
        if project_id in projects:
            raise ValueError(f"Duplicate normalized trainerd project id: {project_id!r}")
        if isinstance(entry, str):
            config_value = entry
        elif isinstance(entry, dict):
            config_value = entry.get("config")
        else:
            config_value = None
        if not config_value:
            raise ValueError(f"Project {project_id!r} requires a config path")
        config_path = _manifest_path(config_value, projects_path)
        config = load_config(config_path)
        if config.project != project_id:
            raise ValueError(
                f"Project registry key {project_id!r} does not match "
                f"{config_path} project {config.project!r}"
            )
        if config.max_concurrent_jobs < 1:
            raise ValueError(
                f"Project {project_id!r} max_concurrent_jobs must be at least 1"
            )
        projects[project_id] = ConfiguredProject(project_id, config_path, config)

    log_dirs = [item.config.log_dir.resolve() for item in projects.values()]
    if len(log_dirs) != len(set(log_dirs)):
        raise ValueError("Each allowlisted project requires a distinct log_dir/jobs.db")

    default_project = str(raw.get("default_project", "")).strip()
    if default_project not in projects:
        raise ValueError("default_project must name one of the allowlisted projects")

    api_key_value = str(raw.get("api_key", "")).strip()
    if not api_key_value:
        raise ValueError("registry-mode trainerd requires one daemon API key")
    api_key = _expand_env(api_key_value)
    if not api_key:
        raise ValueError("registry-mode trainerd requires one daemon API key")

    server_port = int(raw.get("server", {}).get("port", projects[default_project].config.server_port))
    max_jobs = int(raw.get("max_concurrent_jobs", 1))
    if max_jobs < 1:
        raise ValueError("max_concurrent_jobs must be at least 1")

    return ServerConfig(
        projects=projects,
        default_project=default_project,
        api_key=api_key,
        server_port=server_port,
        max_concurrent_jobs=max_jobs,
        registry_mode=True,
    )
