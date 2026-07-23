"""Self-configuring, explicitly insecure LAN mode.

LAN mode accepts only an HTTP(S) Git repository URL and a named, repository-
owned task.  The daemon owns every checkout and state path; HTTP clients never
provide commands or filesystem paths.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from .config import (
    RepoConfig,
    StepConfig,
    TrainingConfig,
    ValidationConfig,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_REPO_URL = 2048
_MAX_COMMAND = 16_384
_MAX_STRING = 1024
_MAX_STEPS = 64
_MAX_TASKS = 64
_GIT_TIMEOUT_SECONDS = 300


class LanConfigError(ValueError):
    """A bounded LAN request, checkout, or manifest was invalid."""


@dataclass(frozen=True)
class LanPreparedProject:
    project: str
    repo_url: str
    repo_key: str
    task: str
    repo_path: Path
    manifest_path: Path
    config: TrainingConfig


def default_state_dir() -> Path:
    """Return a platform-native state location requiring no operator setup."""
    program_data = os.environ.get("PROGRAMDATA")
    if os.name == "nt" and program_data:
        return Path(program_data) / "trainerd" / "state"
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "trainerd"
    return Path.home() / ".local" / "state" / "trainerd"


def normalize_repo_url(value: str) -> str:
    """Validate an anonymous Git HTTP(S) URL and return a stable form."""
    if not isinstance(value, str):
        raise LanConfigError("repo must be a string")
    value = value.strip()
    if not value or len(value) > _MAX_REPO_URL:
        raise LanConfigError(f"repo must contain 1-{_MAX_REPO_URL} characters")
    if any(char.isspace() or ord(char) < 32 for char in value):
        raise LanConfigError("repo must not contain whitespace or control characters")

    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise LanConfigError("LAN mode accepts only http:// or https:// Git repositories")
    if not parsed.hostname:
        raise LanConfigError("repo must include a host")
    if parsed.username is not None or parsed.password is not None:
        raise LanConfigError("repo URLs containing credentials are not accepted")
    if parsed.query or parsed.fragment:
        raise LanConfigError("repo URLs must not contain a query string or fragment")
    if not parsed.path or parsed.path == "/":
        raise LanConfigError("repo must include a repository path")

    # urlsplit lowercases hostname for us. Preserve an explicit port and path,
    # while avoiding user-controlled URL text in local directory names.
    host = parsed.hostname.lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise LanConfigError("repo contains an invalid port") from exc
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def repo_key(repo_url: str) -> str:
    return hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:20]


def prepare_lan_project(state_dir: Path, repo_url: str, task: str) -> LanPreparedProject:
    """Clone/update a managed checkout and load one task from `.trainerd.yaml`."""
    normalized_url = normalize_repo_url(repo_url)
    task = _safe_id(task, "task")
    state_dir = state_dir.expanduser().resolve()
    key = repo_key(normalized_url)
    checkout = state_dir / "repos" / key
    checkout.parent.mkdir(parents=True, exist_ok=True)

    if checkout.exists():
        if not (checkout / ".git").is_dir():
            raise LanConfigError(f"Managed checkout is not a Git repository: {checkout}")
        actual_url = _git(checkout, "remote", "get-url", "origin").strip()
        if normalize_repo_url(actual_url) != normalized_url:
            raise LanConfigError("Managed checkout origin does not match requested repo")
        _require_clean_tracked_checkout(checkout)
        branch = _git(checkout, "branch", "--show-current").strip()
        if not branch or not _SAFE_ID.fullmatch(branch):
            raise LanConfigError("Managed checkout must be on a simple named branch")
        _git(checkout, "pull", "--ff-only", "origin", branch)
    else:
        _run_git(["git", "clone", "--origin", "origin", "--", normalized_url, str(checkout)])
        branch = _git(checkout, "branch", "--show-current").strip()
        if not branch or not _SAFE_ID.fullmatch(branch):
            raise LanConfigError("Cloned repository must have a simple default branch")

    manifest = checkout / ".trainerd.yaml"
    resolved_manifest = manifest.resolve()
    if not manifest.is_file() or not _within(resolved_manifest, checkout.resolve()):
        raise LanConfigError("Repository root must contain a regular .trainerd.yaml file")

    project = f"lan-{key}-{task}"
    work_dir = state_dir / "work" / key / task
    log_dir = state_dir / "jobs" / project
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config = load_lan_task(
        resolved_manifest,
        task=task,
        project=project,
        repo_url=normalized_url,
        repo_path=checkout,
        branch=branch,
        work_dir=work_dir,
        log_dir=log_dir,
    )
    return LanPreparedProject(
        project=project,
        repo_url=normalized_url,
        repo_key=key,
        task=task,
        repo_path=checkout,
        manifest_path=resolved_manifest,
        config=config,
    )


def load_lan_task(
    manifest_path: Path,
    *,
    task: str,
    project: str,
    repo_url: str,
    repo_path: Path,
    branch: str,
    work_dir: Path,
    log_dir: Path,
) -> TrainingConfig:
    """Parse the strict repository-owned LAN task manifest."""
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise LanConfigError(f"Could not read .trainerd.yaml: {exc}") from exc
    if not isinstance(raw, dict):
        raise LanConfigError(".trainerd.yaml must be a mapping")
    _only_keys(raw, {"version", "tasks"}, ".trainerd.yaml")
    if raw.get("version") != 1:
        raise LanConfigError(".trainerd.yaml version must be 1")
    tasks = raw.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise LanConfigError(".trainerd.yaml requires a non-empty tasks mapping")
    if len(tasks) > _MAX_TASKS:
        raise LanConfigError(f".trainerd.yaml supports at most {_MAX_TASKS} tasks")
    for task_id in tasks:
        _safe_id(task_id, "task id")
    if task not in tasks:
        allowed = ", ".join(sorted(str(item) for item in tasks))
        raise LanConfigError(f"Unknown task {task!r}. Available tasks: {allowed}")

    task_raw = tasks[task]
    if not isinstance(task_raw, dict):
        raise LanConfigError(f"Task {task!r} must be a mapping")
    _only_keys(task_raw, {"steps", "validation", "max_concurrent_jobs"}, f"task {task!r}")
    steps_raw = task_raw.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise LanConfigError(f"Task {task!r} requires a non-empty steps list")
    if len(steps_raw) > _MAX_STEPS:
        raise LanConfigError(f"Task {task!r} supports at most {_MAX_STEPS} steps")

    repo_root = repo_path.resolve()
    work_root = work_dir.resolve()
    steps: list[StepConfig] = []
    seen_steps: set[str] = set()
    for index, value in enumerate(steps_raw):
        label = f"task {task!r} step {index}"
        if not isinstance(value, dict):
            raise LanConfigError(f"{label} must be a mapping")
        _only_keys(value, {"id", "name", "cmd", "cwd", "env", "timeout_seconds"}, label)
        step_id = _safe_id(value.get("id"), f"{label} id")
        if step_id in seen_steps:
            raise LanConfigError(f"Duplicate step id: {step_id}")
        seen_steps.add(step_id)
        cmd = _bounded_string(value.get("cmd"), f"{label} cmd", _MAX_COMMAND)
        name = _bounded_string(value.get("name", step_id), f"{label} name", _MAX_STRING)
        cwd = _safe_cwd(value.get("cwd", "."), repo_root, work_root, label)
        env = _safe_env(value.get("env", {}), label)
        timeout = _bounded_int(value.get("timeout_seconds", 7200), 1, 604800, f"{label} timeout_seconds")
        steps.append(StepConfig(step_id, name, cmd, cwd, env, timeout))

    validation = _load_validation(task_raw.get("validation"), repo_root, work_root, task)
    max_jobs = _bounded_int(
        task_raw.get("max_concurrent_jobs", 1),
        1,
        64,
        f"task {task!r} max_concurrent_jobs",
    )
    return TrainingConfig(
        project=project,
        repo=RepoConfig(repo_url, branch, str(repo_root)),
        work_dir=work_root,
        steps=steps,
        validation=validation,
        promotion=None,
        api_key="",
        server_port=7860,
        log_dir=log_dir.resolve(),
        max_concurrent_jobs=max_jobs,
    )


def _load_validation(
    value: Any,
    repo_root: Path,
    work_root: Path,
    task: str,
) -> ValidationConfig | None:
    if value is None:
        return None
    label = f"task {task!r} validation"
    if not isinstance(value, dict):
        raise LanConfigError(f"{label} must be a mapping")
    _only_keys(
        value,
        {"cmd", "cwd", "env", "success_key", "success_value", "output_is_json"},
        label,
    )
    return ValidationConfig(
        cmd=_resolve_managed_paths(
            _bounded_string(value.get("cmd"), f"{label} cmd", _MAX_COMMAND),
            repo_root,
            work_root,
        ),
        cwd=_safe_cwd(value.get("cwd", "."), repo_root, work_root, label),
        env={
            name: _resolve_managed_paths(env_value, repo_root, work_root)
            for name, env_value in _safe_env(value.get("env", {}), label).items()
        },
        success_key=_bounded_string(value.get("success_key", "status"), f"{label} success_key", 128),
        success_value=_bounded_string(value.get("success_value", "pass"), f"{label} success_value", 128),
        output_is_json=_strict_bool(value.get("output_is_json", True), f"{label} output_is_json"),
    )


def _safe_cwd(value: Any, repo_root: Path, work_root: Path, label: str) -> str:
    text = _bounded_string(value, f"{label} cwd", _MAX_STRING)
    text = _resolve_managed_paths(text, repo_root, work_root)
    if "{" in text or "}" in text:
        raise LanConfigError(f"{label} cwd contains an unsupported placeholder")
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    if not (_within(resolved, repo_root) or _within(resolved, work_root)):
        raise LanConfigError(f"{label} cwd must stay within the managed repo or work directory")
    return str(resolved)


def _resolve_managed_paths(value: str, repo_root: Path, work_root: Path) -> str:
    return value.replace("{repo_path}", str(repo_root)).replace("{work_dir}", str(work_root))


def _safe_env(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise LanConfigError(f"{label} env must be a mapping")
    if len(value) > 64:
        raise LanConfigError(f"{label} env supports at most 64 variables")
    result: dict[str, str] = {}
    for name, raw_value in value.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise LanConfigError(f"{label} contains an invalid environment variable name")
        result[name] = _bounded_string(raw_value, f"{label} env {name}", 4096, allow_empty=True)
    return result


def _require_clean_tracked_checkout(repo_path: Path) -> None:
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = _run_git(["git", "-C", str(repo_path), *args], check=False)
        if result.returncode == 1:
            raise LanConfigError(
                "Managed checkout has tracked changes; finish or remove them before updating"
            )
        if result.returncode != 0:
            raise LanConfigError(f"Could not inspect managed checkout: {result.stderr.strip()}")


def _git(repo_path: Path, *args: str) -> str:
    return _run_git(["git", "-C", str(repo_path), *args]).stdout


def _run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LanConfigError(f"Git operation failed: {exc}") from exc
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise LanConfigError(f"Git operation failed: {detail}")
    return result


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _only_keys(value: dict[Any, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise LanConfigError(f"{label} contains unknown field(s): {', '.join(unknown)}")


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise LanConfigError(f"{label} must be a safe identifier (letters, numbers, ., _, -)")
    return value


def _bounded_string(
    value: Any,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise LanConfigError(f"{label} must be a string")
    if (not value and not allow_empty) or len(value) > maximum or "\x00" in value:
        qualifier = "0" if allow_empty else "1"
        raise LanConfigError(f"{label} must contain {qualifier}-{maximum} characters")
    return value


def _bounded_int(value: Any, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise LanConfigError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise LanConfigError(f"{label} must be a boolean")
    return value
