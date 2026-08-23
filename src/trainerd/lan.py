"""Self-configuring, explicitly insecure LAN mode.

LAN mode accepts only an HTTP(S) Git repository URL and a named, repository-
owned task.  The daemon owns every checkout and state path; HTTP clients never
provide commands or filesystem paths.
"""
from __future__ import annotations

import copy
import errno
import getpass
import hashlib
import json
import os
import subprocess
import tempfile
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
from .contracts import is_safe_identifier
from .managed_env import ManagedEnvError, load_managed_env, validate_env_name

_MAX_REPO_URL = 2048
_MAX_COMMAND = 16_384
_MAX_STRING = 1024
_MAX_TASKS = 64
_MAX_STEPS = 64
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
    revision: str
    config: TrainingConfig


@dataclass(frozen=True)
class LanRepositoryPolicy:
    """One normalized LAN repository and optional server-owned tasks."""

    repo_url: str
    tasks: dict[str, dict[str, Any]] | None


def load_lan_server_config(path: Path) -> dict[str, LanRepositoryPolicy]:
    """Load and validate the persistent server-owned LAN repository policy."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise LanConfigError(f"Could not read LAN config: {exc}") from exc
    if not isinstance(raw, dict):
        raise LanConfigError("LAN config must be a mapping")
    _only_keys(raw, {"version", "repositories"}, "LAN config")
    if raw.get("version") != 1:
        raise LanConfigError("LAN config version must be 1")
    entries = raw.get("repositories")
    if not isinstance(entries, list) or not entries or len(entries) > 256:
        raise LanConfigError("LAN config requires 1-256 repository entries")

    repositories: dict[str, LanRepositoryPolicy] = {}
    for index, entry in enumerate(entries):
        label = f"LAN config repository {index}"
        if not isinstance(entry, dict):
            raise LanConfigError(f"{label} must be a mapping")
        _only_keys(entry, {"repo", "tasks"}, label)
        repo_url = normalize_repo_url(entry.get("repo"))
        if repo_url in repositories:
            raise LanConfigError(f"Duplicate normalized repository: {repo_url}")
        tasks = entry.get("tasks")
        if tasks is not None:
            if not isinstance(tasks, dict) or not tasks or len(tasks) > _MAX_TASKS:
                raise LanConfigError(
                    f"{label} tasks must be a mapping of 1-{_MAX_TASKS} tasks"
                )
            for task in tasks:
                _load_lan_task_definitions(
                    tasks,
                    task=str(task),
                    project=f"lan-{'0' * 20}-{task}",
                    repo_url=repo_url,
                    repo_path=Path("/trainerd/repo"),
                    branch="main",
                    work_dir=Path("/trainerd/work"),
                    log_dir=Path("/trainerd/logs"),
                    source="server_config",
                )
        repositories[repo_url] = LanRepositoryPolicy(
            repo_url,
            copy.deepcopy(tasks) if tasks is not None else None,
        )
    return repositories


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


def normalize_branch(value: str) -> str:
    """Validate a branch name with Git's own ref rules."""
    if not isinstance(value, str) or not value or len(value) > 255 or value != value.strip():
        raise LanConfigError("branch must contain 1-255 characters without outer whitespace")
    if value == "HEAD":
        raise LanConfigError("branch must name a branch, not HEAD")
    result = _run_git(["git", "check-ref-format", "--branch", value], check=False)
    if result.returncode != 0 or result.stdout.strip() != value:
        raise LanConfigError("branch is not a valid Git branch name")
    return value


def prepare_lan_project(
    state_dir: Path,
    repo_url: str,
    task: str,
    *,
    branch: str | None = None,
    task_definitions: dict[str, dict[str, Any]] | None = None,
) -> LanPreparedProject:
    """Clone/update a managed checkout and load one task from `.trainerd.yaml`."""
    normalized_url = normalize_repo_url(repo_url)
    task = _safe_id(task, "task")
    state_dir = state_dir.expanduser().resolve()
    key = repo_key(normalized_url)
    checkout = state_dir / "repos" / key
    checkout.parent.mkdir(parents=True, exist_ok=True)

    selected_branch = normalize_branch(branch) if branch is not None else None
    if checkout.exists():
        if not (checkout / ".git").is_dir():
            raise LanConfigError(f"Managed checkout is not a Git repository: {checkout}")
        actual_url = _git(checkout, "remote", "get-url", "origin").strip()
        if normalize_repo_url(actual_url) != normalized_url:
            raise LanConfigError("Managed checkout origin does not match requested repo")
        _require_writable_checkout(checkout)
        _require_clean_tracked_checkout(checkout)
        _git(checkout, "fetch", "origin")
        if selected_branch is not None:
            _git(checkout, "checkout", selected_branch)
        selected_branch = _git(checkout, "branch", "--show-current").strip()
        if not selected_branch:
            raise LanConfigError("Managed checkout must be on a named branch")
        _git(checkout, "pull", "--ff-only", "origin", selected_branch)
    else:
        _run_git(["git", "clone", "--origin", "origin", "--", normalized_url, str(checkout)])
        if selected_branch is not None:
            _git(checkout, "checkout", selected_branch)
        selected_branch = _git(checkout, "branch", "--show-current").strip()
        if not selected_branch:
            raise LanConfigError("Cloned repository must be on a named branch")

    manifest = checkout / ".trainerd.yaml"
    revision = _git(checkout, "rev-parse", "HEAD").strip()
    if not revision:
        raise LanConfigError("Managed checkout has no Git revision")
    resolved_manifest = manifest.resolve()

    project = f"lan-{key}-{task}"
    work_dir = state_dir / "work" / key / task
    log_dir = state_dir / "jobs" / project
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if task_definitions is None:
        if not manifest.is_file() or not _within(resolved_manifest, checkout.resolve()):
            raise LanConfigError("Repository root must contain a regular .trainerd.yaml file")
        config = load_lan_task(
            resolved_manifest,
            task=task,
            project=project,
            repo_url=normalized_url,
            repo_path=checkout,
            branch=selected_branch,
            work_dir=work_dir,
            log_dir=log_dir,
        )
    else:
        config = _load_lan_task_definitions(
            task_definitions,
            task=task,
            project=project,
            repo_url=normalized_url,
            repo_path=checkout,
            branch=selected_branch,
            work_dir=work_dir,
            log_dir=log_dir,
            source="server_config",
        )
    inject_managed_env(config, state_dir)
    return LanPreparedProject(
        project=project,
        repo_url=normalized_url,
        repo_key=key,
        task=task,
        repo_path=checkout,
        manifest_path=resolved_manifest,
        revision=revision,
        config=config,
    )


def prepare_persisted_lan_project(
    state_dir: Path,
    project: str,
    repositories: dict[str, LanRepositoryPolicy],
) -> LanPreparedProject:
    """Rebuild one persisted LAN runtime from its managed checkout."""
    prefix = "lan-"
    value = project.removeprefix(prefix)
    key, separator, task = value.partition("-")
    if (
        not project.startswith(prefix)
        or not separator
        or len(key) != 20
        or any(character not in "0123456789abcdef" for character in key)
    ):
        raise LanConfigError(f"Invalid persisted LAN project id: {project}")
    checkout = state_dir.expanduser().resolve() / "repos" / key
    repo_url = normalize_repo_url(_git(checkout, "remote", "get-url", "origin").strip())
    if repo_key(repo_url) != key:
        raise LanConfigError(f"Persisted LAN project {project} does not match its repository")
    if repositories and repo_url not in repositories:
        raise LanConfigError(f"Persisted LAN project {project} is no longer allowlisted")
    policy = repositories.get(repo_url)
    return prepare_lan_project(
        state_dir,
        repo_url,
        task,
        task_definitions=policy.tasks if policy else None,
    )


def inject_managed_env(config: TrainingConfig, state_dir: Path) -> None:
    """Inject only the managed names explicitly requested by a LAN task."""
    try:
        managed_env = load_managed_env(state_dir)
    except ManagedEnvError as exc:
        raise LanConfigError(str(exc)) from exc
    missing_env = [name for name in config.required_env if name not in managed_env]
    if missing_env:
        raise LanConfigError(
            "Task requires managed environment variable(s): "
            + ", ".join(missing_env)
            + ". Configure them once with: trainerd env set NAME --from-env NAME"
        )
    selected_env = {name: managed_env[name] for name in config.required_env}
    for step in config.steps:
        step.env = {**step.env, **selected_env}
    if config.validation is not None:
        config.validation.env = {**config.validation.env, **selected_env}


def load_lan_job_config(
    current: TrainingConfig,
    checkout: Path,
    *,
    branch: str,
    task_definition_hash: str | None,
) -> TrainingConfig:
    """Load a LAN task from its pinned job checkout using daemon-owned paths."""
    normalized_url = normalize_repo_url(current.repo.url)
    key = repo_key(normalized_url)
    project_prefix = f"lan-{key}-"
    if not current.project.startswith(project_prefix):
        raise LanConfigError(f"Invalid managed LAN project id: {current.project}")
    task = current.project.removeprefix(project_prefix)
    base_checkout = Path(current.repo.local_path).resolve()
    state_dir = base_checkout.parent.parent
    if base_checkout != state_dir / "repos" / key:
        raise LanConfigError("Managed LAN checkout is outside the state repository directory")
    if current.lan_task_definition is None:
        config = load_lan_task(
            checkout / ".trainerd.yaml",
            task=task,
            project=current.project,
            repo_url=normalized_url,
            repo_path=checkout,
            branch=branch,
            work_dir=current.work_dir,
            log_dir=current.log_dir,
        )
    else:
        config = _load_lan_task_definitions(
            {task: current.lan_task_definition},
            task=task,
            project=current.project,
            repo_url=normalized_url,
            repo_path=checkout,
            branch=branch,
            work_dir=current.work_dir,
            log_dir=current.log_dir,
            source="server_config",
        )
        if config.lan_task_definition_hash != task_definition_hash:
            raise LanConfigError("Server-owned LAN task definition changed after submission")
    inject_managed_env(config, state_dir)
    return config


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
    return _load_lan_task_definitions(
        tasks,
        task=task,
        project=project,
        repo_url=repo_url,
        repo_path=repo_path,
        branch=branch,
        work_dir=work_dir,
        log_dir=log_dir,
        source="repository_manifest",
    )


def _load_lan_task_definitions(
    tasks: Any,
    *,
    task: str,
    project: str,
    repo_url: str,
    repo_path: Path,
    branch: str,
    work_dir: Path,
    log_dir: Path,
    source: str,
) -> TrainingConfig:
    """Build one bounded LAN task from repository- or server-owned definitions."""
    if not isinstance(tasks, dict) or not tasks:
        raise LanConfigError(f"{source} requires a non-empty tasks mapping")
    for task_id in tasks:
        _safe_id(task_id, "task id")
    if task not in tasks:
        allowed = ", ".join(sorted(str(item) for item in tasks))
        raise LanConfigError(f"Unknown task {task!r}. Available tasks: {allowed}")

    task_raw = tasks[task]
    if not isinstance(task_raw, dict):
        raise LanConfigError(f"Task {task!r} must be a mapping")
    _only_keys(
        task_raw,
        {"steps", "validation", "max_concurrent_jobs", "required_env"},
        f"task {task!r}",
    )
    required_env_raw = task_raw.get("required_env", [])
    if not isinstance(required_env_raw, list) or len(required_env_raw) > 64:
        raise LanConfigError(f"task {task!r} required_env must be a list of at most 64 names")
    required_env: list[str] = []
    for raw_name in required_env_raw:
        try:
            name = validate_env_name(raw_name)
        except ManagedEnvError as exc:
            raise LanConfigError(f"task {task!r} required_env contains an invalid name") from exc
        if name in required_env:
            raise LanConfigError(f"task {task!r} required_env contains duplicate name {name}")
        required_env.append(name)
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
        _only_keys(
            value,
            {"id", "name", "cmd", "cwd", "env", "timeout_seconds", "queue", "units"},
            label,
        )
        step_id = _safe_id(value.get("id"), f"{label} id")
        if step_id in seen_steps:
            raise LanConfigError(f"Duplicate step id: {step_id}")
        seen_steps.add(step_id)
        cmd = _bounded_string(value.get("cmd"), f"{label} cmd", _MAX_COMMAND)
        name = _bounded_string(value.get("name", step_id), f"{label} name", _MAX_STRING)
        cwd = _safe_cwd(value.get("cwd", "."), repo_root, work_root, label)
        env = _safe_env(value.get("env", {}), label)
        timeout = _bounded_int(value.get("timeout_seconds", 7200), 1, 604800, f"{label} timeout_seconds")
        queue = value.get("queue")
        if queue not in (None, "cpu", "gpu"):
            raise LanConfigError(f"{label} queue must be cpu or gpu")
        units = _bounded_int(value.get("units", 1), 1, 64, f"{label} units")
        if queue is None and "units" in value:
            raise LanConfigError(f"{label} units require a queue")
        steps.append(StepConfig(step_id, name, cmd, cwd, env, timeout, queue, units))

    if any(step.queue for step in steps) and not all(step.queue for step in steps):
        raise LanConfigError(f"task {task!r} must assign every step to a queue")

    validation = _load_validation(task_raw.get("validation"), repo_root, work_root, task)
    max_jobs = _bounded_int(
        task_raw.get("max_concurrent_jobs", 1),
        1,
        64,
        f"task {task!r} max_concurrent_jobs",
    )
    task_definition = copy.deepcopy(task_raw) if source == "server_config" else None
    task_definition_hash = (
        hashlib.sha256(
            json.dumps(
                task_definition,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if task_definition is not None
        else None
    )
    return TrainingConfig(
        project=project,
        repo=RepoConfig(repo_url, branch, str(repo_root), sync_before_job=True),
        work_dir=work_root,
        steps=steps,
        validation=validation,
        promotion=None,
        api_key="",
        server_port=7860,
        log_dir=log_dir.resolve(),
        max_concurrent_jobs=max_jobs,
        required_env=tuple(required_env),
        lan_task_source=source,
        lan_task_definition=task_definition,
        lan_task_definition_hash=task_definition_hash,
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
        try:
            validate_env_name(name)
        except ManagedEnvError as exc:
            raise LanConfigError(
                f"{label} contains an invalid environment variable name"
            ) from exc
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


def _require_writable_checkout(checkout: Path) -> None:
    """Fail before sync when the current account cannot write Git metadata.

    A checkout may have reflogs or refs owned by a different Windows account
    after a service-user change. The `.git` root can look writable while an
    existing reflog is not, so probe the nested paths `git fetch` writes
    instead of relying on the root or on permission bits.
    """
    git_dir = checkout / ".git"
    problems: list[str] = []
    if not _probe_writable_dir(git_dir):
        problems.append(f"Git metadata root is not writable: {git_dir}")
    for relative in sorted(_existing_reflogs(git_dir)):
        if not _probe_appendable(git_dir / relative):
            problems.append(f"Reflog is not writable: {relative}")
    for relative in (
        "logs",
        "refs",
        "logs/refs/remotes",
        "refs/remotes",
        "logs/refs/heads",
        "refs/heads",
    ):
        directory = git_dir / relative
        if directory.is_dir() and not _probe_writable_dir(directory):
            problems.append(f"Git metadata directory is not writable: {relative}")
    if problems:
        try:
            service = getpass.getuser()
        except KeyError:
            service = f"uid {os.getuid()}" if hasattr(os, "getuid") else "unknown"
        raise LanConfigError(
            f"Managed checkout Git metadata is not writable by the current "
            f"service identity {service!r}: {checkout} ({'; '.join(problems)}). "
            f"Grant the trainerd service account recursive control of the "
            f"checkout, or keep the state directory owned by one stable "
            f"Windows account, then resubmit."
        )


def _existing_reflogs(git_dir: Path) -> list[Path]:
    """Relative paths of every reflog file under the metadata directory."""
    logs = git_dir / "logs"
    if not logs.is_dir():
        return []
    return [path.relative_to(git_dir) for path in logs.rglob("*") if path.is_file()]


def _probe_appendable(path: Path) -> bool:
    """Check write permission by opening append-only; no content is written."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
    except OSError as exc:
        if exc.errno == errno.ENOENT or getattr(exc, "winerror", None) in {32, 33}:
            return True
        return False
    try:
        return True
    finally:
        os.close(fd)


def _probe_writable_dir(path: Path) -> bool:
    """Check that a new entry can be created in a directory."""
    try:
        probe = Path(tempfile.mkdtemp(prefix=".trainerd-write-probe-", dir=path))
    except OSError:
        return False
    try:
        probe.rmdir()
    except OSError:
        # Creation proved writability; removal can fail independently.
        pass
    return True


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
    if not is_safe_identifier(value):
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
