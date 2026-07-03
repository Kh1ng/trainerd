"""Config loading for training server."""
from __future__ import annotations

import os
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


def _resolve(value: str, work_dir: Path, repo_path: str) -> str:
    return (
        value
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
    repo = RepoConfig(
        url=repo_raw.get("url", ""),
        branch=repo_raw.get("branch", "main"),
        local_path=repo_raw.get("local_path", str(Path.home() / "repos" / raw.get("project", "project"))),
    )

    work_dir = Path(raw.get("work_dir", str(Path.home() / "training-work")))
    api_key = raw.get("api_key", "")
    # Expand env vars in api_key
    if api_key.startswith("${") and api_key.endswith("}"):
        api_key = os.environ.get(api_key[2:-1], "")

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

    log_dir = Path(raw.get("log_dir", str(work_dir / "training-server-logs")))
    log_dir.mkdir(parents=True, exist_ok=True)

    return TrainingConfig(
        project=raw.get("project", "unknown"),
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
