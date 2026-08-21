"""Unit tests for trainerd runner template resolution."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys

import pytest

_ROOT = Path(__file__).resolve().parent.parent

# Resolve imports for both monorepo layout and post-extraction flat/nested layouts
try:
    import trainerd.runner
except ImportError:
    if (_ROOT / "trainerd").exists():
        sys.path.insert(0, str(_ROOT))
    else:
        # Flat layout where _ROOT is the package directory
        if _ROOT.name == "trainerd":
            sys.path.insert(0, str(_ROOT.parent))
        else:
            import tempfile
            import atexit
            import shutil
            tmp_dir = tempfile.mkdtemp(prefix="trainerd_test_path_")
            symlink_path = Path(tmp_dir) / "trainerd"
            try:
                symlink_path.symlink_to(_ROOT, target_is_directory=True)
                sys.path.insert(0, tmp_dir)
                atexit.register(lambda: shutil.rmtree(tmp_dir, ignore_errors=True))
            except Exception:
                sys.path.insert(0, str(_ROOT.parent))

from trainerd.config import RepoConfig, StepConfig, TrainingConfig
from trainerd.runner import JobRunner, RepoSyncError, _resolve_template, _sync_repo_checkout, _with_repo_pythonpath
from trainerd.storage import JobStore


def test_resolve_markets_flag_populated() -> None:
    """{markets_flag} emits --markets VALUE when markets is non-empty."""
    result = _resolve_template(
        "train --frame data.parquet {markets_flag} --gpu",
        version="v1",
        repo_path="/repo",
        work_dir="/cache",
        branch="main",
        markets="player_to_receive_card",
    )
    assert "--markets player_to_receive_card" in result
    assert "{markets_flag}" not in result


def test_resolve_markets_flag_empty() -> None:
    """{markets_flag} emits empty string when markets is empty."""
    result = _resolve_template(
        "train --frame data.parquet {markets_flag} --gpu",
        version="v1",
        repo_path="/repo",
        work_dir="/cache",
        branch="main",
        markets="",
    )
    assert "--markets" not in result
    assert "{markets_flag}" not in result


def test_resolve_markets_flag_none() -> None:
    """{markets_flag} emits empty string when markets is None/not provided."""
    result = _resolve_template(
        "train --frame data.parquet {markets_flag} --gpu",
        version="v1",
        repo_path="/repo",
        work_dir="/cache",
        branch="main",
        markets="",
    )
    assert "--markets" not in result


def test_resolve_no_trailing_space() -> None:
    """Empty markets_flag should not leave a bare --markets flag."""
    result = _resolve_template(
        "train --frame data.parquet {markets_flag}--gpu",
        version="v1",
        repo_path="/repo",
        work_dir="/cache",
        branch="main",
        markets="",
    )
    # Should be: "train --frame data.parquet --gpu" with no dangling flag
    assert "--markets" not in result
    assert "--gpu" in result


def test_resolve_extra_args_populated() -> None:
    result = _resolve_template(
        "train --frame data.parquet {extra_args} --gpu",
        version="v1",
        repo_path="/repo",
        work_dir="/cache",
        branch="main",
        markets="player_shots",
        extra_args="--shuffle-labels --dedupe",
    )
    assert "--shuffle-labels --dedupe" in result
    assert "{extra_args}" not in result


def test_with_repo_pythonpath_prepends_repo_path() -> None:
    env = _with_repo_pythonpath("/repo/project-a", {"PYTHONPATH": "/shared/lib"})
    assert env["PYTHONPATH"].split(os.pathsep)[0] == "/repo/project-a"
    assert "/shared/lib" in env["PYTHONPATH"].split(os.pathsep)


def test_with_repo_pythonpath_dedupes_repo_path() -> None:
    existing = os.pathsep.join(["/repo", "/other"])
    env = _with_repo_pythonpath("/repo", {"PYTHONPATH": existing})
    assert env["PYTHONPATH"] == existing


def test_runner_exports_managed_artifact_location(tmp_path: Path, monkeypatch) -> None:
    import trainerd.runner as runner_module

    config = TrainingConfig(
        project="test",
        repo=RepoConfig("", "main", str(tmp_path / "repo")),
        work_dir=tmp_path / "work",
        steps=[StepConfig("run", "Run", "run")],
        validation=None,
        promotion=None,
        api_key="",
        server_port=7860,
        log_dir=tmp_path / "logs",
    )
    config.log_dir.mkdir()
    store = JobStore(tmp_path / "jobs.db")
    store.create_job("job-123", steps=["run"], version="v1")
    captured: dict[str, str] = {}

    async def fake_run_cmd(cmd, cwd, logfile, env, timeout, **kwargs):
        captured.update(env)
        return True

    monkeypatch.setattr(runner_module, "_run_cmd", fake_run_cmd)
    asyncio.run(JobRunner(store, config).run_job("job-123"))

    assert captured["TRAINERD_JOB_ID"] == "job-123"
    assert captured["TRAINERD_ARTIFACT_DIR"] == str((config.work_dir / "job-123").resolve())
    assert Path(captured["TRAINERD_ARTIFACT_DIR"]).is_dir()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_sync_repo_checkout_fast_forwards_and_returns_revision(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    worker = tmp_path / "worker"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(origin), str(seed)], check=True, capture_output=True)
    _git(seed, "config", "user.name", "Test")
    _git(seed, "config", "user.email", "test@example.com")
    (seed / "value.txt").write_text("one\n", encoding="utf-8")
    _git(seed, "add", "value.txt")
    _git(seed, "commit", "-m", "first")
    branch = _git(seed, "branch", "--show-current")
    _git(seed, "push", "origin", branch)
    subprocess.run(["git", "clone", str(origin), str(worker)], check=True, capture_output=True)

    (seed / "value.txt").write_text("two\n", encoding="utf-8")
    _git(seed, "commit", "-am", "second")
    _git(seed, "push", "origin", branch)

    messages: list[str] = []
    revision = _sync_repo_checkout(
        RepoConfig(str(origin), branch, str(worker), sync_before_job=True),
        branch=branch,
        log_fn=messages.append,
    )

    assert revision == _git(seed, "rev-parse", "HEAD")
    assert _git(worker, "rev-parse", "HEAD") == revision
    assert (worker / "value.txt").read_text(encoding="utf-8") == "two\n"
    assert any(revision in message for message in messages)


def test_sync_repo_checkout_refuses_tracked_changes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "value.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "value.txt")
    _git(repo, "commit", "-m", "first")
    (repo / "value.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RepoSyncError, match="tracked changes"):
        _sync_repo_checkout(
            RepoConfig("", _git(repo, "branch", "--show-current"), str(repo), True),
            branch=_git(repo, "branch", "--show-current"),
            log_fn=lambda _: None,
        )
