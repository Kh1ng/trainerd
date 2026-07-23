"""Unit tests for trainerd runner template resolution."""
from __future__ import annotations

import os
from pathlib import Path
import sys

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

from trainerd.runner import _resolve_template, _with_repo_pythonpath


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
    env = _with_repo_pythonpath("/repo", {"PYTHONPATH": "/repo:/other"})
    assert env["PYTHONPATH"] == "/repo:/other"
