from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

import trainerd.server as server
from trainerd.cli import main as trainerd_main
from trainerd.lan import (
    LanConfigError,
    LanPreparedProject,
    load_lan_task,
    normalize_repo_url,
    repo_key,
)


def _write_manifest(repo: Path, *, cwd: str = ".") -> Path:
    path = repo / ".trainerd.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "tasks": {
                    "nfl-train": {
                        "steps": [
                            {
                                "id": "train",
                                "name": "Train",
                                "cmd": 'py -3.12 scripts/train.py --work-dir "{work_dir}"',
                                "cwd": cwd,
                                "timeout_seconds": 14400,
                            }
                        ],
                        "validation": {
                            "cmd": 'py -3.12 scripts/validate.py --work-dir "{work_dir}"',
                            "cwd": ".",
                            "output_is_json": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _prepared(tmp_path: Path) -> LanPreparedProject:
    repo_url = "http://git.local/team/repo.git"
    key = repo_key(normalize_repo_url(repo_url))
    repo_path = tmp_path / "managed-repo"
    repo_path.mkdir()
    manifest_path = _write_manifest(repo_path)
    work_dir = tmp_path / "state" / "work"
    log_dir = tmp_path / "state" / "jobs"
    work_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    config = load_lan_task(
        manifest_path,
        task="nfl-train",
        project=f"lan-{key}-nfl-train",
        repo_url=repo_url,
        repo_path=repo_path,
        branch="main",
        work_dir=work_dir,
        log_dir=log_dir,
    )
    return LanPreparedProject(
        project=config.project,
        repo_url=config.repo.url,
        repo_key=key,
        task="nfl-train",
        repo_path=repo_path,
        manifest_path=manifest_path,
        config=config,
    )


@pytest.mark.parametrize(
    "value",
    [
        "ssh://git@git.local/team/repo.git",
        "git@git.local:team/repo.git",
        "file:///tmp/repo",
        "https://user:secret@git.local/team/repo.git",
        "http://git.local/team/repo.git?token=secret",
        "http://git.local/team/repo.git#main",
    ],
)
def test_lan_repo_accepts_only_anonymous_http_urls(value: str) -> None:
    with pytest.raises(LanConfigError):
        normalize_repo_url(value)

    assert (
        normalize_repo_url("HTTP://GIT.LOCAL:8080/team/repo.git/")
        == "http://git.local:8080/team/repo.git"
    )


def test_lan_manifest_resolves_managed_paths(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    config = prepared.config

    assert config.repo.local_path == str(prepared.repo_path.resolve())
    assert config.steps[0].cwd == str(prepared.repo_path.resolve())
    assert "{work_dir}" in config.steps[0].cmd
    assert config.validation is not None
    assert str(config.work_dir) in config.validation.cmd
    assert config.api_key == ""
    assert config.promotion is None


def test_lan_manifest_rejects_cwd_escape_and_unknown_fields(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _write_manifest(repo, cwd="../outside")
    work = tmp_path / "state" / "work"
    logs = tmp_path / "state" / "logs"
    work.mkdir(parents=True)
    logs.mkdir(parents=True)

    with pytest.raises(LanConfigError, match="cwd must stay within"):
        load_lan_task(
            manifest,
            task="nfl-train",
            project="lan-test",
            repo_url="http://git.local/team/repo.git",
            repo_path=repo,
            branch="main",
            work_dir=work,
            log_dir=logs,
        )

    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["tasks"]["nfl-train"]["command_from_client"] = "no"
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(LanConfigError, match="unknown field"):
        load_lan_task(
            manifest,
            task="nfl-train",
            project="lan-test",
            repo_url="http://git.local/team/repo.git",
            repo_path=repo,
            branch="main",
            work_dir=work,
            log_dir=logs,
        )


def test_lan_manifest_rejects_non_utf8_as_client_error(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = repo / ".trainerd.yaml"
    manifest.write_bytes(b"\xff\xfe")
    work = tmp_path / "state" / "work"
    logs = tmp_path / "state" / "logs"
    work.mkdir(parents=True)
    logs.mkdir(parents=True)

    with pytest.raises(LanConfigError, match="Could not read"):
        load_lan_task(
            manifest,
            task="nfl-train",
            project="lan-test",
            repo_url="http://git.local/team/repo.git",
            repo_path=repo,
            branch="main",
            work_dir=work,
            log_dir=logs,
        )


def test_cli_lan_mode_has_zero_config_listener_defaults() -> None:
    with patch("trainerd.server.main") as serve:
        rc = trainerd_main(["serve", "--lan"])

    assert rc == 0
    serve.assert_called_once_with(
        host="0.0.0.0",
        port=None,
        projects_config=None,
        config=None,
        lan=True,
        state_dir=None,
        max_concurrent_jobs=None,
    )


def test_lan_post_repo_and_task_installs_runtime_and_queues_job(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    old_state = (
        server._server_config,
        server._projects,
        server._default_project,
        server._store,
        server._runner,
        server._config,
        server._config_path,
        server._lan_mode_active,
        server._lan_state_dir,
        server._lan_prepare_lock,
        server._running_tasks,
    )
    server._server_config = None
    server._projects = {}
    server._default_project = None
    server._store = None
    server._runner = None
    server._config = None
    server._config_path = None
    server._lan_mode_active = True
    server._lan_state_dir = tmp_path / "state"
    server._lan_prepare_lock = asyncio.Lock()
    server._running_tasks = {}

    client = TestClient(server.app)
    try:
        with patch("trainerd.server.prepare_lan_project", return_value=prepared) as prepare:
            response = client.post(
                "/api/jobs",
                json={
                    "repo": "http://git.local/team/repo.git",
                    "task": "nfl-train",
                },
            )

        assert response.status_code == 200
        result = response.json()
        assert result["job_id"]
        assert result["project"] == prepared.project
        assert result["steps"] == ["train"]
        assert result["queued"] is True
        prepare.assert_called_once()
        runtime = server._projects[prepared.project]
        assert runtime.store.get_job(result["job_id"])
        assert client.get("/api/health").json()["mode"] == "lan"

        # A runner marks the row completed before its validation subprocess
        # returns. Its task reservation must still block a checkout pull.
        runtime.store.set_completed(result["job_id"])
        server._running_tasks[result["job_id"]] = object()  # type: ignore[assignment]
        with patch("trainerd.server.prepare_lan_project") as prepare_again:
            validating = client.post(
                "/api/jobs",
                json={
                    "repo": "http://git.local/team/repo.git",
                    "task": "nfl-train",
                },
            )
        assert validating.status_code == 409
        prepare_again.assert_not_called()
        server._running_tasks.pop(result["job_id"], None)

        incompatible = client.post(
            "/api/jobs",
            json={
                "repo": "http://git.local/team/repo.git",
                "task": "nfl-train",
                "extra_args": "--arbitrary-command",
            },
        )
        assert incompatible.status_code == 400
    finally:
        client.close()
        (
            server._server_config,
            server._projects,
            server._default_project,
            server._store,
            server._runner,
            server._config,
            server._config_path,
            server._lan_mode_active,
            server._lan_state_dir,
            server._lan_prepare_lock,
            server._running_tasks,
        ) = old_state
