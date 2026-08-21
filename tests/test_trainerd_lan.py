from __future__ import annotations

import asyncio
import sqlite3
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

import trainerd.server as server
from trainerd.cli import main as trainerd_main
from trainerd.config import StepConfig
from trainerd.lan import (
    LanConfigError,
    LanPreparedProject,
    load_lan_task,
    normalize_branch,
    normalize_repo_url,
    repo_key,
)
from trainerd.storage import JobStore


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
        revision="a" * 40,
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


@pytest.mark.parametrize("value", ["feature/mod4", "release-0.3", "user/topic_1"])
def test_lan_branch_uses_git_ref_rules(value: str) -> None:
    assert normalize_branch(value) == value


@pytest.mark.parametrize("value", ["", " HEAD", "HEAD", "bad..branch", "bad branch"])
def test_lan_branch_rejects_invalid_names(value: str) -> None:
    with pytest.raises(LanConfigError, match="branch"):
        normalize_branch(value)


def test_lan_manifest_resolves_managed_paths(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    config = prepared.config

    assert config.repo.local_path == str(prepared.repo_path.resolve())
    assert config.repo.sync_before_job is True
    assert config.steps[0].cwd == str(prepared.repo_path.resolve())
    assert "{work_dir}" in config.steps[0].cmd
    assert config.validation is not None
    assert str(config.work_dir) in config.validation.cmd
    assert config.api_key == ""
    assert config.promotion is None


def test_lan_manifest_loads_explicit_stage_queues(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    raw["tasks"]["nfl-train"]["steps"] = [
        {"id": "prepare", "cmd": "prepare", "queue": "cpu"},
        {"id": "train", "cmd": "train", "queue": "gpu", "units": 2},
    ]
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_lan_task(
        manifest,
        task="nfl-train",
        project="test",
        repo_url="http://git.local/team/repo.git",
        repo_path=tmp_path,
        branch="main",
        work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs",
    )

    assert [step.queue for step in config.steps] == ["cpu", "gpu"]
    assert [step.units for step in config.steps] == [1, 2]


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


def test_lan_manifest_does_not_cap_named_tasks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = _write_manifest(repo)
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    template = raw["tasks"]["nfl-train"]
    raw["tasks"] = {
        f"task-{index}": template for index in range(100)
    }
    manifest.write_text(yaml.safe_dump(raw), encoding="utf-8")
    work = tmp_path / "state" / "work"
    logs = tmp_path / "state" / "logs"
    work.mkdir(parents=True)
    logs.mkdir(parents=True)

    config = load_lan_task(
        manifest,
        task="task-99",
        project="lan-test",
        repo_url="http://git.local/team/repo.git",
        repo_path=repo,
        branch="main",
        work_dir=work,
        log_dir=logs,
    )

    assert config.steps[0].id == "train"


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
        cpu_concurrency=None,
        gpu_capacity=None,
        allowed_repos=None,
    )


def test_cli_passes_gpu_capacity() -> None:
    with patch("trainerd.server.main") as serve:
        rc = trainerd_main(["serve", "--lan", "--gpu-capacity", "3"])

    assert rc == 0
    assert serve.call_args.kwargs["gpu_capacity"] == 3


def test_cli_lan_mode_passes_repository_allowlist() -> None:
    with patch("trainerd.server.main") as serve:
        rc = trainerd_main(
            [
                "serve",
                "--lan",
                "--allow-repo",
                "https://git.local/team/repo.git",
            ]
        )

    assert rc == 0
    assert serve.call_args.kwargs["allowed_repos"] == [
        "https://git.local/team/repo.git"
    ]


def test_lan_allowlist_requires_api_key_and_normalizes(monkeypatch) -> None:
    monkeypatch.setenv(
        "TRAINERD_ALLOWED_REPOS",
        "HTTP://GIT.LOCAL:8080/team/repo.git/",
    )
    monkeypatch.delenv("TRAINERD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TRAINERD_API_KEY is required"):
        server._validate_lan_security()

    monkeypatch.setenv("TRAINERD_API_KEY", "secret")
    assert server._validate_lan_security() == {
        "http://git.local:8080/team/repo.git"
    }


def test_lan_post_repo_and_task_installs_runtime_and_queues_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TRAINERD_API_KEY", raising=False)
    monkeypatch.delenv("TRAINERD_ALLOWED_REPOS", raising=False)
    prepared = _prepared(tmp_path)
    prepared.config.steps.append(StepConfig("evaluate", "Evaluate", "evaluate"))
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
                    "branch": "feature/mod4",
                    "steps": ["train"],
                },
            )

        assert response.status_code == 200
        result = response.json()
        assert result["job_id"]
        assert result["project"] == prepared.project
        assert result["steps"] == ["train"]
        assert result["queued"] is True
        prepare.assert_called_once_with(
            server._lan_state_dir,
            "http://git.local/team/repo.git",
            "nfl-train",
            branch="feature/mod4",
        )
        runtime = server._projects[prepared.project]
        assert runtime.store.get_job(result["job_id"])["branch"] == "feature/mod4"
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

        prepared.config.max_concurrent_jobs = 2
        with patch("trainerd.server.prepare_lan_project", return_value=prepared):
            first = client.post(
                "/api/jobs",
                json={
                    "repo": "http://git.local/team/repo.git",
                    "task": "nfl-train",
                    "version": "v2",
                },
            )
            second = client.post(
                "/api/jobs",
                json={
                    "repo": "http://git.local/team/repo.git",
                    "task": "nfl-train",
                    "version": "v3",
                },
            )
            full = client.post(
                "/api/jobs",
                json={
                    "repo": "http://git.local/team/repo.git",
                    "task": "nfl-train",
                    "version": "v4",
                },
            )
        assert first.status_code == 200
        assert second.status_code == 200
        assert full.status_code == 409
        assert runtime.store.get_job(first.json()["job_id"])["repo_sha"] == prepared.revision

        incompatible = client.post(
            "/api/jobs",
            json={
                "repo": "http://git.local/team/repo.git",
                "task": "nfl-train",
                "extra_args": "--arbitrary-command",
            },
        )
        assert incompatible.status_code == 400

        monkeypatch.setenv("TRAINERD_API_KEY", "secret")
        monkeypatch.setenv(
            "TRAINERD_ALLOWED_REPOS",
            "http://git.local/team/repo.git",
        )
        assert client.get("/api/jobs").status_code == 200
        assert client.get(f"/api/jobs/{result['job_id']}").status_code == 200
        assert client.get(
            f"/api/jobs/{result['job_id']}/logs"
        ).status_code == 200
        assert client.get("/api/models").status_code == 200
        assert client.post(
            "/api/jobs",
            json={
                "repo": "http://git.local/team/repo.git",
                "task": "nfl-train",
            },
        ).status_code == 401
        with patch("trainerd.server.prepare_lan_project") as blocked_prepare:
            blocked = client.post(
                "/api/jobs",
                headers={"X-API-Key": "secret"},
                json={
                    "repo": "http://git.local/other/repo.git",
                    "task": "nfl-train",
                },
            )
        assert blocked.status_code == 403
        blocked_prepare.assert_not_called()
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


def test_lan_lists_historical_jobs_from_read_only_database(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    job_dir = state_dir / "jobs" / "historical-project"
    store = JobStore(job_dir / "jobs.db")
    store.create_job("historical-job", ["train"], "v1")
    database = job_dir / "jobs.db"
    with sqlite3.connect(database) as conn:
        conn.execute("DROP INDEX jobs_created_at")
        conn.execute("DROP INDEX jobs_status_created_at")
        conn.execute("DROP INDEX jobs_version_status_created_at")
    database.chmod(stat.S_IREAD)
    old_state = (
        server._projects,
        server._store,
        server._runner,
        server._config,
        server._config_path,
        server._lan_mode_active,
        server._lan_state_dir,
    )
    server._projects = {}
    server._store = None
    server._runner = None
    server._config = None
    server._config_path = None
    server._lan_mode_active = True
    server._lan_state_dir = state_dir

    client = TestClient(server.app)
    try:
        response = client.get("/api/jobs?limit=1")

        assert response.status_code == 200
        assert response.json()[0]["job_id"] == "historical-job"
    finally:
        client.close()
        database.chmod(stat.S_IWRITE | stat.S_IREAD)
        (
            server._projects,
            server._store,
            server._runner,
            server._config,
            server._config_path,
            server._lan_mode_active,
            server._lan_state_dir,
        ) = old_state


def test_lan_reads_persisted_jobs_after_restart(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project = "lan-persisted-nfl-research"
    log_dir = state_dir / "jobs" / project
    store = JobStore(log_dir / "jobs.db")
    job = store.create_job(
        "e49c6bca",
        steps=["evaluate"],
        version="v38",
    )
    store.set_completed(job["job_id"])
    (log_dir / "e49c6bca.log").write_text("completed\n", encoding="utf-8")
    old_state = (
        server._lan_mode_active,
        server._lan_state_dir,
        server._projects,
        server._default_project,
        server._store,
        server._runner,
        server._config,
        server._config_path,
    )
    server._lan_mode_active = True
    server._lan_state_dir = state_dir
    server._projects = {}
    server._default_project = None
    server._store = None
    server._runner = None
    server._config = None
    server._config_path = None

    client = TestClient(server.app)
    try:
        assert client.get("/api/jobs").json()[0]["job_id"] == "e49c6bca"
        assert client.get("/api/jobs/e49c6bca").json()["status"] == "completed"
        assert client.get("/api/jobs/e49c6bca/logs").text == "completed\n"
    finally:
        client.close()
        (
            server._lan_mode_active,
            server._lan_state_dir,
            server._projects,
            server._default_project,
            server._store,
            server._runner,
            server._config,
            server._config_path,
        ) = old_state
