from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
    _probe_appendable,
    _probe_writable_dir,
    _require_writable_checkout,
    load_lan_task,
    normalize_branch,
    normalize_repo_url,
    prepare_lan_project,
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
        with patch(
            "trainerd.server.prepare_lan_project", return_value=prepared
        ) as prepare_again:
            validating = client.post(
                "/api/jobs",
                json={
                    "repo": "http://git.local/team/repo.git",
                    "task": "nfl-train",
                },
            )
        assert validating.status_code == 409
        prepare_again.assert_called_once()
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


def test_lan_uses_new_task_limit_and_repository_workspace_locks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("TRAINERD_ALLOWED_REPOS", raising=False)
    prepared = _prepared(tmp_path)
    prepared.config.max_concurrent_jobs = 1
    other_project = f"lan-{prepared.repo_key}-other"
    other = replace(
        prepared,
        project=other_project,
        task="other",
        config=replace(
            prepared.config,
            project=other_project,
            work_dir=tmp_path / "state" / "work" / prepared.repo_key / "other",
            log_dir=tmp_path / "state" / "jobs" / other_project,
            max_concurrent_jobs=2,
        ),
    )
    third_key = "b" * 20
    third_project = f"lan-{third_key}-other"
    third = replace(
        other,
        project=third_project,
        repo_key=third_key,
        config=replace(
            other.config,
            project=third_project,
            work_dir=tmp_path / "state" / "work" / third_key / "other",
            log_dir=tmp_path / "state" / "jobs" / third_project,
        ),
    )
    old_state = (
        server._projects,
        server._default_project,
        server._store,
        server._runner,
        server._config,
        server._config_path,
        server._lan_mode_active,
        server._lan_state_dir,
    )
    server._projects = {}
    server._default_project = None
    server._store = None
    server._runner = None
    server._config = None
    server._config_path = None
    server._lan_mode_active = True
    server._lan_state_dir = tmp_path / "state"
    try:
        first = server._install_lan_runtime(prepared)
        first.store.create_job("existing", ["train"], "v1")
        with patch("trainerd.server.prepare_lan_project", return_value=other):
            second, _ = asyncio.run(
                server._prepare_lan_runtime(
                    {"repo": prepared.repo_url, "task": "other"}
                )
            )
        unrelated = server._install_lan_runtime(third)

        assert second.workspace_lock is first.workspace_lock
        assert unrelated.workspace_lock is not first.workspace_lock
    finally:
        (
            server._projects,
            server._default_project,
            server._store,
            server._runner,
            server._config,
            server._config_path,
            server._lan_mode_active,
            server._lan_state_dir,
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


def test_lan_reads_persisted_jobs_after_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state_dir = tmp_path / "state"
    key = "a" * 20
    task = "nfl-research"
    project = f"lan-{key}-{task}"
    log_dir = state_dir / "jobs" / project
    store = JobStore(log_dir / "jobs.db")
    job = store.create_job(
        "e49c6bca",
        steps=["evaluate"],
        version="v38",
    )
    store.set_completed(job["job_id"])
    (log_dir / "e49c6bca.log").write_text("completed\n", encoding="utf-8")
    artifact_dir = state_dir / "work" / key / task / job["job_id"]
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "result.json"
    artifact.write_bytes(b'{"status":"ok"}\n')
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "run_label": "v38",
                "job_id": job["job_id"],
                "produced_at": "2026-08-23T12:00:00Z",
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAINERD_API_KEY", "secret")
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
        headers = {"X-API-Key": "secret"}
        listed = client.get("/api/jobs/e49c6bca/artifacts", headers=headers)
        assert listed.status_code == 200
        downloaded = client.get("/api/jobs/e49c6bca/artifacts/0", headers=headers)
        assert downloaded.content == artifact.read_bytes()
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


def test_prepare_lan_project_fails_actionably_when_git_metadata_unwritable(
    tmp_path: Path,
) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits are not authoritative on Windows")
    state_dir = tmp_path / "state"
    repo_url = normalize_repo_url("http://git.local/team/repo.git")
    checkout = state_dir / "repos" / repo_key(repo_url)
    checkout.parent.mkdir(parents=True)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "http://git.local/team/repo.git"],
        check=True,
        capture_output=True,
    )
    _write_manifest(checkout)
    (checkout / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "add", ".trainerd.yaml", "tracked.txt"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-m", "one"],
        check=True,
        capture_output=True,
    )

    git_dir = checkout / ".git"
    git_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(LanConfigError) as error:
            prepare_lan_project(state_dir, repo_url, "nfl-train")
    finally:
        git_dir.chmod(stat.S_IRWXU)

    message = str(error.value)
    assert str(checkout) in message
    assert "service identity" in message
    assert "stable Windows account" in message


def test_prepare_lan_project_probes_existing_reflog_writability(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX permission bits are not authoritative on Windows")
    state_dir = tmp_path / "state"
    repo_url = normalize_repo_url("http://git.local/team/repo.git")
    checkout = state_dir / "repos" / repo_key(repo_url)
    checkout.parent.mkdir(parents=True)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "http://git.local/team/repo.git"],
        check=True,
        capture_output=True,
    )
    _write_manifest(checkout)
    (checkout / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "add", ".trainerd.yaml", "tracked.txt"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-m", "one"],
        check=True,
        capture_output=True,
    )

    # The issue's exact failure mode: a writable .git root containing an
    # unwritable existing reflog. The root probe passes; the reflog probe must
    # still catch it before git fetch fails mid-job.
    git_dir = checkout / ".git"
    reflog = git_dir / "logs" / "refs" / "remotes" / "origin" / "main"
    reflog.parent.mkdir(parents=True)
    reflog.write_text("", encoding="utf-8")
    reflog.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        with pytest.raises(LanConfigError) as error:
            prepare_lan_project(state_dir, repo_url, "nfl-train")
    finally:
        reflog.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    message = str(error.value)
    assert "Reflog is not writable" in message
    assert "logs/refs/remotes" in message
    assert "service identity" in message


def test_writable_directory_probe_is_unique_and_cleanup_tolerant(
    tmp_path: Path,
) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(_probe_writable_dir, [tmp_path] * 32))
    assert list(tmp_path.iterdir()) == []

    class FailedCleanup:
        def close(self) -> None:
            raise OSError("cleanup failed")

    with patch("trainerd.lan.tempfile.TemporaryFile", return_value=FailedCleanup()):
        assert _probe_writable_dir(tmp_path)


def test_reflog_probe_ignores_disappearance_and_windows_sharing(
    tmp_path: Path,
) -> None:
    with patch(
        "trainerd.lan.os.open",
        side_effect=FileNotFoundError(2, "No such file or directory"),
    ):
        assert _probe_appendable(tmp_path / "gone")

    sharing_violation = OSError("sharing violation")
    sharing_violation.winerror = 32  # type: ignore[attr-defined]
    with patch("trainerd.lan.os.open", side_effect=sharing_violation):
        assert _probe_appendable(tmp_path / "busy")

    with patch("trainerd.lan.os.open", return_value=123), patch(
        "trainerd.lan.os.close"
    ) as close:
        assert _probe_appendable(tmp_path / "reflog")
        close.assert_called_once_with(123)


def test_writable_checkout_handles_unknown_service_user(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    with patch("trainerd.lan._probe_writable_dir", return_value=False), patch(
        "trainerd.lan.getpass.getuser", side_effect=KeyError
    ):
        with pytest.raises(LanConfigError, match=r"service identity '(uid \d+|unknown)'"):
            _require_writable_checkout(checkout)
