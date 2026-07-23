from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

import trainerd.server as server
from trainerd.cli import main as trainerd_main
from trainerd.config import (
    ConfiguredProject,
    ServerConfig,
    load_config,
    load_server_config,
)
from trainerd.contracts import validate_payload
from trainerd.runner import JobRunner
from trainerd.storage import JobStore


def _write_project_config(
    root: Path,
    project: str,
    *,
    api_key: str = "shared-key",
) -> Path:
    project_root = root / project
    repo = project_root / "repo"
    logs = project_root / "logs"
    repo.mkdir(parents=True)
    path = project_root / "training.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "project": project,
                "api_key": api_key,
                "repo": {"local_path": str(repo)},
                "log_dir": str(logs),
                "max_concurrent_jobs": 1,
                "steps": [{"id": "train", "cmd": f"run-{project}"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_registry(root: Path, projects: dict[str, Path]) -> Path:
    path = root / "projects.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_project": "alpha",
                "api_key": "${TRAINERD_TEST_API_KEY}",
                "max_concurrent_jobs": 1,
                "projects": {
                    project: {"config": str(config_path)}
                    for project, config_path in projects.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_server_config_preserves_single_project_mode(tmp_path: Path) -> None:
    config_path = _write_project_config(tmp_path, "alpha", api_key="")

    config = load_server_config(single_config_path=config_path)

    assert list(config.projects) == ["alpha"]
    assert config.default_project == "alpha"
    assert config.projects["alpha"].config_path == config_path.resolve()
    assert config.max_concurrent_jobs == 1
    assert config.registry_mode is False


def test_load_server_config_builds_fixed_allowlist(tmp_path: Path, monkeypatch) -> None:
    alpha = _write_project_config(tmp_path, "alpha")
    beta = _write_project_config(tmp_path, "beta")
    registry_path = _write_registry(tmp_path, {"alpha": alpha, "beta": beta})
    monkeypatch.setenv("TRAINERD_TEST_API_KEY", "daemon-secret")

    config = load_server_config(projects_path=registry_path)

    assert sorted(config.projects) == ["alpha", "beta"]
    assert config.default_project == "alpha"
    assert config.api_key == "daemon-secret"
    assert config.max_concurrent_jobs == 1
    assert config.registry_mode is True


def test_load_server_config_fails_closed_without_daemon_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alpha = _write_project_config(tmp_path, "alpha")
    beta = _write_project_config(tmp_path, "beta")
    registry_path = _write_registry(tmp_path, {"alpha": alpha, "beta": beta})
    monkeypatch.delenv("TRAINERD_TEST_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing required environment variable"):
        load_server_config(projects_path=registry_path)


def test_load_server_config_rejects_registry_config_identity_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alpha = _write_project_config(tmp_path, "alpha")
    registry_path = _write_registry(tmp_path, {"alpha": alpha, "beta": alpha})
    monkeypatch.setenv("TRAINERD_TEST_API_KEY", "daemon-secret")

    with pytest.raises(ValueError, match="does not match"):
        load_server_config(projects_path=registry_path)


def test_load_server_config_rejects_shared_queue_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alpha = _write_project_config(tmp_path, "alpha")
    beta = _write_project_config(tmp_path, "beta")
    beta_raw = yaml.safe_load(beta.read_text(encoding="utf-8"))
    beta_raw["log_dir"] = yaml.safe_load(alpha.read_text(encoding="utf-8"))["log_dir"]
    beta.write_text(yaml.safe_dump(beta_raw), encoding="utf-8")
    registry_path = _write_registry(tmp_path, {"alpha": alpha, "beta": beta})
    monkeypatch.setenv("TRAINERD_TEST_API_KEY", "daemon-secret")

    with pytest.raises(ValueError, match="distinct log_dir"):
        load_server_config(projects_path=registry_path)


def test_job_contract_accepts_only_project_id_not_client_paths_or_commands() -> None:
    assert validate_payload({"project": "alpha", "version": "v1"}) == []

    problems = validate_payload(
        {
            "project": "alpha",
            "config_path": "C:/arbitrary/training.yaml",
            "cmd": "arbitrary-command",
        }
    )

    assert "unknown field: config_path" in problems
    assert "unknown field: cmd" in problems


def test_server_routes_jobs_only_to_explicit_allowlisted_project(
    tmp_path: Path,
) -> None:
    alpha_path = _write_project_config(tmp_path, "alpha")
    beta_path = _write_project_config(tmp_path, "beta")
    configured: dict[str, ConfiguredProject] = {}
    runtimes: dict[str, server.ProjectRuntime] = {}
    for project, path in (("alpha", alpha_path), ("beta", beta_path)):
        config = load_config(path)
        item = ConfiguredProject(project, path, config)
        configured[project] = item
        store = JobStore(config.log_dir / "jobs.db")
        runtimes[project] = server.ProjectRuntime(
            project,
            path,
            config,
            store,
            JobRunner(store, config, config_path=path),
        )

    old_state = (
        server._server_config,
        server._projects,
        server._default_project,
        server._store,
        server._runner,
        server._config,
        server._config_path,
        server._running_tasks,
    )
    server._server_config = ServerConfig(
        projects=configured,
        default_project="alpha",
        api_key="shared-key",
        server_port=7860,
        max_concurrent_jobs=1,
        registry_mode=True,
    )
    server._projects = runtimes
    server._default_project = "alpha"
    server._store = runtimes["alpha"].store
    server._runner = runtimes["alpha"].runner
    server._config = runtimes["alpha"].config
    server._config_path = runtimes["alpha"].config_path
    server._running_tasks = {}

    client = TestClient(server.app)
    headers = {"X-API-Key": "shared-key"}
    try:
        missing_project = client.post("/api/jobs", headers=headers, json={"version": "v1"})
        alpha_submit = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v1"},
        )
        beta_submit = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "beta", "version": "v1"},
        )
        alpha_duplicate = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v1"},
        )
        unknown = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "not-allowed", "version": "v3"},
        )
        padded_project = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": " alpha ", "version": "v3"},
        )
        unsafe_branch = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v3", "branch": "other"},
        )
        unsafe_args = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v3", "extra_args": "--output C:/tmp"},
        )
        unsafe_markets = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v3", "markets": "safe;whoami"},
        )
        unsafe_version = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v3;whoami"},
        )
        unsafe_steps = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v3", "steps": ["train;whoami"]},
        )
        unknown_field = client.post(
            "/api/jobs",
            headers=headers,
            json={"project": "alpha", "version": "v3", "config_path": "C:/tmp/config.yaml"},
        )

        assert missing_project.status_code == 400
        assert alpha_submit.status_code == 200
        assert alpha_submit.json()["project"] == "alpha"
        assert beta_submit.status_code == 200
        assert beta_submit.json()["project"] == "beta"
        assert alpha_duplicate.status_code == 409
        assert unknown.status_code == 400
        assert padded_project.status_code == 400
        assert unsafe_branch.status_code == 400
        assert unsafe_args.status_code == 400
        assert unsafe_markets.status_code == 400
        assert unsafe_version.status_code == 400
        assert unsafe_steps.status_code == 400
        assert unknown_field.status_code == 422
        assert runtimes["alpha"].store.get_job(alpha_submit.json()["job_id"])
        assert runtimes["beta"].store.get_job(beta_submit.json()["job_id"])

        listed = client.get("/api/jobs", headers=headers)
        assert {job["project"] for job in listed.json()} == {"alpha", "beta"}

        health = client.get("/api/health")
        assert health.json()["project"] == "alpha"
        assert health.json()["projects"] == ["alpha", "beta"]
        assert health.json()["max_concurrent_jobs"] == 1

        alpha_id = alpha_submit.json()["job_id"]
        beta_id = beta_submit.json()["job_id"]
        runtimes["alpha"].store.update_job(alpha_id, created_at="2026-01-02T00:00:00Z")
        runtimes["beta"].store.update_job(beta_id, created_at="2026-01-01T00:00:00Z")
        candidates = server._pending_candidates(1)
        assert [(runtime.project, job["job_id"]) for runtime, job in candidates] == [
            ("beta", beta_id)
        ]

        # A claimed-but-not-yet-running task reserves beta's per-project slot.
        reserved_task = Mock()
        server._running_tasks[beta_id] = reserved_task  # type: ignore[assignment]
        candidates = server._pending_candidates(2)
        assert [(runtime.project, job["job_id"]) for runtime, job in candidates] == [
            ("alpha", alpha_id)
        ]
        assert client.get("/api/health").json()["queue_capacity"] == 0

        runtimes["alpha"].store.set_completed(alpha_id)
        with patch("trainerd.server.asyncio.create_task") as create_task:
            promoted = client.post(f"/api/jobs/{alpha_id}/promote", headers=headers)
            create_task.call_args.args[0].close()
        assert promoted.json()["project"] == "alpha"

        beta_log = runtimes["beta"].config.log_dir / f"{beta_id}.log"
        beta_log.write_text("private log\n", encoding="utf-8")
        assert client.get(f"/api/jobs/{beta_id}/logs?tail=1").status_code == 401
        assert client.get(f"/api/jobs/{beta_id}/logs?tail=1", headers=headers).status_code == 200

        cancelled = client.delete(f"/api/jobs/{beta_id}", headers=headers)
        assert cancelled.json()["project"] == "beta"
        reserved_task.cancel.assert_called_once_with()
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
            server._running_tasks,
        ) = old_state


def test_runner_refuses_project_identity_change_during_reload(tmp_path: Path) -> None:
    config_path = _write_project_config(tmp_path, "alpha")
    config = load_config(config_path)
    store = JobStore(config.log_dir / "jobs.db")
    store.create_job("job-1", steps=["train"], version="v1")
    runner = JobRunner(store, config, config_path=config_path)
    replacement = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    replacement["project"] = "beta"
    config_path.write_text(yaml.safe_dump(replacement), encoding="utf-8")

    with pytest.raises(ValueError, match="Configured project changed"):
        asyncio.run(runner.run_job("job-1"))

    job = store.get_job("job-1")
    assert job["status"] == "pending"
    assert not (config.log_dir / "job-1.log").exists()


def test_registry_mode_with_one_project_still_requires_explicit_project(
    tmp_path: Path,
) -> None:
    path = _write_project_config(tmp_path, "alpha")
    config = load_config(path)
    store = JobStore(config.log_dir / "jobs.db")
    runtime = server.ProjectRuntime(
        "alpha",
        path,
        config,
        store,
        JobRunner(store, config, config_path=path),
    )
    old_state = (server._server_config, server._projects, server._default_project)
    server._server_config = ServerConfig(
        projects={"alpha": ConfiguredProject("alpha", path, config)},
        default_project="alpha",
        api_key="shared-key",
        server_port=7860,
        max_concurrent_jobs=1,
        registry_mode=True,
    )
    server._projects = {"alpha": runtime}
    server._default_project = "alpha"
    try:
        with pytest.raises(Exception) as error:
            server._select_runtime()
        assert getattr(error.value, "status_code", None) == 400
    finally:
        server._server_config, server._projects, server._default_project = old_state


def test_duplicate_historical_job_ids_are_rejected_at_startup(tmp_path: Path) -> None:
    runtimes: dict[str, server.ProjectRuntime] = {}
    for project in ("alpha", "beta"):
        path = _write_project_config(tmp_path, project)
        config = load_config(path)
        store = JobStore(config.log_dir / "jobs.db")
        store.create_job("same-id", steps=["train"], version="v1")
        runtimes[project] = server.ProjectRuntime(
            project,
            path,
            config,
            store,
            JobRunner(store, config, config_path=path),
        )

    with pytest.raises(RuntimeError, match="Duplicate historical job id"):
        server._validate_unique_job_ids(runtimes)


def test_cli_submit_sends_project_without_paths_or_commands() -> None:
    result = {
        "job_id": "job-123",
        "project": "beta",
        "status": "pending",
        "version": "v1",
        "steps": ["train"],
    }
    with patch("trainerd.cli._request_json", return_value=result) as request:
        rc = trainerd_main(
            [
                "submit",
                "--server-url",
                "http://trainerd.local",
                "--project",
                "beta",
                "--version",
                "v1",
            ]
        )

    assert rc == 0
    payload = request.call_args.args[3]
    assert payload["project"] == "beta"
    assert "config_path" not in payload
    assert "cmd" not in payload


def test_openapi_job_request_forbids_additional_properties() -> None:
    openapi = server.app.openapi()
    schema = openapi["components"]["schemas"]["JobRequest"]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["project"]["title"] == "Project"
    security_schemes = openapi["components"]["securitySchemes"]
    assert security_schemes["APIKeyHeader"]["name"] == "X-API-Key"
    submit = openapi["paths"]["/api/jobs"]["post"]
    assert {"APIKeyHeader": []} in submit["security"]


def test_cli_serve_passes_explicit_registry_and_listener() -> None:
    with patch("trainerd.server.main") as serve:
        rc = trainerd_main(
            [
                "serve",
                "--projects-config",
                "projects.yaml",
                "--host",
                "0.0.0.0",
                "--port",
                "7861",
            ]
        )

    assert rc == 0
    serve.assert_called_once_with(
        host="0.0.0.0",
        port=7861,
        projects_config="projects.yaml",
        config=None,
        lan=False,
        state_dir=None,
        max_concurrent_jobs=None,
    )
