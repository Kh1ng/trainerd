from __future__ import annotations

import ast
import asyncio
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import trainerd.server as server_mod
from trainerd.config import (
    ConfiguredProject,
    ServerConfig,
    load_config,
)
from trainerd.contracts import (
    ARTIFACT_MANIFEST_SCHEMA,
    JobRequest,
    is_safe_identifier,
    validate_payload,
)
from trainerd.runner import JobRunner, StageQueuePool
from trainerd.storage import JobStatus, JobStore


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


def _configure_trainerd_server(tmp_path: Path) -> tuple[TestClient, server_mod.ProjectRuntime]:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": "test",
                "repo": {"local_path": str(repo_path)},
                "log_dir": str(tmp_path / "logs"),
                "steps": [{"id": "train", "name": "Train", "cmd": "train-cmd"}],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    store = JobStore(tmp_path / "jobs.db")
    project = server_mod.ProjectRuntime(
        "test",
        config_path,
        config,
        store,
        JobRunner(store, config, config_path=config_path),
    )
    server_mod._runtime.configure_projects(
        ServerConfig(
            projects={"test": ConfiguredProject("test", config_path, config)},
            default_project="test",
            api_key=config.api_key,
            server_port=config.server_port,
            max_concurrent_jobs=config.max_concurrent_jobs,
            registry_mode=False,
        ),
        StageQueuePool(),
        projects={"test": project},
    )
    return TestClient(server_mod.app), project


def _restore_trainerd_server(_project: server_mod.ProjectRuntime) -> None:
    server_mod._runtime.reset()


def test_trainerd_source_is_domain_neutral() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "trainerd"
    forbidden_import_prefixes = ("scripts", "api", "frontend", "registry", "models_staging")
    forbidden_tokens = ("world_cup", "wc_model_scores", "player_", "market_type", "no_vig", "brier")

    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            bad = [name for name in names if name.startswith(forbidden_import_prefixes)]
            assert not bad, f"{path.name} imports repo/domain modules: {bad}"

        lowered = path.read_text(encoding="utf-8").lower()
        hits = [token for token in forbidden_tokens if token in lowered]
        assert not hits, f"{path.name} contains domain vocabulary: {hits}"


def test_trainerd_job_payload_validation_rejects_domain_fields() -> None:
    assert validate_payload({"version": "v42", "steps": ["pull", "train"], "force": True}) == []

    problems = validate_payload({"version": 42, "steps": ["train", 2], "sport": "soccer"})

    assert any("version: expected string" == problem for problem in problems)
    assert any("steps[1]: expected string at index 1" == problem for problem in problems)
    assert any("unknown field: sport" == problem for problem in problems)


def test_job_request_is_the_schema_and_identifier_policy_owner() -> None:
    schema = JobRequest.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "project", "repo", "repo_url", "task", "version", "steps", "branch",
        "markets", "extra_args", "force", "triggered_by",
    }
    assert is_safe_identifier("job.v1-2_ok")
    assert not is_safe_identifier("bad id")


def test_job_status_owns_terminal_state_policy() -> None:
    assert JobStatus.is_terminal(JobStatus.COMPLETED)
    assert JobStatus.is_terminal(JobStatus.VALIDATED)
    assert JobStatus.is_terminal(JobStatus.PROMOTED)
    assert JobStatus.is_terminal(JobStatus.FAILED)
    assert not JobStatus.is_terminal(JobStatus.PENDING)
    assert not JobStatus.is_terminal(JobStatus.RUNNING)


def test_trainerd_artifact_manifest_validation() -> None:
    manifest = {
        "run_label": "v42",
        "job_id": "job-123",
        "produced_at": "2026-07-03T06:00:00Z",
        "metadata": {"promotion_eligible": False, "market": "opaque-to-trainerd"},
        "artifacts": [{"path": "models/cv_v42/model.joblib", "sha256": "ab", "bytes": 1}],
    }
    assert validate_payload(manifest, ARTIFACT_MANIFEST_SCHEMA) == []

    problems = validate_payload({"run_label": "v42", "artifacts": [{"path": 1}]}, ARTIFACT_MANIFEST_SCHEMA)

    assert any("missing required field: produced_at" == problem for problem in problems)
    assert any("artifacts[0]: path: expected string" == problem for problem in problems)


def test_trainerd_submit_status_logs_and_cancel_contract(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        submit = client.post("/api/jobs", json={"steps": ["train"], "version": "v9"})
        assert submit.status_code == 200
        payload = submit.json()
        assert payload["queued"] is True
        assert payload["status"] == "pending"
        assert payload["version"] == "v9"
        assert payload["steps"] == ["train"]

        job_id = payload["job_id"]

        status = client.get(f"/api/jobs/{job_id}")
        assert status.status_code == 200
        assert status.json()["job_id"] == job_id
        assert status.json()["status"] == "pending"

        log_path = old_state.config.log_dir / f"{job_id}.log"
        log_path.write_text("line one\nline two\n", encoding="utf-8")
        logs = client.get(f"/api/jobs/{job_id}/logs?tail=1")
        assert logs.status_code == 200
        assert logs.text == "line two\n"

        cancel = client.delete(f"/api/jobs/{job_id}")
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "failed"
        assert "Cancelled via API" in cancel.json()["error"]

        after = client.get(f"/api/jobs/{job_id}")
        assert after.status_code == 200
        assert after.json()["status"] == "failed"
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_run_job_wrapper_propagates_cancellation(tmp_path: Path, monkeypatch) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    job_id = "job-cancelled"
    old_state.store.create_job(job_id, steps=["train"], version="v1")

    async def cancel(_job_id: str) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(old_state.runner, "run_job", cancel)
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(server_mod._run_job_wrapper(job_id))
        assert old_state.store.get_job(job_id, "status") == "failed"
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_trainerd_submit_persists_stage_queue_handoff(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        server_mod._runtime.stage_queues = StageQueuePool(gpu=2)
        config_path = old_state.config_path
        assert config_path is not None
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["steps"] = [
            {"id": "prepare", "cmd": "prepare", "queue": "cpu"},
            {"id": "train", "cmd": "train", "queue": "gpu", "units": 2},
        ]
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        old_state.config = load_config(config_path)
        old_state.runner.update_config(old_state.config, config_path=config_path)

        submitted = client.post(
            "/api/jobs",
            json={"version": "v10", "steps": ["train", "prepare"]},
        )
        job = client.get(f"/api/jobs/{submitted.json()['job_id']}").json()

        assert job["stages"]["prepare"]["queue"] == "cpu"
        assert job["stages"]["prepare"]["queued_at"]
        assert job["stages"]["train"] == {
            "queue": "gpu",
            "units": 2,
            "status": "pending",
        }
        assert job["stage_queues"]["gpu"] == {
            "running": 0,
            "total": 2,
            "limit": 2,
            "claimed": 0,
            "available": 2,
            "occupancy": 0,
        }
        assert client.get("/api/health").json()["stage_queues"]["gpu"]["total"] == 2
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_active_queue_lists_running_then_pending_with_position(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        server_mod._runtime.stage_queues = StageQueuePool(gpu=2)
        config_path = old_state.config_path
        assert config_path is not None
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["steps"] = [
            {"id": "prepare", "cmd": "prepare", "queue": "cpu"},
            {"id": "train", "cmd": "train", "queue": "gpu", "units": 2},
        ]
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        old_state.config = load_config(config_path)
        old_state.runner.update_config(old_state.config, config_path=config_path)

        older = client.post("/api/jobs", json={"version": "v1"}).json()["job_id"]
        newer = client.post("/api/jobs", json={"version": "v2"}).json()["job_id"]
        running = client.post("/api/jobs", json={"version": "v3"}).json()["job_id"]
        old_state.store.set_running(running, step="prepare")

        queue = client.get("/api/queue").json()

        assert queue["running_jobs"] == 1
        assert queue["pending_jobs"] == 2
        assert queue["max_concurrent_jobs"] == 1
        assert queue["queue_capacity"] == 0
        assert queue["stage_queues"]["gpu"]["total"] == 2

        jobs = queue["jobs"]
        assert jobs[0]["job_id"] == running
        assert jobs[0]["status"] == "running"
        assert jobs[0]["current_stage"] == "prepare"
        assert jobs[0]["queue"] == "cpu"
        assert jobs[0]["queue_position"] is None

        pending_ids = [job["job_id"] for job in jobs[1:]]
        assert pending_ids == [older, newer]
        assert [job["queue_position"] for job in jobs[1:]] == [1, 2]
        assert jobs[1]["status"] == "pending"
        assert jobs[1]["next_stage"] == "prepare"
        assert jobs[1]["queue"] == "cpu"
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_active_queue_orders_pending_by_created_at_globally(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        first = client.post("/api/jobs", json={"version": "v1"}).json()["job_id"]
        second = client.post("/api/jobs", json={"version": "v2"}).json()["job_id"]

        queue = client.get("/api/queue").json()

        assert [job["job_id"] for job in queue["jobs"]] == [first, second]
        assert queue["jobs"][0]["queue_position"] == 1
        assert queue["jobs"][0]["created_at"]
        assert queue["jobs"][0]["steps"] == ["train"]
        # Jobs without configured stage queues still report the next step.
        assert queue["jobs"][0]["next_stage"] == "train"
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_active_queue_positions_follow_scheduler_order_under_per_project_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    alpha_path = _write_project_config(tmp_path, "alpha")
    beta_path = _write_project_config(tmp_path, "beta")
    runtimes: dict[str, server_mod.ProjectRuntime] = {}
    for project, path in (("alpha", alpha_path), ("beta", beta_path)):
        config = load_config(path)
        store = JobStore(config.log_dir / "jobs.db")
        runtimes[project] = server_mod.ProjectRuntime(
            project,
            path,
            config,
            store,
            JobRunner(store, config, config_path=path),
        )
    configured = {p: ConfiguredProject(p, r.config_path, r.config) for p, r in runtimes.items()}
    server_mod._runtime.configure_projects(
        ServerConfig(
        projects=configured,
        default_project="alpha",
        api_key="shared-key",
        server_port=7860,
        max_concurrent_jobs=2,
        registry_mode=True,
        ),
        StageQueuePool(),
        projects=runtimes,
    )

    client = TestClient(server_mod.app)
    headers = {"X-API-Key": "shared-key"}
    try:
        # alpha is at its per-project limit (one running job), so its older
        # pending job is not claimable yet. beta's newer pending job is.
        alpha_running = client.post(
            "/api/jobs", headers=headers, json={"project": "alpha", "version": "v1"}
        ).json()["job_id"]
        runtimes["alpha"].store.set_running(alpha_running, step="train")
        alpha_pending = client.post(
            "/api/jobs", headers=headers, json={"project": "alpha", "version": "v2"}
        ).json()["job_id"]
        beta_pending = client.post(
            "/api/jobs", headers=headers, json={"project": "beta", "version": "v1"}
        ).json()["job_id"]
        runtimes["alpha"].store.update_job(
            alpha_pending, created_at="2026-01-01T00:00:00Z"
        )

        queue = client.get("/api/queue", headers=headers).json()

        assert [job["job_id"] for job in queue["jobs"]] == [
            alpha_running,
            beta_pending,
            alpha_pending,
        ]
        assert queue["jobs"][1]["queue_position"] == 1
        assert queue["jobs"][2]["queue_position"] == 2
    finally:
        client.close()
        server_mod._runtime.reset()


def test_active_queue_excludes_dormant_lan_stores(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    project = "lan-dormant-project"
    store = JobStore(state_dir / "jobs" / project / "jobs.db")
    store.create_job("dormant-pending", ["train"], "v1")
    server_mod._runtime.configure_lan(state_dir, 1, StageQueuePool())

    client = TestClient(server_mod.app)
    try:
        queue = client.get("/api/queue").json()
        assert queue["jobs"] == []
        assert queue["pending_jobs"] == 0
        assert queue["running_jobs"] == 0
        assert queue["queue_capacity"] == 1

        # The queue agrees with /api/health instead of the dormant store.
        health = client.get("/api/health").json()
        assert health["pending_jobs"] == 0
        assert health["running_jobs"] == 0
        # The dormant job remains visible on the historical listing.
        assert client.get("/api/jobs").json()[0]["job_id"] == "dormant-pending"
    finally:
        client.close()
        server_mod._runtime.reset()


def test_list_jobs_supports_unbounded_fetch(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    for index in range(1005):
        store.create_job(f"job-{index:04d}", ["train"], f"v{index}")

    assert len(store.list_jobs(limit=None)) == 1005
    assert len(store.list_jobs(limit=None, status="pending")) == 1005
    assert len(store.list_jobs(status="pending")) == 20


def test_active_queue_is_read_only_and_authenticated(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        assert client.post("/api/queue").status_code == 405

        client.post("/api/jobs", json={"version": "v1"})
        body = client.get("/api/queue").json()
        assert "commands" not in body
        assert all("env" not in job for job in body["jobs"])
        assert all("extra_args" not in job for job in body["jobs"])
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_trainerd_rejects_stage_request_over_gpu_capacity(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        server_mod._runtime.stage_queues = StageQueuePool(gpu=1)
        config_path = old_state.config_path
        assert config_path is not None
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["steps"] = [
            {"id": "train", "cmd": "train", "queue": "gpu", "units": 2},
        ]
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        submitted = client.post("/api/jobs", json={"version": "v10"})

        assert submitted.status_code == 400
        assert submitted.json()["detail"] == "GPU stage requests 2 units but capacity is 1"
        assert old_state.store.list_jobs() == []
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_job_artifacts_are_authenticated_and_integrity_checked(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        work_dir = tmp_path / "work"
        job_id = "job-123"
        job_dir = work_dir / job_id
        job_dir.mkdir(parents=True)
        artifact = job_dir / "result.json"
        artifact.write_bytes(b'{"status":"ok"}\n')
        second_artifact = job_dir / "metrics.json"
        second_artifact.write_bytes(b'{"score":1}\n')
        old_state.config.work_dir = work_dir
        old_state.store.create_job(job_id, steps=["train"], version="v9")
        old_state.store.set_completed(job_id)

        manifest = {
            "run_label": "v9",
            "job_id": job_id,
            "produced_at": "2026-08-21T12:00:00Z",
            "artifacts": [
                {
                    "path": "result.json",
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                },
                {
                    "path": "metrics.json",
                    "bytes": second_artifact.stat().st_size,
                    "sha256": hashlib.sha256(second_artifact.read_bytes()).hexdigest(),
                },
            ],
        }
        manifest_path = job_dir / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        assert client.get(f"/api/jobs/{job_id}/artifacts").status_code == 503
        old_state.config.api_key = "secret"
        assert client.get(f"/api/jobs/{job_id}/artifacts").status_code == 401
        headers = {"X-API-Key": "secret"}
        listed = client.get(f"/api/jobs/{job_id}/artifacts", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["artifacts"][0]["download_url"].endswith("/artifacts/0")
        downloaded = client.get(f"/api/jobs/{job_id}/artifacts/0", headers=headers)
        assert downloaded.status_code == 200
        assert downloaded.content == artifact.read_bytes()

        second_artifact.write_bytes(b"tampered")
        assert client.get(
            f"/api/jobs/{job_id}/artifacts/0", headers=headers
        ).status_code == 200
        assert client.get(
            f"/api/jobs/{job_id}/artifacts/1", headers=headers
        ).status_code == 422
        second_artifact.write_bytes(b'{"score":1}\n')

        artifact.write_bytes(b"tampered")
        assert client.get(f"/api/jobs/{job_id}/artifacts/0", headers=headers).status_code == 422

        outside = work_dir / "outside.bin"
        outside.write_bytes(b"outside")
        manifest["artifacts"][0] = {
            "path": "../outside.bin",
            "bytes": outside.stat().st_size,
            "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert client.get(f"/api/jobs/{job_id}/artifacts", headers=headers).status_code == 422
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_trainerd_submit_rejects_invalid_payload_shape(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    try:
        response = client.post("/api/jobs", json={"version": 42, "sport": "soccer"})

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert {error["type"] for error in errors} == {"extra_forbidden", "string_type"}
    finally:
        client.close()
        _restore_trainerd_server(old_state)


def test_example_configs_resolve_correctly() -> None:
    root = Path(__file__).resolve().parent.parent
    examples_dir = root / "examples"
    assert examples_dir.exists(), f"Examples directory not found at {examples_dir}"

    for config_name in ["sleep_job_config.yaml", "python_job_config.yaml"]:
        config_path = examples_dir / config_name
        assert config_path.exists(), f"{config_name} not found"

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        work_dir = data.get("work_dir", "")
        log_dir = data.get("log_dir", "")
        assert "examples/trainerd" not in work_dir, f"Stale examples/trainerd path in work_dir of {config_name}"
        assert "examples/trainerd" not in log_dir, f"Stale examples/trainerd path in log_dir of {config_name}"

        for step in data.get("steps", []):
            cmd = step.get("cmd", "")
            assert "examples/trainerd" not in cmd, f"Stale examples/trainerd path in cmd of {config_name}"
            # Check that the script referenced in the command actually exists in the examples directory
            import re
            match = re.search(r"(\w+\.py)", cmd)
            if match:
                script_name = match.group(1)
                script_path = examples_dir / script_name
                assert script_path.exists(), f"Referenced script {script_name} does not exist at {script_path}"

    manifest_path = examples_dir / "artifact_manifest.example.json"
    assert manifest_path.exists()
    manifest_content = manifest_path.read_text(encoding="utf-8")
    assert "examples/trainerd" not in manifest_content, "Stale examples/trainerd path in artifact_manifest.example.json"
