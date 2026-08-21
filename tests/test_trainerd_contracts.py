from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

import trainerd.server as server_mod
from trainerd.contracts import ARTIFACT_MANIFEST_SCHEMA, validate_payload
from trainerd.runner import JobRunner, StageQueuePool
from trainerd.storage import JobStore


def _configure_trainerd_server(tmp_path: Path) -> tuple[TestClient, tuple]:
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

    old_state = (
        server_mod._store,
        server_mod._runner,
        server_mod._config,
        server_mod._config_path,
    )
    server_mod._config_path = config_path
    server_mod._config = server_mod.load_config(config_path)
    server_mod._store = JobStore(tmp_path / "jobs.db")
    server_mod._runner = JobRunner(server_mod._store, server_mod._config, config_path=config_path)
    return TestClient(server_mod.app), old_state


def _restore_trainerd_server(old_state: tuple) -> None:
    (
        server_mod._store,
        server_mod._runner,
        server_mod._config,
        server_mod._config_path,
    ) = old_state


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

    problems = validate_payload({"version": 42, "sport": "soccer"})

    assert any("version: expected string" == problem for problem in problems)
    assert any("unknown field: sport" == problem for problem in problems)


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

        log_path = server_mod._config.log_dir / f"{job_id}.log"
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


def test_trainerd_submit_persists_stage_queue_handoff(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    old_stage_queues = server_mod._stage_queues
    try:
        server_mod._stage_queues = StageQueuePool(gpu=2)
        config_path = server_mod._config_path
        assert config_path is not None
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["steps"] = [
            {"id": "prepare", "cmd": "prepare", "queue": "cpu"},
            {"id": "train", "cmd": "train", "queue": "gpu", "units": 2},
        ]
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        server_mod._config = server_mod.load_config(config_path)
        server_mod._runner.update_config(server_mod._config, config_path=config_path)

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
        server_mod._stage_queues = old_stage_queues
        client.close()
        _restore_trainerd_server(old_state)


def test_trainerd_rejects_stage_request_over_gpu_capacity(tmp_path: Path) -> None:
    client, old_state = _configure_trainerd_server(tmp_path)
    old_stage_queues = server_mod._stage_queues
    try:
        server_mod._stage_queues = StageQueuePool(gpu=1)
        config_path = server_mod._config_path
        assert config_path is not None
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        raw["steps"] = [
            {"id": "train", "cmd": "train", "queue": "gpu", "units": 2},
        ]
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        submitted = client.post("/api/jobs", json={"version": "v10"})

        assert submitted.status_code == 400
        assert submitted.json()["detail"] == "GPU stage requests 2 units but capacity is 1"
        assert server_mod._store.list_jobs() == []
    finally:
        server_mod._stage_queues = old_stage_queues
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
        server_mod._config.work_dir = work_dir
        server_mod._store.create_job(job_id, steps=["train"], version="v9")
        server_mod._store.set_completed(job_id)

        manifest = {
            "run_label": "v9",
            "job_id": job_id,
            "produced_at": "2026-08-21T12:00:00Z",
            "artifacts": [
                {
                    "path": "result.json",
                    "bytes": artifact.stat().st_size,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }
        manifest_path = job_dir / "artifact_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        assert client.get(f"/api/jobs/{job_id}/artifacts").status_code == 503
        server_mod._config.api_key = "secret"
        assert client.get(f"/api/jobs/{job_id}/artifacts").status_code == 401
        headers = {"X-API-Key": "secret"}
        listed = client.get(f"/api/jobs/{job_id}/artifacts", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["artifacts"][0]["download_url"].endswith("/artifacts/0")
        downloaded = client.get(f"/api/jobs/{job_id}/artifacts/0", headers=headers)
        assert downloaded.status_code == 200
        assert downloaded.content == artifact.read_bytes()

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
