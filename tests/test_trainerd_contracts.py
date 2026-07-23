from __future__ import annotations

import ast
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

import trainerd.server as server_mod
from trainerd.contracts import ARTIFACT_MANIFEST_SCHEMA, validate_payload
from trainerd.runner import JobRunner
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
