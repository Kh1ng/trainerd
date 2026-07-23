from __future__ import annotations

import json
from pathlib import Path

from trainerd.cli import main as trainerd_main
from trainerd.config import RepoConfig, StepConfig, TrainingConfig, ValidationConfig
from trainerd.lan import LanConfigError, inject_managed_env
from trainerd.managed_env import load_managed_env, set_managed_env


def _config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        project="test",
        repo=RepoConfig("http://git.local/repo.git", "main", str(tmp_path)),
        work_dir=tmp_path,
        steps=[StepConfig("train", "Train", "train", env={"PUBLIC": "yes"})],
        validation=ValidationConfig("validate", env={"CHECK": "yes"}),
        promotion=None,
        api_key="",
        server_port=7860,
        log_dir=tmp_path,
        required_env=("NFL_DATABASE_URL",),
    )


def test_managed_env_cli_persists_without_printing_values(
    tmp_path: Path, capsys
) -> None:
    secret = "postgresql://nfl_app:not-for-output@db/nfl"
    rc = trainerd_main(
        [
            "env",
            "--state-dir",
            str(tmp_path),
            "set",
            "NFL_DATABASE_URL",
            "--value",
            secret,
        ]
    )
    output = capsys.readouterr()

    assert rc == 0
    assert "NFL_DATABASE_URL configured" in output.out
    assert secret not in output.out
    assert secret not in output.err
    assert load_managed_env(tmp_path) == {"NFL_DATABASE_URL": secret}

    rc = trainerd_main(["env", "--state-dir", str(tmp_path), "list"])
    output = capsys.readouterr()
    assert rc == 0
    assert output.out.strip() == "NFL_DATABASE_URL"
    assert secret not in output.out


def test_managed_env_injects_only_declared_names(tmp_path: Path) -> None:
    set_managed_env(tmp_path, "NFL_DATABASE_URL", "postgresql://db/nfl")
    set_managed_env(tmp_path, "UNRELATED_SECRET", "do-not-inject")
    config = _config(tmp_path)

    inject_managed_env(config, tmp_path)

    assert config.steps[0].env == {
        "PUBLIC": "yes",
        "NFL_DATABASE_URL": "postgresql://db/nfl",
    }
    assert config.validation is not None
    assert config.validation.env == {
        "CHECK": "yes",
        "NFL_DATABASE_URL": "postgresql://db/nfl",
    }
    assert "UNRELATED_SECRET" not in config.steps[0].env


def test_managed_env_fails_closed_when_required_name_is_missing(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    try:
        inject_managed_env(config, tmp_path)
    except LanConfigError as exc:
        assert "NFL_DATABASE_URL" in str(exc)
    else:
        raise AssertionError("missing required environment did not fail closed")


def test_managed_env_file_does_not_accept_plain_mapping(tmp_path: Path) -> None:
    (tmp_path / "job-env.json").write_text(
        json.dumps({"NFL_DATABASE_URL": "secret"}),
        encoding="utf-8",
    )
    config = _config(tmp_path)

    try:
        inject_managed_env(config, tmp_path)
    except LanConfigError as exc:
        assert "unsupported schema" in str(exc)
    else:
        raise AssertionError("unsupported managed environment schema was accepted")
