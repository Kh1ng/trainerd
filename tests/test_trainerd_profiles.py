from __future__ import annotations

import json
from unittest.mock import patch

from trainerd.cli import main as trainerd_main
from trainerd.profiles import load_profiles


def test_profile_set_list_and_remove_use_user_owned_file(
    tmp_path, monkeypatch, capsys
) -> None:
    profiles_path = tmp_path / "profiles.json"
    monkeypatch.setenv("TRAINERD_PROFILES_PATH", str(profiles_path))

    assert trainerd_main(
        [
            "profile",
            "set",
            "gpu",
            "--server-url",
            "http://trainerd.local:7860",
            "--project",
            "stamp",
        ]
    ) == 0
    assert load_profiles() == {
        "gpu": {
            "server_url": "http://trainerd.local:7860",
            "project": "stamp",
        }
    }
    assert "api_key" not in profiles_path.read_text(encoding="utf-8")

    assert trainerd_main(["profile", "list"]) == 0
    assert "gpu\thttp://trainerd.local:7860\tstamp" in capsys.readouterr().out
    assert trainerd_main(["profile", "remove", "gpu"]) == 0
    assert load_profiles() == {}


def test_submit_uses_profile_defaults_and_keeps_api_key_in_environment(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAINERD_PROFILES_PATH", str(tmp_path / "profiles.json"))
    monkeypatch.setenv("TRAINERD_API_KEY", "secret")
    assert trainerd_main(
        [
            "profile",
            "set",
            "gpu",
            "--server-url",
            "http://trainerd.local:7860",
            "--project",
            "stamp",
        ]
    ) == 0
    result = {
        "job_id": "job-123",
        "project": "stamp",
        "status": "pending",
        "version": "v1",
        "steps": ["baseline"],
    }

    with patch("trainerd.cli._request_json", return_value=result) as request:
        assert trainerd_main(
            ["submit", "--profile", "gpu", "--steps", "baseline"]
        ) == 0

    assert request.call_args.args[:3] == (
        "POST",
        "http://trainerd.local:7860/api/jobs",
        "secret",
    )
    assert request.call_args.args[3]["project"] == "stamp"


def test_profiles_reject_secret_and_unknown_fields(tmp_path, monkeypatch) -> None:
    profiles_path = tmp_path / "profiles.json"
    monkeypatch.setenv("TRAINERD_PROFILES_PATH", str(profiles_path))
    profiles_path.write_text(
        json.dumps(
            {
                "version": 1,
                "profiles": {
                    "gpu": {
                        "server_url": "http://trainerd.local:7860",
                        "project": "stamp",
                        "api_key": "must-not-be-stored",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert trainerd_main(["profile", "list"]) == 2
    assert trainerd_main(
        [
            "profile",
            "set",
            "bad",
            "--server-url",
            "http://trainerd.local:99999",
            "--project",
            "stamp",
        ]
    ) == 2
