"""Static safety regression for the Windows upgrade launcher.

The PowerShell helper runs only on Windows and can cut over the production
Scheduled Task. These tests pin its safety invariants so a future edit cannot
silently reintroduce the failure modes from #18: killing unrelated Python
processes, overwriting startup logs, cutting over with a busy queue, or
missing the restore-on-failure path.
"""
from __future__ import annotations

from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "trainerd-upgrade.ps1"


def _script_text() -> str:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def test_upgrade_script_refuses_cutover_with_pending_or_running_jobs() -> None:
    text = _script_text()
    assert "pending_jobs" in text
    assert "running_jobs" in text
    assert "Refusing cutover" in text
    assert 'if ($pending -gt 0 -or $running -gt 0)' in text


def test_upgrade_script_installs_versioned_venv_and_probes_alternate_port() -> None:
    text = _script_text()
    assert 'venvs\\$Version' in text
    assert "py -3.12 -m venv" in text
    assert "-ProbePort" in text or "$ProbePort" in text
    assert "isolated state" in text
    assert "127.0.0.1:$ProbePort" in text
    assert "probe-state" in text


def test_upgrade_script_stops_only_the_scheduled_task() -> None:
    text = _script_text()
    assert "Stop-ScheduledTask" in text
    assert "Get-ScheduledTask" in text
    # Must never kill arbitrary processes or stop python by name.
    assert "Stop-Process" in text  # only the candidate probe process
    assert "taskkill" not in text
    assert "Get-Process" not in text


def test_upgrade_script_preserves_prior_action_and_restores_on_failure() -> None:
    text = _script_text()
    assert "$priorAction" in text
    assert "$priorExecute" in text
    assert "$priorArguments" in text
    assert "restoring prior action" in text
    assert "Set-ScheduledTask -TaskName $TaskName -Action $oldAction" in text
    assert "Rollback failed" in text


def test_upgrade_script_appends_versioned_startup_logs() -> None:
    text = _script_text()
    assert "startup-$Version.log" in text
    assert "*>>" in text
    # The old launcher used Start-Process -RedirectStandardOutput which
    # replaced the prior logs. The task launcher body must append instead.
    assert "RedirectStandardOutput" in text  # probe log only
    assert "*>> \"$startupLog\"" in text


def test_upgrade_script_verifies_health_and_version_after_switch() -> None:
    text = _script_text()
    assert 'status -eq "ok"' in text
    assert "version -eq $ExpectedVersion" in text
    assert "Verifying $Version on $HealthUrl" in text
