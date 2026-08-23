"""Static safety regression for the Windows upgrade launcher.

The PowerShell helper runs only on Windows and can cut over the production
Scheduled Task. These tests pin its safety invariants so a future edit cannot
silently reintroduce the failure modes from #18: killing unrelated Python
processes, overwriting startup logs, cutting over with a busy queue, deleting
the live state directory, or skipping the restore-on-failure path.
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
    assert '$queue.Pending -gt 0 -or $queue.Running -gt 0' in text


def test_upgrade_script_rechecks_queue_immediately_before_cutover() -> None:
    text = _script_text()
    # Two queue checks: the initial gate and a recheck immediately before stop.
    assert text.count("Get-QueueState -BaseUrl $HealthUrl") >= 2
    assert "queue gained" in text


def test_upgrade_script_installs_fresh_versioned_venv_and_probes_alternate_port() -> None:
    text = _script_text()
    assert 'venvs\\$Version' in text
    assert "py -3.12 -m venv" in text
    # A stale candidate environment must be rebuilt, never reused.
    assert "Remove-Item -Recurse -Force $venvDir" in text
    assert "-ProbePort" in text or "$ProbePort" in text
    assert "127.0.0.1:$ProbePort" in text
    assert "probe-" in text


def test_upgrade_script_uses_distinct_probe_output_streams() -> None:
    text = _script_text()
    assert "-RedirectStandardOutput $probeOutputLog" in text
    assert "-RedirectStandardError $probeErrorLog" in text
    assert "$probeOutputLog and $probeErrorLog" in text


def test_upgrade_script_probe_state_cannot_touch_live_state() -> None:
    text = _script_text()
    assert "ProbeStateDir must be outside the live StateDir" in text
    assert "[guid]::NewGuid" in text
    # Only the unique probe subdirectory may be recursively deleted.
    assert "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $probeDir" in text
    assert "Remove-Item -Recurse -Force $ProbeStateDir" not in text


def test_upgrade_script_stops_only_the_scheduled_task() -> None:
    text = _script_text()
    assert "Stop-ScheduledTask" in text
    assert "Get-ScheduledTask" in text
    # Must never kill arbitrary processes or stop python by name.
    assert "Stop-Process" in text  # only the candidate probe process
    assert "taskkill" not in text
    assert "Get-Process" not in text


def test_upgrade_script_builds_launcher_before_stopping_task() -> None:
    text = _script_text()
    launcher_index = text.index("Out-File -FilePath $launcherPath")
    stop_index = text.index("Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop")
    assert launcher_index < stop_index


def test_upgrade_script_preserves_prior_action_and_restores_on_failure() -> None:
    text = _script_text()
    assert "$priorAction" in text
    assert "$priorExecute" in text
    assert "$priorArguments" in text
    # One shared rollback function used by both the switch and verify paths.
    assert "function Restore-PriorAction" in text
    assert text.count("Restore-PriorAction -Execute") >= 2
    assert "Rollback failed" in text


def test_upgrade_script_rollback_verifies_prior_version() -> None:
    text = _script_text()
    assert "$priorVersion" in text
    assert "ExpectedVersion $priorVersion" in text
    assert "did not return to prior version" in text


def test_upgrade_script_appends_versioned_startup_logs() -> None:
    text = _script_text()
    assert "startup-$Version.log" in text
    assert "*>>" in text
    # The old launcher used Start-Process -RedirectStandardOutput which
    # replaced the prior logs. The task launcher body must append instead.
    assert "RedirectStandardOutput" in text  # probe log only
    assert '*>> "$startupLog"' in text


def test_upgrade_script_verifies_health_and_version_after_switch() -> None:
    text = _script_text()
    assert 'status -eq "ok"' in text
    assert "version -eq $ExpectedVersion" in text
    assert "Verifying $Version on $HealthUrl" in text


def test_upgrade_script_never_joins_an_http_url() -> None:
    text = _script_text()
    # Join-Path is provider-based and throws on http(s) URLs.
    assert "Join-Path $BaseUrl" not in text
    assert '"$base/api/health"' in text
