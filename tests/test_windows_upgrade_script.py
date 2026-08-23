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


def test_upgrade_script_stops_only_a_verified_trainerd_listener() -> None:
    text = _script_text()
    assert "Stop-ScheduledTask" in text
    assert "Get-ScheduledTask" in text
    assert "Get-NetTCPConnection" in text
    assert "Get-CimInstance Win32_Process" in text
    assert "$process.CommandLine" in text
    assert "$health.version -ne $ExpectedVersion" in text
    assert "$health.pending_jobs -gt 0 -or $health.running_jobs -gt 0" in text
    assert "Stop-Process -Id $listenerPid" in text
    # Must never kill a process by executable name.
    assert "taskkill" not in text
    assert "Get-Process" not in text


def test_upgrade_script_requires_explicit_live_and_probe_arguments() -> None:
    text = _script_text()
    assert "[string]$ServeArguments" in text
    assert "[string]$ProbeArguments" in text
    assert '-ArgumentList "-m trainerd serve $probeArgumentsResolved"' in text
    assert 'serve $ServeArguments *>> "$startupLog"' in text
    assert 'ProbeArguments must contain {probe_port}' in text
    assert 'LAN ProbeArguments must contain {probe_state_dir}' in text


def test_upgrade_script_compares_live_and_candidate_policy() -> None:
    text = _script_text()
    assert "function Assert-SamePolicy" in text
    for field in (
        "mode",
        "authentication_required",
        "allowed_repository_count",
        "projects",
    ):
        assert f'"{field}"' in text
    assert "Assert-SamePolicy -Expected $live -Candidate $candidateHealth" in text
    assert "Assert-SamePolicy -Expected $live -Candidate $installedHealth" in text


def test_upgrade_script_builds_launcher_before_stopping_task() -> None:
    text = _script_text()
    launcher_index = text.index("Out-File -FilePath $launcherPath")
    stop_index = text.index("Stop-DaemonTask -ExpectedVersion $priorVersion")
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
    restore = text.split("function Restore-PriorAction", 1)[1].split("# 1.", 1)[0]
    assert "$priorVersion" in text
    assert "ExpectedVersion $priorVersion" in text
    assert "did not return to prior version" in text
    assert "Stop-DaemonTask -ExpectedVersion $ExpectedVersion" in restore


def test_upgrade_script_restarts_prior_task_if_action_switch_fails() -> None:
    text = _script_text()
    failure_branch = text.split("if (-not $taskActionChanged) {", 1)[1]

    assert "Start-ScheduledTask -TaskName $TaskName" in failure_branch
    assert "Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $priorVersion" in failure_branch


def test_upgrade_script_preserves_caller_quoting_for_probe_state() -> None:
    text = _script_text()

    assert '.Replace("{probe_state_dir}", "$probeDir")' in text
    template = '--state-dir "{probe_state_dir}"'
    assert template.replace("{probe_state_dir}", r"C:\Program Data\trainerd") == (
        '--state-dir "C:\\Program Data\\trainerd"'
    )


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
