"""Static safety regression for the Windows upgrade helper.

The PowerShell helper runs only on Windows and can cut over the production
Scheduled Task. These tests pin its safety invariants so a future edit cannot
silently reintroduce the failure modes from #18: killing unrelated Python
processes, cutting over with a busy queue, deleting the live state directory,
or skipping the restore-on-failure path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "trainerd-upgrade.ps1"
PREVIOUS_LAN_HEALTH = (
    Path(__file__).resolve().parent / "fixtures" / "health-0.3.13-lan.json"
)


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
    assert "-m pip install --upgrade pip" not in text
    assert "& $pythonExe -m pip install $WheelUrl" in text
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
    assert '-Argument "serve $ServeArguments"' in text
    assert 'ProbeArguments must contain {probe_port}' in text
    assert 'LAN ProbeArguments must contain {probe_state_dir}' in text


def test_upgrade_script_compares_live_and_candidate_policy() -> None:
    text = _script_text()
    assert "function Assert-SamePolicy" in text
    for field in (
        "mode",
        "authentication_required",
        "lan_policy_hash",
    ):
        assert f'"{field}"' in text
    policy_check = text.split("function Assert-SamePolicy", 1)[1].split(
        "function Stop-VerifiedTrainerdListener", 1
    )[0]
    assert '$Expected.mode -eq "lan"' in policy_check
    assert '$fields += "projects"' in policy_check
    assert "-Candidate $candidateHealth" in text
    assert "-Candidate $installedHealth" in text
    assert text.count("-ExpectedLanPolicy $liveLanPolicy") == 2


def test_upgrade_script_verifies_legacy_lan_policy_without_health_hash() -> None:
    live = json.loads(PREVIOUS_LAN_HEALTH.read_text(encoding="utf-8"))
    assert live["mode"] == "lan"
    assert live["authentication_required"] is True
    assert live["allowed_repository_count"] == 2
    assert "lan_policy_hash" not in live

    text = _script_text()
    assert "function Get-LanPolicyJson" in text
    assert "$liveLanPolicy = Get-LanPolicyJson -BaseUrl $HealthUrl" in text
    assert "Could not verify live LAN policy" in text
    assert "$Expected.allowed_repository_count" in text
    assert "-ExpectedLanPolicy $liveLanPolicy" in text
    assert "[string]$LegacyLanConfig" in text
    assert "function Assert-LegacyLanConfig" in text
    assert "--lan-config" in text
    assert "LastWriteTimeUtc" in text
    assert "CreationDate" in text
    assert "policy-hash" in text
    assert text.count("-ExpectedLanPolicyHash $liveLanPolicyHash") == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell regression")
def test_legacy_lan_policy_fallback_runs_in_windows_powershell(tmp_path: Path) -> None:
    text = _script_text()
    functions = text[
        text.index("function Get-LanPolicyJson"):
        text.index("function Stop-VerifiedTrainerdListener")
    ]
    live = json.loads(PREVIOUS_LAN_HEALTH.read_text(encoding="utf-8"))
    candidate = {**live, "version": "0.3.14", "lan_policy_hash": "reviewed-hash"}
    policy = json.dumps(
        {
            "repositories": [
                {
                    "repo": "http://git.local/team/one.git",
                    "task_source": "repository_manifest",
                    "tasks": [],
                },
                {
                    "repo": "http://git.local/team/two.git",
                    "task_source": "server_config",
                    "tasks": ["train"],
                },
            ]
        },
        separators=(",", ":"),
    )
    legacy_config = tmp_path / "legacy lan.yaml"
    legacy_config.write_text("version: 1\nrepositories: []\n", encoding="utf-8")

    def quoted(value: str) -> str:
        return value.replace("'", "''")

    harness = tmp_path / "legacy-policy.ps1"
    harness.write_text(
        functions
        + f"""
$configPath = '{quoted(str(legacy_config))}'
$config = Get-Item -LiteralPath $configPath
$started = $config.LastWriteTimeUtc.AddSeconds(2)
function Get-VerifiedTrainerdListenerProcess {{
    return [pscustomobject]@{{
        CommandLine = 'trainerd.exe serve --lan --lan-config "' + $config.FullName + '"'
        CreationDate = $started
    }}
}}
$boundPath = Assert-LegacyLanConfig -ConfigPath $configPath
if ($boundPath -cne $config.FullName) {{ throw 'Legacy LAN config was not bound.' }}
$config.LastWriteTimeUtc = $started.AddSeconds(2)
$rejected = $false
try {{
    Assert-LegacyLanConfig -ConfigPath $configPath
}}
catch {{
    $rejected = $true
}}
if (-not $rejected) {{ throw 'Changed legacy LAN config was accepted.' }}
function Get-LanPolicyJson {{ param([string]$BaseUrl) return '{quoted(policy)}' }}
$live = '{quoted(json.dumps(live))}' | ConvertFrom-Json
$candidate = '{quoted(json.dumps(candidate))}' | ConvertFrom-Json
Assert-SamePolicy -Expected $live -Candidate $candidate -ExpectedLanPolicy '{quoted(policy)}' -ExpectedLanPolicyHash 'reviewed-hash' -CandidateBaseUrl 'http://candidate'
$changedCandidate = '{quoted(json.dumps({**candidate, "lan_policy_hash": "changed-hash"}))}' | ConvertFrom-Json
$rejected = $false
try {{
    Assert-SamePolicy -Expected $live -Candidate $changedCandidate -ExpectedLanPolicy '{quoted(policy)}' -ExpectedLanPolicyHash 'reviewed-hash' -CandidateBaseUrl 'http://candidate'
}}
catch {{
    $rejected = $true
}}
if (-not $rejected) {{ throw 'Changed complete LAN policy was accepted.' }}
function Get-LanPolicyJson {{ param([string]$BaseUrl) return '{{\"repositories\":[]}}' }}
$rejected = $false
try {{
    Assert-SamePolicy -Expected $live -Candidate $candidate -ExpectedLanPolicy '{quoted(policy)}' -ExpectedLanPolicyHash 'reviewed-hash' -CandidateBaseUrl 'http://candidate'
}}
catch {{
    $rejected = $true
}}
if (-not $rejected) {{ throw 'Mismatched legacy LAN policy was accepted.' }}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_upgrade_script_uses_direct_trainerd_scheduled_task_action() -> None:
    text = _script_text()
    action_index = text.index("New-ScheduledTaskAction -Execute $trainerdExe")
    stop_index = text.index("Stop-DaemonTask -ExpectedVersion $priorVersion")
    assert action_index < stop_index
    assert '-Argument "serve $ServeArguments"' in text
    assert "$launcherPath" not in text
    assert '-Execute "powershell.exe"' not in text


def test_upgrade_script_preserves_prior_action_and_restores_on_failure() -> None:
    text = _script_text()
    assert "$priorAction" in text
    assert "$priorExecute" in text
    assert "$priorArguments" in text
    # One shared rollback function used by both the switch and verify paths.
    assert "function Restore-PriorAction" in text
    assert text.count("Restore-PriorAction -Execute") >= 2
    assert "Rollback failed" in text


def test_upgrade_script_restores_action_without_empty_working_directory() -> None:
    restore = _script_text().split("function Restore-PriorAction", 1)[1].split(
        "# 1.", 1
    )[0]
    assert "[string]::IsNullOrWhiteSpace($WorkingDirectory)" in restore
    assert "New-ScheduledTaskAction @actionParameters" in restore


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


def test_upgrade_script_keeps_distinct_versioned_probe_logs() -> None:
    text = _script_text()
    assert "probe-$Version.stdout.log" in text
    assert "probe-$Version.stderr.log" in text


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
