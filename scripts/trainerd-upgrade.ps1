<#
.SYNOPSIS
Atomically upgrade the trainerd Windows daemon.

.DESCRIPTION
Installs a candidate release into a new versioned virtual environment, probes
it on an alternate port with an isolated LAN state directory, and only then
switches the trainerd-lan Scheduled Task action. Every cutover is reversible:
if the candidate fails to start or verify, the prior task action is restored
and its health rechecked.

Safety rules enforced by this script:
  - Refuses to cut over while the live daemon reports pending or running jobs.
  - Installs the candidate in a new versioned virtual environment, never
    overwriting the current environment.
  - Probes the candidate on an alternate port with an isolated LAN state dir.
  - Stops only the trainerd-lan Scheduled Task. It never kills unrelated
    Python processes.
  - Preserves the prior task action and appends versioned startup logs.
  - Verifies status=ok and the expected version after the switch.
  - On failure, restores the prior action and verifies its health.

.PARAMETER Version
Candidate release version, e.g. 0.3.12. Used for the venv directory and logs.

.PARAMETER WheelUrl
URL or local path to the candidate wheel to install.

.PARAMETER TaskName
Name of the Scheduled Task that runs the daemon (default: trainerd-lan).

.PARAMETER StateDir
LAN state directory the daemon manages. The probe uses its own isolated copy.

.PARAMETER InstallRoot
Parent directory for venvs and logs (default: C:\ProgramData\trainerd).

.PARAMETER HealthUrl
Base URL of the running daemon (default: http://127.0.0.1:7860).

.PARAMETER ProbePort
Alternate port used to probe the candidate (default: 7861).

.PARAMETER ProbeStateDir
Isolated LAN state directory for the candidate probe. Defaults to
<InstallRoot>\probe-state.

.EXAMPLE
.\trainerd-upgrade.ps1 -Version 0.3.13 `
  -WheelUrl https://github.com/Kh1ng/trainerd/releases/download/v0.3.13/trainerd-0.3.13-py3-none-any.whl
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$WheelUrl,

    [string]$TaskName = "trainerd-lan",
    [string]$StateDir = "C:\ProgramData\trainerd\state",
    [string]$InstallRoot = "C:\ProgramData\trainerd",
    [string]$HealthUrl = "http://127.0.0.1:7860",
    [int]$ProbePort = 7861,
    [string]$ProbeStateDir = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProbeStateDir) {
    $ProbeStateDir = Join-Path $InstallRoot "probe-state"
}

$venvDir = Join-Path $InstallRoot "venvs\$Version"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$trainerdExe = Join-Path $venvDir "Scripts\trainerd.exe"
$logDir = Join-Path $InstallRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-HealthJson {
    param([string]$BaseUrl)
    $health = Join-Path $BaseUrl "api/health"
    try {
        $response = Invoke-RestMethod -Uri $health -Method Get -TimeoutSec 15
        return $response
    }
    catch {
        return $null
    }
}

function Wait-For-Health {
    param(
        [string]$BaseUrl,
        [string]$ExpectedVersion,
        [int]$Attempts = 30,
        [int]$DelaySeconds = 2
    )
    for ($i = 0; $i -lt $Attempts; $i++) {
        $json = Get-HealthJson -BaseUrl $BaseUrl
        if ($json -and $json.status -eq "ok" -and $json.version -eq $ExpectedVersion) {
            return $true
        }
        Start-Sleep -Seconds $DelaySeconds
    }
    return $false
}

# 1. Refuse the cutover while pending or running jobs exist.
Write-Host "Checking live daemon queue at $HealthUrl ..."
$live = Get-HealthJson -BaseUrl $HealthUrl
if (-not $live) {
    throw "Live daemon is not reachable at $HealthUrl. Refusing to cut over."
}
$pending = [int]$live.pending_jobs
$running = [int]$live.running_jobs
if ($pending -gt 0 -or $running -gt 0) {
    throw "Refusing cutover: $pending pending and $running running jobs. Wait for the queue to drain."
}
Write-Host "Queue is empty (pending=$pending running=$running)."

# 2. Install the candidate in a new versioned virtual environment.
Write-Host "Installing candidate $Version into $venvDir ..."
if (-not (Test-Path $venvDir)) {
    & py -3.12 -m venv $venvDir
}
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment was not created: $pythonExe"
}
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed for $venvDir" }
& $pythonExe -m pip install $WheelUrl
if ($LASTEXITCODE -ne 0) { throw "pip install failed for $WheelUrl" }
& $trainerdExe --version
if ($LASTEXITCODE -ne 0) { throw "Candidate trainerd.exe did not run" }

# 3. Probe the candidate on an alternate port with an isolated state dir.
Write-Host "Probing candidate on port $ProbePort with isolated state $ProbeStateDir ..."
$probeLog = Join-Path $logDir "probe-$Version.log"
$probeHealth = "http://127.0.0.1:$ProbePort"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProbeStateDir
New-Item -ItemType Directory -Force -Path $ProbeStateDir | Out-Null
$probeProcess = Start-Process -FilePath $trainerdExe `
    -ArgumentList @("serve", "--lan", "--state-dir", "`"$ProbeStateDir`"", "--host", "127.0.0.1", "--port", "$ProbePort") `
    -RedirectStandardOutput $probeLog `
    -RedirectStandardError $probeLog `
    -PassThru `
    -WindowStyle Hidden
try {
    $healthy = Wait-For-Health -BaseUrl $probeHealth -ExpectedVersion $Version
    if (-not $healthy) {
        throw "Candidate did not reach status=ok version=$Version on port $ProbePort. See $probeLog"
    }
    Write-Host "Candidate probe healthy."
}
finally {
    if ($probeProcess -and -not $probeProcess.HasExited) {
        Stop-Process -Id $probeProcess.Id -Force
    }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $ProbeStateDir
}

# 4. Preserve the prior task action and stop only the trainerd-lan task.
Write-Host "Preserving prior task action for $TaskName ..."
$task = Get-ScheduledTask -TaskName $TaskName
if (-not $task -or -not $task.Actions -or $task.Actions.Count -lt 1) {
    throw "Scheduled Task $TaskName has no actions to preserve."
}
$priorAction = $task.Actions[0]
$priorExecute = $priorAction.Execute
$priorArguments = $priorAction.Arguments
$priorWorkingDirectory = $priorAction.WorkingDirectory

# 5. Switch the task action to a versioned launcher that appends logs.
Write-Host "Stopping task $TaskName and switching to $Version ..."
Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
$launcherPath = Join-Path $logDir "run-lan-$Version.ps1"
$startupLog = Join-Path $logDir "startup-$Version.log"
$launcherBody = @"
param()
`$ErrorActionPreference = "Stop"
& "$trainerdExe" serve --lan --state-dir "$StateDir" --host 0.0.0.0 --port 7860 *>> "$startupLog"
"@
$launcherBody | Out-File -FilePath $launcherPath -Encoding utf8
$newAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$launcherPath`"" `
    -WorkingDirectory $logDir
try {
    Set-ScheduledTask -TaskName $TaskName -Action $newAction -ErrorAction Stop
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
}
catch {
    Write-Warning "Switch failed; restoring prior action."
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $oldAction = New-ScheduledTaskAction -Execute $priorExecute -Argument $priorArguments -WorkingDirectory $priorWorkingDirectory
    Set-ScheduledTask -TaskName $TaskName -Action $oldAction
    Start-ScheduledTask -TaskName $TaskName
    throw "Switch to $Version failed. Prior action restored: $_"
}

# 6. Verify status=ok and the expected version.
Write-Host "Verifying $Version on $HealthUrl ..."
$verified = Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $Version
if (-not $verified) {
    Write-Warning "Candidate verification failed; restoring prior action."
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $oldAction = New-ScheduledTaskAction -Execute $priorExecute -Argument $priorArguments -WorkingDirectory $priorWorkingDirectory
    Set-ScheduledTask -TaskName $TaskName -Action $oldAction
    Start-ScheduledTask -TaskName $TaskName
    $priorVerified = Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $Version -Attempts 30
    if (-not $priorVerified) {
        throw "Rollback failed: daemon did not return to health after restoring the prior action."
    }
    throw "Candidate verification failed. Prior action restored and healthy."
}

Write-Host "trainerd upgraded to $Version. Prior action preserved for rollback: $priorExecute $priorArguments"
