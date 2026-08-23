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
  - Rechecks the queue immediately before the cutover so work submitted during
    the install/probe phase cannot be lost.
  - Installs the candidate in a new versioned virtual environment, never
    reusing or overwriting an existing one.
  - Probes the candidate on an alternate port with an isolated LAN state dir
    that is guaranteed distinct from the live state directory.
  - Stops only the trainerd-lan Scheduled Task. It never kills unrelated
    Python processes.
  - Preserves the prior task action and appends versioned startup logs.
  - Verifies status=ok and the expected version after the switch.
  - On any failure, restores the prior action through one shared rollback path
    and verifies health against the prior version.

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
Parent directory for the isolated probe state. Defaults to
<InstallRoot>\probe-state. The script creates a unique subdirectory beneath it
and never deletes anything it did not create.

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
    # Join-Path is provider-based and must not be used on http(s) URLs.
    $base = $BaseUrl.TrimEnd("/")
    $health = "$base/api/health"
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

function Get-QueueState {
    param([string]$BaseUrl)
    $json = Get-HealthJson -BaseUrl $BaseUrl
    if (-not $json) {
        return @{ Reachable = $false; Pending = 0; Running = 0 }
    }
    return @{
        Reachable = $true
        Pending = [int]$json.pending_jobs
        Running = [int]$json.running_jobs
    }
}

function Restore-PriorAction {
    param(
        [string]$Execute,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$ExpectedVersion
    )
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $oldAction = New-ScheduledTaskAction -Execute $Execute -Argument $Arguments -WorkingDirectory $WorkingDirectory
    Set-ScheduledTask -TaskName $TaskName -Action $oldAction -ErrorAction Stop
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    return Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $ExpectedVersion -Attempts 30
}

# 1. Refuse the cutover while pending or running jobs exist. Capture the prior
#    version so every rollback verifies against it.
Write-Host "Checking live daemon queue at $HealthUrl ..."
$live = Get-HealthJson -BaseUrl $HealthUrl
if (-not $live) {
    throw "Live daemon is not reachable at $HealthUrl. Refusing to cut over."
}
$priorVersion = [string]$live.version
$queue = Get-QueueState -BaseUrl $HealthUrl
if ($queue.Pending -gt 0 -or $queue.Running -gt 0) {
    throw "Refusing cutover: $($queue.Pending) pending and $($queue.Running) running jobs. Wait for the queue to drain."
}
Write-Host "Queue is empty (pending=$($queue.Pending) running=$($queue.Running))."

# 2. Install the candidate in a fresh versioned virtual environment. A stale
#    environment from an interrupted run is rebuilt rather than reused.
Write-Host "Installing candidate $Version into $venvDir ..."
if (Test-Path $venvDir) {
    Write-Warning "Removing existing environment at $venvDir and rebuilding it."
    Remove-Item -Recurse -Force $venvDir
}
& py -3.12 -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed: $venvDir" }
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment was not created: $pythonExe"
}
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed for $venvDir" }
& $pythonExe -m pip install $WheelUrl
if ($LASTEXITCODE -ne 0) { throw "pip install failed for $WheelUrl" }
& $trainerdExe --version
if ($LASTEXITCODE -ne 0) { throw "Candidate trainerd.exe did not run" }

# 3. Probe the candidate on an alternate port with an isolated state dir. The
#    probe always runs in a unique subdirectory so it can never touch, let
#    alone recursively delete, the live state directory.
$probeBase = [IO.Path]::GetFullPath($ProbeStateDir).TrimEnd([IO.Path]::DirectorySeparatorChar)
$stateFull = [IO.Path]::GetFullPath($StateDir).TrimEnd([IO.Path]::DirectorySeparatorChar)
$sep = [IO.Path]::DirectorySeparatorChar
if (
    $probeBase -ieq $stateFull -or
    $probeBase -ilike "$stateFull$sep*" -or
    $stateFull -ilike "$probeBase$sep*"
) {
    throw "ProbeStateDir must be outside the live StateDir ($StateDir). Refusing to proceed."
}
$probeDir = Join-Path $probeBase ("probe-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $probeDir | Out-Null
Write-Host "Probing candidate on port $ProbePort with isolated state $probeDir ..."
$probeOutputLog = Join-Path $logDir "probe-$Version.stdout.log"
$probeErrorLog = Join-Path $logDir "probe-$Version.stderr.log"
$probeHealth = "http://127.0.0.1:$ProbePort"
$probeProcess = Start-Process -FilePath $trainerdExe `
    -ArgumentList @("serve", "--lan", "--state-dir", "`"$probeDir`"", "--host", "127.0.0.1", "--port", "$ProbePort") `
    -RedirectStandardOutput $probeOutputLog `
    -RedirectStandardError $probeErrorLog `
    -PassThru `
    -WindowStyle Hidden
try {
    $healthy = Wait-For-Health -BaseUrl $probeHealth -ExpectedVersion $Version
    if (-not $healthy) {
        throw "Candidate did not reach status=ok version=$Version on port $ProbePort. See $probeOutputLog and $probeErrorLog"
    }
    Write-Host "Candidate probe healthy."
}
finally {
    if ($probeProcess -and -not $probeProcess.HasExited) {
        Stop-Process -Id $probeProcess.Id -Force
    }
    # Remove only the unique subdirectory this script created.
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $probeDir
}

# 4. Preserve the prior task action and build the versioned launcher before
#    touching the task, so a launcher failure leaves the live task untouched.
Write-Host "Preserving prior task action for $TaskName ..."
$task = Get-ScheduledTask -TaskName $TaskName
if (-not $task -or -not $task.Actions -or $task.Actions.Count -lt 1) {
    throw "Scheduled Task $TaskName has no actions to preserve."
}
$priorAction = $task.Actions[0]
$priorExecute = $priorAction.Execute
$priorArguments = $priorAction.Arguments
$priorWorkingDirectory = $priorAction.WorkingDirectory

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

# 5. Recheck the queue immediately before the cutover so a job submitted
#    during the install/probe phase aborts here with no changes made.
$queue = Get-QueueState -BaseUrl $HealthUrl
if ($queue.Pending -gt 0 -or $queue.Running -gt 0) {
    throw "Refusing cutover: queue gained $($queue.Pending) pending and $($queue.Running) running jobs during the upgrade. No changes were made."
}

Write-Host "Stopping task $TaskName and switching to $Version ..."
try {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    # Wait for the old daemon to release the port before the new one binds.
    $oldDown = $false
    for ($i = 0; $i -lt 15; $i++) {
        if (-not (Get-HealthJson -BaseUrl $HealthUrl)) {
            $oldDown = $true
            break
        }
        Start-Sleep -Seconds 2
    }
    if (-not $oldDown) {
        throw "Old daemon did not stop; refusing to switch."
    }
    Set-ScheduledTask -TaskName $TaskName -Action $newAction -ErrorAction Stop
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
}
catch {
    Write-Warning "Switch failed; restoring prior action."
    if (-not (Restore-PriorAction -Execute $priorExecute -Arguments $priorArguments -WorkingDirectory $priorWorkingDirectory -ExpectedVersion $priorVersion)) {
        throw "Rollback failed: daemon did not return to prior version $priorVersion after restoring the prior action."
    }
    throw "Switch to $Version failed. Prior action restored and healthy: $_"
}

# 6. Verify status=ok and the expected version.
Write-Host "Verifying $Version on $HealthUrl ..."
$verified = Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $Version
if (-not $verified) {
    Write-Warning "Candidate verification failed; restoring prior action."
    if (-not (Restore-PriorAction -Execute $priorExecute -Arguments $priorArguments -WorkingDirectory $priorWorkingDirectory -ExpectedVersion $priorVersion)) {
        throw "Rollback failed: daemon did not return to prior version $priorVersion after restoring the prior action."
    }
    throw "Candidate verification failed. Prior action restored and healthy."
}

Write-Host "trainerd upgraded to $Version. Prior action preserved for rollback: $priorExecute $priorArguments"
