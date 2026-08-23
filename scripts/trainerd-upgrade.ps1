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
  - Stops the Scheduled Task and, only when needed, the verified trainerd
    process that owns the configured listening port.
  - Requires explicit live and isolated-probe serve arguments, then compares
    their authentication mode and complete LAN policy before cutover.
  - Preserves the prior task action and makes Task Scheduler own trainerd.exe.
  - Verifies status=ok and the expected version after the switch.
  - On any failure, restores the prior action through one shared rollback path
    and verifies health against the prior version.

.PARAMETER Version
Candidate release version, e.g. 0.3.12. Used for the venv directory and logs.

.PARAMETER WheelUrl
URL or local path to the candidate wheel to install.

.PARAMETER ServeArguments
Exact arguments after `trainerd serve` for the replacement daemon.

.PARAMETER ProbeArguments
Exact arguments after `trainerd serve` for the isolated candidate probe. Use
{probe_port} and {probe_state_dir} placeholders where those values belong.
Quote {probe_state_dir}; the helper preserves the caller's quoting.

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
  -WheelUrl https://github.com/Kh1ng/trainerd/releases/download/v0.3.13/trainerd-0.3.13-py3-none-any.whl `
  -ServeArguments '--lan --state-dir "C:\ProgramData\trainerd\state" --host 0.0.0.0 --port 7860' `
  -ProbeArguments '--lan --state-dir "{probe_state_dir}" --host 127.0.0.1 --port {probe_port}'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [Parameter(Mandatory = $true)]
    [string]$WheelUrl,

    [Parameter(Mandatory = $true)]
    [string]$ServeArguments,

    [Parameter(Mandatory = $true)]
    [string]$ProbeArguments,

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
if (-not $ProbeArguments.Contains("{probe_port}")) {
    throw "ProbeArguments must contain {probe_port}."
}
if ($ProbeArguments -match '(?i)(^|\s)--lan(\s|$)' -and -not $ProbeArguments.Contains("{probe_state_dir}")) {
    throw "LAN ProbeArguments must contain {probe_state_dir}."
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

function Assert-SamePolicy {
    param(
        [object]$Expected,
        [object]$Candidate
    )
    $fields = @("mode", "authentication_required")
    if ($Expected.mode -eq "lan") {
        $fields += "lan_policy_hash"
    }
    else {
        $fields += "projects"
    }
    foreach ($field in $fields) {
        $expectedValue = $Expected.$field | ConvertTo-Json -Compress
        $candidateValue = $Candidate.$field | ConvertTo-Json -Compress
        if ($expectedValue -cne $candidateValue) {
            throw "Candidate policy mismatch for $field (live=$expectedValue candidate=$candidateValue)."
        }
    }
}

function Stop-VerifiedTrainerdListener {
    param([string]$ExpectedVersion)
    $health = Get-HealthJson -BaseUrl $HealthUrl
    if (
        -not $health -or
        $health.status -ne "ok" -or
        $health.version -ne $ExpectedVersion -or
        $health.pending_jobs -gt 0 -or $health.running_jobs -gt 0
    ) {
        return $false
    }

    $listenPort = ([uri]$HealthUrl).Port
    $listenerPids = @(
        Get-NetTCPConnection -LocalPort $listenPort -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique
    )
    if ($listenerPids.Count -ne 1) {
        return $false
    }
    $listenerPid = [int]$listenerPids[0]
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid"
    if (
        -not $process -or
        $process.CommandLine -notmatch '(?i)trainerd' -or
        $process.CommandLine -notmatch '(?i)(^|\s)serve(\s|$)'
    ) {
        return $false
    }
    Stop-Process -Id $listenerPid -Force -ErrorAction Stop
    return $true
}

function Stop-DaemonTask {
    param([string]$ExpectedVersion)
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    for ($i = 0; $i -lt 15; $i++) {
        if (-not (Get-HealthJson -BaseUrl $HealthUrl)) {
            return $true
        }
        Start-Sleep -Seconds 2
    }
    if (-not (Stop-VerifiedTrainerdListener -ExpectedVersion $ExpectedVersion)) {
        return $false
    }
    for ($i = 0; $i -lt 15; $i++) {
        if (-not (Get-HealthJson -BaseUrl $HealthUrl)) {
            return $true
        }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Restore-PriorAction {
    param(
        [string]$Execute,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$ExpectedVersion
    )
    if (-not (Stop-DaemonTask -ExpectedVersion $ExpectedVersion)) {
        return $false
    }
    $actionParameters = @{
        Execute = $Execute
        Argument = $Arguments
    }
    if (-not [string]::IsNullOrWhiteSpace($WorkingDirectory)) {
        $actionParameters.WorkingDirectory = $WorkingDirectory
    }
    $oldAction = New-ScheduledTaskAction @actionParameters
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
$probeArgumentsResolved = $ProbeArguments.Replace("{probe_port}", "$ProbePort").Replace("{probe_state_dir}", "$probeDir")
$probeProcess = Start-Process -FilePath $pythonExe `
    -ArgumentList "-m trainerd serve $probeArgumentsResolved" `
    -RedirectStandardOutput $probeOutputLog `
    -RedirectStandardError $probeErrorLog `
    -PassThru `
    -WindowStyle Hidden
try {
    $healthy = Wait-For-Health -BaseUrl $probeHealth -ExpectedVersion $Version
    if (-not $healthy) {
        throw "Candidate did not reach status=ok version=$Version on port $ProbePort. See $probeOutputLog and $probeErrorLog"
    }
    $candidateHealth = Get-HealthJson -BaseUrl $probeHealth
    Assert-SamePolicy -Expected $live -Candidate $candidateHealth
    Write-Host "Candidate probe healthy."
}
finally {
    if ($probeProcess -and -not $probeProcess.HasExited) {
        Stop-Process -Id $probeProcess.Id -Force
    }
    # Remove only the unique subdirectory this script created.
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $probeDir
}

# 4. Preserve the prior task action and build the direct replacement action
#    before touching the task.
Write-Host "Preserving prior task action for $TaskName ..."
$task = Get-ScheduledTask -TaskName $TaskName
if (-not $task -or -not $task.Actions -or $task.Actions.Count -lt 1) {
    throw "Scheduled Task $TaskName has no actions to preserve."
}
$priorAction = $task.Actions[0]
$priorExecute = $priorAction.Execute
$priorArguments = $priorAction.Arguments
$priorWorkingDirectory = $priorAction.WorkingDirectory

$newAction = New-ScheduledTaskAction -Execute $trainerdExe `
    -Argument "serve $ServeArguments" `
    -WorkingDirectory $logDir

# 5. Recheck the queue immediately before the cutover so a job submitted
#    during the install/probe phase aborts here with no changes made.
$queue = Get-QueueState -BaseUrl $HealthUrl
if ($queue.Pending -gt 0 -or $queue.Running -gt 0) {
    throw "Refusing cutover: queue gained $($queue.Pending) pending and $($queue.Running) running jobs during the upgrade. No changes were made."
}

Write-Host "Stopping task $TaskName and switching to $Version ..."
$taskActionChanged = $false
try {
    if (-not (Stop-DaemonTask -ExpectedVersion $priorVersion)) {
        throw "The old daemon did not release its port. Its listener did not pass the trainerd identity and empty-queue checks."
    }
    Set-ScheduledTask -TaskName $TaskName -Action $newAction -ErrorAction Stop
    $taskActionChanged = $true
    Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
}
catch {
    if (-not $taskActionChanged) {
        if (-not (Get-HealthJson -BaseUrl $HealthUrl)) {
            Start-ScheduledTask -TaskName $TaskName -ErrorAction Stop
            if (-not (Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $priorVersion -Attempts 30)) {
                throw "Rollback failed: prior task action did not return to version $priorVersion."
            }
        }
        throw "Switch to $Version failed before changing the task action. Prior action retained: $_"
    }
    Write-Warning "Switch failed; restoring prior action."
    if (-not (Restore-PriorAction -Execute $priorExecute -Arguments $priorArguments -WorkingDirectory $priorWorkingDirectory -ExpectedVersion $priorVersion)) {
        throw "Rollback failed: daemon did not return to prior version $priorVersion after restoring the prior action."
    }
    throw "Switch to $Version failed. Prior action restored and healthy: $_"
}

# 6. Verify status=ok and the expected version.
Write-Host "Verifying $Version on $HealthUrl ..."
$verified = Wait-For-Health -BaseUrl $HealthUrl -ExpectedVersion $Version
if ($verified) {
    try {
        $installedHealth = Get-HealthJson -BaseUrl $HealthUrl
        Assert-SamePolicy -Expected $live -Candidate $installedHealth
    }
    catch {
        Write-Warning "Installed daemon policy verification failed: $_"
        $verified = $false
    }
}
if (-not $verified) {
    Write-Warning "Candidate verification failed; restoring prior action."
    if (-not (Restore-PriorAction -Execute $priorExecute -Arguments $priorArguments -WorkingDirectory $priorWorkingDirectory -ExpectedVersion $priorVersion)) {
        throw "Rollback failed: daemon did not return to prior version $priorVersion after restoring the prior action."
    }
    throw "Candidate verification failed. Prior action restored and healthy."
}

Write-Host "trainerd upgraded to $Version. Prior action preserved for rollback: $priorExecute $priorArguments"
