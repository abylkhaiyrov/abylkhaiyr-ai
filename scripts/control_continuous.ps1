param(
    [ValidateSet('start', 'stop', 'status')][string]$Action = 'status',
    [ValidateRange(1, 5000)][int]$Steps = 60,
    [ValidateRange(1, 24)][int]$Polls = 6,
    [ValidateRange(60, 86400)][int]$Interval = 600
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $root 'experiments\continuous'
New-Item -Path $work -ItemType Directory -Force | Out-Null
$pidFile = Join-Path $work 'service.pid'
$stopFile = Join-Path $work 'STOP'
$running = $null
if (Test-Path -LiteralPath $pidFile) {
    $storedPid = Get-Content -LiteralPath $pidFile -Raw
    $running = Get-Process -Id ([int]$storedPid) -ErrorAction SilentlyContinue
    if ($running) {
        $startedFile = Join-Path $work 'service.started'
        if ((Test-Path -LiteralPath $startedFile) -and
            ($running.StartTime.ToUniversalTime().Ticks.ToString() -ne (Get-Content -LiteralPath $startedFile))) {
            $running = $null
        }
    }
}
switch ($Action) {
    'status' {
        if ($running) { Write-Output "Continuous learning PID $($running.Id)" }
        else { Write-Output 'Continuous learning is stopped' }
    }
    'stop' {
        Set-Content -LiteralPath $stopFile -Value 'stop'
        Write-Output 'Stop requested'
    }
    'start' {
        if ($running) {
            Write-Output "Continuous learning is already running with PID $($running.Id)"
            return
        }
        Remove-Item -LiteralPath $stopFile -Force -ErrorAction SilentlyContinue
        $python = Join-Path $root '.venv\Scripts\python.exe'
        $arguments = '-u', 'scripts/continuous_learning.py', '--steps', $Steps, '--polls', $Polls, '--interval', $Interval
        $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $work 'service.stdout.log') -RedirectStandardError (Join-Path $work 'service.stderr.log') -PassThru
        $process.Id | Set-Content -LiteralPath $pidFile
        $process.StartTime.ToUniversalTime().Ticks | Set-Content -LiteralPath (Join-Path $work 'service.started')
        Write-Output "Continuous learning PID $($process.Id)"
    }
}
