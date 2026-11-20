$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runDirectory = Join-Path $projectRoot 'experiments\tiny_gpu'
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$pidFile = Join-Path $runDirectory 'process.pid'
if (Test-Path -LiteralPath $pidFile) {
    $trainingPid = [int](Get-Content -LiteralPath $pidFile)
    $existingProcess = Get-Process -Id $trainingPid -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.ProcessName -eq 'python') {
        throw "Training is already running with PID $trainingPid"
    }
}
Remove-Item -LiteralPath (Join-Path $runDirectory 'STOP') -Force -ErrorAction SilentlyContinue
$env:PYTHONUTF8 = '1'
$process = Start-Process -FilePath $pythonPath -ArgumentList '-u', 'scripts/train_gpu.py' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDirectory 'stdout.log') -RedirectStandardError (Join-Path $runDirectory 'stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $runDirectory 'process.pid')
Write-Output "Training PID: $($process.Id)"
