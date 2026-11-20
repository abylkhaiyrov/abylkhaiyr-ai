param([ValidateSet('auto', 'pretrain', 'sft')][string]$Stage = 'auto')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($Stage -eq 'auto') {
    $Stage = if (Test-Path -LiteralPath (Join-Path $projectRoot 'checkpoints\tiny_gpu_sft\best\meta.json')) { 'sft' } else { 'pretrain' }
}
$runName = if ($Stage -eq 'sft') { 'tiny_gpu_sft' } else { 'tiny_gpu' }
$configPath = "configs/$runName.yaml"
$runDirectory = Join-Path $projectRoot "experiments\$runName"
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
foreach ($candidate in @('tiny_gpu', 'tiny_gpu_sft')) {
    $pidFile = Join-Path $projectRoot "experiments\$candidate\process.pid"
    if (Test-Path -LiteralPath $pidFile) {
        $trainingPid = [int](Get-Content -LiteralPath $pidFile)
        $existingProcess = Get-Process -Id $trainingPid -ErrorAction SilentlyContinue
        if ($existingProcess -and $existingProcess.ProcessName -eq 'python') {
            Write-Output "Обучение уже работает (PID $trainingPid). Повторный запуск не требуется."
            return
        }
    }
}
Remove-Item -LiteralPath (Join-Path $runDirectory 'STOP') -Force -ErrorAction SilentlyContinue
$env:PYTHONUTF8 = '1'
$process = Start-Process -FilePath $pythonPath -ArgumentList '-u', 'scripts/train_gpu.py', '--config', $configPath -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDirectory 'stdout.log') -RedirectStandardError (Join-Path $runDirectory 'stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $runDirectory 'process.pid')
Write-Output "Training PID: $($process.Id)"
