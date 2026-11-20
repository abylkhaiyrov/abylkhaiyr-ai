$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
foreach ($runName in @('tiny_gpu', 'tiny_gpu_sft')) {
    $runDirectory = Join-Path $projectRoot "experiments\$runName"
    New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $runDirectory 'STOP') -Value 'stop'
}
Write-Output 'Stop requested. Training will save a checkpoint after the current step.'
