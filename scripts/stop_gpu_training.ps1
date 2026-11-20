$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runDirectory = Join-Path $projectRoot 'experiments\tiny_gpu'
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
Set-Content -LiteralPath (Join-Path $runDirectory 'STOP') -Value 'stop'
Write-Output 'Stop requested. Training will save a checkpoint after the current step.'
