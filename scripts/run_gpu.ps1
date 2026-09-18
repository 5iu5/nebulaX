param([string]$Python = ".venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
& $Python -c "import torch; assert torch.cuda.is_available(), 'Install CUDA-enabled PyTorch first'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA preflight failed" }
& $Python -m acv.prepare
if ($LASTEXITCODE -ne 0) { throw "Preparation failed" }
& $Python -m acv.evaluate --model tcn_mil --nested --device cuda
if ($LASTEXITCODE -ne 0) { throw "Neural evaluation failed or exhausted its time budget" }
Write-Host "Copy the evaluation JSON and fit-cache artifacts back, or rerun all models here before selecting and training."
