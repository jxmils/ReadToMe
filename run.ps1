$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$env:PYTHONUTF8 = "1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    foreach ($p in @("$env:USERPROFILE\.local\bin", "$env:USERPROFILE\.cargo\bin")) {
        if (Test-Path (Join-Path $p "uv.exe")) {
            $env:PATH = "$p;$env:PATH"
            break
        }
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Setting up Python 3.12 environment..."
    uv sync --python 3.12
} else {
    uv pip install --quiet --no-deps -e .
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & $venvPython -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "NVIDIA GPU found, installing CUDA build of PyTorch..."
        uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    }
}

$samplePdf = Join-Path $PSScriptRoot "samples\sample_paper.pdf"
if (-not (Test-Path $samplePdf)) {
    & $venvPython scripts\generate_sample_paper.py
}

if ($args.Count -eq 0) {
    & $venvPython -m readtome --open
} else {
    & $venvPython -m readtome @args
}
