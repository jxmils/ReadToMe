@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PYTHONUTF8=1"

where uv >nul 2>&1
if errorlevel 1 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    ) else if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    ) else (
        echo Installing uv...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    )
)

where uv >nul 2>&1
if errorlevel 1 (
    echo Could not find uv. Install it from https://docs.astral.sh/uv/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Setting up Python 3.12 environment...
    uv sync --python 3.12
    if errorlevel 1 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    uv pip install --quiet --no-deps -e .
)

where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    ".venv\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo NVIDIA GPU found, installing CUDA build of PyTorch...
        uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    )
)

if not exist "samples\sample_paper.pdf" (
    ".venv\Scripts\python.exe" scripts\generate_sample_paper.py
)

if "%~1"=="" (
    ".venv\Scripts\python.exe" -m readtome --open
) else (
    ".venv\Scripts\python.exe" -m readtome %*
)
