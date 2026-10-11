[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $true
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

try {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($Root)) {
        $Root = (Get-Location).Path
    }
    Set-Location $Root

    Write-Step "Selecting Python interpreter (prefer 'py -3')"
    $PythonExe = $null
    $PythonPrefixArgs = @()

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonExe = "py"
        $PythonPrefixArgs = @("-3")
        Write-Host "Using interpreter: py -3"
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = "python"
        Write-Host "Using interpreter: python"
    }
    else {
        throw "No Python interpreter found. Install Python 3 and ensure 'py' or 'python' is in PATH."
    }

    $VenvDir = Join-Path $Root "venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"

    Write-Step "Creating virtual environment at .\venv"
    if (-not (Test-Path $VenvPython)) {
        & $PythonExe @PythonPrefixArgs -m venv $VenvDir
        Write-Host "Virtual environment created."
    }
    else {
        Write-Host "Virtual environment already exists. Reusing it."
    }

    Write-Step "Activating virtual environment"
    & $VenvActivate
    Write-Host "Activated .\venv"

    Write-Step "Upgrading pip, setuptools, wheel"
    & $VenvPython -m pip install --upgrade pip setuptools wheel

    $RequirementsPath = Join-Path $Root "requirements.txt"
    if (-not (Test-Path $RequirementsPath)) {
        throw "requirements.txt was not found at: $RequirementsPath"
    }

    Write-Step "Installing dependencies from requirements.txt"
    Write-Host "Using PyTorch CUDA wheel index from requirements.txt."
    & $VenvPython -m pip install -r $RequirementsPath

    Write-Step "Preparing external repository"
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "Git is required but not found in PATH. Install Git for Windows and rerun."
    }

    $ExternalDir = Join-Path $Root "external"
    $RepoDir = Join-Path $ExternalDir "flappy-bird-gymnasium"
    $RepoUrl = "https://github.com/markub3327/flappy-bird-gymnasium"

    if (-not (Test-Path $ExternalDir)) {
        New-Item -ItemType Directory -Path $ExternalDir | Out-Null
    }

    if (Test-Path (Join-Path $RepoDir ".git")) {
        Write-Host "Repository already exists. Pulling latest changes..."
        & git -C $RepoDir pull --ff-only
    }
    elseif (Test-Path $RepoDir) {
        Write-Host "Folder .\external\flappy-bird-gymnasium exists but is not a git repository. Skipping clone."
        Write-Host "If this is unexpected, remove the folder and rerun setup."
    }
    else {
        Write-Host "Cloning repository into .\external\flappy-bird-gymnasium ..."
        & git clone $RepoUrl $RepoDir
    }

    Write-Step "Installing flappy-bird-gymnasium in editable mode"
    if (-not (Test-Path $RepoDir)) {
        throw "Repository path not found: $RepoDir"
    }
    & $VenvPython -m pip install -e $RepoDir

    Write-Step "Runtime versions and CUDA status"
    $InfoScript = @'
import platform
import subprocess
import sys
import torch

print(f"python version: {platform.python_version()} ({sys.executable})")
pip_version = subprocess.check_output([sys.executable, "-m", "pip", "--version"], text=True).strip()
print(f"pip version: {pip_version}")
print(f"torch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
else:
    print("torch.cuda.get_device_name(0): N/A")
'@
    & $VenvPython -c $InfoScript

    Write-Step "Verifying Gymnasium environment creation and stepping"
    $VerifyScript = @'
import gymnasium as gym
import flappy_bird_gymnasium  # noqa: F401

env_id = "FlappyBird-v0"
env = gym.make(env_id, render_mode=None, audio_on=False)
obs, info = env.reset(seed=123)
for i in range(8):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print(f"Environment check OK: created {env_id} and stepped random actions.")
'@
    & $VenvPython -c $VerifyScript

    Write-Step "Setup finished"
    Write-Host "If torch.cuda.is_available() is False, run:"
    Write-Host ".\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    Write-Host "Then verify with:"
    Write-Host ".\venv\Scripts\python.exe -c ""import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"""
}
catch {
    Write-Host ""
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
