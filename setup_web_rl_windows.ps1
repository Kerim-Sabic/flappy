[CmdletBinding()]
param(
    [int]$Port = 5173
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $argsText = $Arguments -join " "
        throw "Command failed (exit $exitCode): $FilePath $argsText"
    }
}

try {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ([string]::IsNullOrWhiteSpace($Root)) {
        $Root = (Get-Location).Path
    }
    Set-Location $Root

    Write-Step "Selecting Python interpreter (prefer Python 3.12; avoid free-threaded 3.13t)"
    $PythonExe = $null
    $PythonPrefixArgs = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $preferredTags = @("3.12", "3.11", "3.10", "3.13")
        foreach ($tag in $preferredTags) {
            & py "-$tag" -c "import sys; print(sys.executable)" *> $null
            if ($LASTEXITCODE -eq 0) {
                $PythonExe = "py"
                $PythonPrefixArgs = @("-$tag")
                break
            }
        }

        if (-not $PythonExe) {
            $PythonExe = "py"
            $PythonPrefixArgs = @("-3")
        }
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExe = "python"
        $PythonPrefixArgs = @()
    }
    else {
        throw "Python 3 not found. Install Python and ensure 'py' or 'python' is in PATH."
    }

    $PyInfoJson = & $PythonExe @PythonPrefixArgs -c "import json,sys; print(json.dumps({'version': f'{sys.version_info.major}.{sys.version_info.minor}', 'abiflags': getattr(sys, 'abiflags', ''), 'executable': sys.executable}))"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to inspect selected Python interpreter."
    }
    $PyInfo = $PyInfoJson | ConvertFrom-Json
    $PyVersion = [version]$PyInfo.version
    if ($PyInfo.abiflags -like "*t*") {
        throw "Detected free-threaded Python at '$($PyInfo.executable)' (abiflags='$($PyInfo.abiflags)'). Use regular Python 3.12 (not 3.13t) and rerun."
    }
    if ($PyVersion -ge [version]"3.13") {
        throw "Detected Python $($PyInfo.version) at '$($PyInfo.executable)'. This setup currently targets Python 3.12 for wheel compatibility."
    }
    Write-Host "Using interpreter: $($PyInfo.executable) (Python $($PyInfo.version))"

    $VenvDir = Join-Path $Root "venv_web"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"

    Write-Step "Creating virtual environment at .\venv_web"
    if (-not (Test-Path $VenvPython)) {
        $createVenvArgs = @()
        $createVenvArgs += $PythonPrefixArgs
        $createVenvArgs += @("-m", "venv", $VenvDir)
        Invoke-External -FilePath $PythonExe -Arguments $createVenvArgs
        Write-Host "Created .\venv_web"
    }
    else {
        Write-Host ".\venv_web already exists; reusing."
    }

    Write-Step "Activating virtual environment"
    . $VenvActivate

    Write-Step "Upgrading pip, setuptools, wheel"
    Invoke-External -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

    $ReqPath = Join-Path $Root "requirements_web_rl.txt"
    if (-not (Test-Path $ReqPath)) {
        throw "Missing requirements file: $ReqPath"
    }

    Write-Step "Installing Python dependencies"
    try {
        Invoke-External -FilePath $VenvPython -Arguments @("-m", "pip", "install", "-r", $ReqPath)
    }
    catch {
        Write-Host "Primary dependency install failed. Trying explicit CUDA torch install fallback..." -ForegroundColor Yellow
        Invoke-External -FilePath $VenvPython -Arguments @(
            "-m", "pip", "install", "--upgrade",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu128"
        )
        Invoke-External -FilePath $VenvPython -Arguments @(
            "-m", "pip", "install",
            "gymnasium==0.29.1",
            "stable-baselines3==2.3.2",
            "numpy==1.26.4",
            "playwright==1.49.1",
            "tensorboard==2.17.1",
            "Pillow>=10.0.0"
        )
    }

    Write-Step "Installing Playwright Chromium"
    Invoke-External -FilePath $VenvPython -Arguments @("-m", "playwright", "install", "chromium")

    Write-Step "Checking Node.js and npm"
    $NodeInfo = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $NodeInfo) {
        $NodeInfo = Get-Command node -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    if (-not $NodeInfo) {
        throw "Node.js is not installed or not in PATH. Install Node.js LTS from https://nodejs.org and rerun."
    }
    $NodeExe = $NodeInfo.Source

    $NpmInfo = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $NpmInfo) {
        throw "npm.cmd was not found in PATH. Reinstall Node.js LTS and rerun."
    }
    $NpmCmd = $NpmInfo.Source

    $nodeVersion = & $NodeExe --version
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to run 'node --version'."
    }
    $npmVersion = & $NpmCmd --version
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to run 'npm --version'."
    }
    Write-Host "node version: $nodeVersion"
    Write-Host "npm version: $npmVersion"

    Write-Step "Creating/patching local webgame in .\webgame"
    $PatchScript = Join-Path $Root "patch_webgame_for_rl.js"
    if (-not (Test-Path $PatchScript)) {
        throw "Missing patch script: $PatchScript"
    }
    $WebgameDir = Join-Path $Root "webgame"
    Invoke-External -FilePath $NodeExe -Arguments @($PatchScript, "--target", $WebgameDir)

    if (-not (Test-Path (Join-Path $WebgameDir "package.json"))) {
        throw "webgame package.json was not created. Check patch_webgame_for_rl.js output."
    }

    Write-Step "Installing webgame npm dependencies"
    Push-Location $WebgameDir
    try {
        Invoke-External -FilePath $NpmCmd -Arguments @("install")
    }
    finally {
        Pop-Location
    }

    Write-Step "Verifying window.__rl.getState() through Playwright"
    $serverProcess = Start-Process `
        -FilePath $NpmCmd `
        -ArgumentList @("run", "start", "--", "--port", "$Port") `
        -WorkingDirectory $WebgameDir `
        -PassThru `
        -WindowStyle Hidden

    try {
        $env:WEB_RL_PORT = "$Port"
        $VerifyScript = @'
import os
import time
import urllib.request
from playwright.sync_api import sync_playwright

port = int(os.environ["WEB_RL_PORT"])
url = f"http://127.0.0.1:{port}"

deadline = time.time() + 25.0
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200:
                break
    except Exception as exc:
        last_error = exc
        time.sleep(0.5)
else:
    raise RuntimeError(f"Webgame server not reachable at {url}. Last error: {last_error}")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=15000)
    page.wait_for_function(
        "() => window.__rl && typeof window.__rl.getState === 'function' && "
        "typeof window.__rl.reset === 'function' && typeof window.__rl.flap === 'function'",
        timeout=15000,
    )
    page.evaluate(
        "() => window.__rl.reset({seed: 123, drLevel: 'moderate', curriculumProgress: 0.4})"
    )
    state = page.evaluate("() => window.__rl.getState()")
    dr = page.evaluate(
        "() => (window.__rl.getRandomization ? window.__rl.getRandomization() : null)"
    )
    browser.close()

print(f"window.__rl.getState() OK: {state}")
print(f"window.__rl.getRandomization() OK: {dr}")
'@
        Invoke-External -FilePath $VenvPython -Arguments @("-c", $VerifyScript)
    }
    finally {
        Remove-Item Env:WEB_RL_PORT -ErrorAction SilentlyContinue
        if ($serverProcess -and -not $serverProcess.HasExited) {
            Stop-Process -Id $serverProcess.Id -Force
        }
    }

    Write-Step "Setup complete"
    Write-Host "Start the local webgame:"
    Write-Host "  cd .\webgame"
    Write-Host "  npm run start -- --port $Port"
    Write-Host ""
    Write-Host "Train PPO:"
    Write-Host "  .\venv_web\Scripts\python.exe .\train_web_ppo.py --url http://127.0.0.1:$Port"
    Write-Host "Hybrid train (local + live TapCap):"
    Write-Host "  .\venv_web\Scripts\python.exe .\train_tapcap_hybrid.py --stage hybrid --local-url http://127.0.0.1:$Port --live-url https://play.tapcapgame.com/ --n-envs 4 --subproc"
    Write-Host "Dual trainer launcher (2 GPUs):"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\launch_dual_trainers.ps1"
    Write-Host ""
    Write-Host "Watch trained agent:"
    Write-Host "  .\venv_web\Scripts\python.exe .\play_web.py --url http://127.0.0.1:$Port"
    Write-Host "Watch live TapCap PPO:"
    Write-Host "  .\venv_web\Scripts\python.exe .\play_tapcap_ppo.py --url https://play.tapcapgame.com/"
}
catch {
    Write-Host ""
    Write-Host "Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "If CUDA torch install failed, run this manually inside venv_web:" -ForegroundColor Yellow
    Write-Host ".\venv_web\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor Yellow
    exit 1
}
