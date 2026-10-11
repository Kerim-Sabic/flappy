[CmdletBinding()]
param(
    [string]$LocalUrl = "http://127.0.0.1:5173",
    [string]$LiveUrl = "https://play.tapcapgame.com/",
    [int]$EnvPerTrainer = 1,
    [int]$TotalTimestepsLocal = 1500000,
    [int]$TotalTimestepsLive = 5000000,
    [string]$ModelDir = ".\models_web\hybrid",
    [string]$LogDir = ".\logs_web\hybrid",
    [switch]$LiveOnly = $true,
    [int]$PromoteIntervalSeconds = 0,
    [int]$EvalFreqLive = 64,
    [int]$SaveFreqLive = 256,
    [int]$EvalEpisodesLive = 2,
    [switch]$NoSubproc
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Resolve-Python {
    $venvPython = Join-Path $PSScriptRoot "venv_web\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    throw "Missing Python interpreter: $venvPython. Run setup_web_rl_windows.ps1 first."
}

function Start-Trainer {
    param(
        [string]$PythonExe,
        [string]$GpuId,
        [string]$TrainerId,
        [int]$Seed,
        [string]$StdOutPath,
        [string]$StdErrPath,
        [string]$LocalUrl,
        [string]$LiveUrl,
        [int]$EnvPerTrainer,
        [int]$TotalTimestepsLocal,
        [int]$TotalTimestepsLive,
        [int]$EvalFreqLive,
        [int]$SaveFreqLive,
        [int]$EvalEpisodesLive,
        [string]$ModelDir,
        [string]$LogDir,
        [switch]$LiveOnly,
        [switch]$NoSubproc
    )

    $subprocArg = ""
    if (-not $NoSubproc.IsPresent) {
        $subprocArg = "--subproc"
    }

    if ($LiveOnly.IsPresent) {
        $cmd = @"
set CUDA_VISIBLE_DEVICES=$GpuId && set PYTHONUNBUFFERED=1 && "$PythonExe" ".\train_tapcap_hybrid.py" --profile tapcap_live_strict --stage live --trainer-id $TrainerId --seed $Seed --live-url "$LiveUrl" --n-envs-live $EnvPerTrainer --vec-mode dummy --total-timesteps-live $TotalTimestepsLive --eval-freq $EvalFreqLive --save-freq $SaveFreqLive --eval-episodes $EvalEpisodesLive --health-log-interval 30 --model-dir "$ModelDir" --log-dir "$LogDir" --headless --state-stack 4 --action-repeat 2
"@
    } else {
        $cmd = @"
set CUDA_VISIBLE_DEVICES=$GpuId && set PYTHONUNBUFFERED=1 && "$PythonExe" ".\train_tapcap_hybrid.py" --stage hybrid --trainer-id $TrainerId --seed $Seed --local-url "$LocalUrl" --live-url "$LiveUrl" --n-envs $EnvPerTrainer --total-timesteps-local $TotalTimestepsLocal --total-timesteps-live $TotalTimestepsLive --model-dir "$ModelDir" --log-dir "$LogDir" --headless --dr-level moderate --state-stack 4 --action-repeat 2 $subprocArg
"@
    }

    return Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList @("/c", $cmd) `
        -WorkingDirectory $PSScriptRoot `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdOutPath `
        -RedirectStandardError $StdErrPath
}

function Invoke-Selection {
    param(
        [string]$PythonExe,
        [string]$LocalUrl,
        [string]$LiveUrl,
        [string]$CandidateGlob = ".\models_web\hybrid\trainer_*\final_model.zip",
        [int]$MaxCandidates = 0,
        [switch]$LiveOnly
    )

    $matches = @(Get-ChildItem -Path $CandidateGlob -File -ErrorAction SilentlyContinue)
    if ($matches.Count -eq 0) {
        Write-Host "No candidate models found for pattern: $CandidateGlob (skipping selection)." -ForegroundColor Yellow
        return $false
    }

    if ($LiveOnly.IsPresent) {
        & $PythonExe ".\select_best_tapcap_model.py" `
            --model-glob $CandidateGlob `
            --max-candidates $MaxCandidates `
            --live-url $LiveUrl `
            --eval-suite live `
            --selection-metric consistency `
            --eval-episodes 12 `
            --output-best-model ".\models_web\hybrid\best_model.zip" `
            --output-best-vecnorm ".\models_web\hybrid\best_vecnormalize.pkl" `
            --output-report ".\models_web\hybrid\selection_report.json"
        return ($LASTEXITCODE -eq 0)
    } else {
        & $PythonExe ".\select_best_tapcap_model.py" `
            --model-glob $CandidateGlob `
            --max-candidates $MaxCandidates `
            --local-url $LocalUrl `
            --live-url $LiveUrl `
            --eval-suite both `
            --selection-metric consistency `
            --eval-episodes 8 `
            --output-best-model ".\models_web\hybrid\best_model.zip" `
            --output-best-vecnorm ".\models_web\hybrid\best_vecnormalize.pkl" `
            --output-report ".\models_web\hybrid\selection_report.json"
        return ($LASTEXITCODE -eq 0)
    }
}

try {
    Set-Location $PSScriptRoot
    $pythonExe = Resolve-Python

    $runDir = Join-Path $PSScriptRoot "logs_web\dual_launcher"
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    if ($LiveOnly.IsPresent) {
        Write-Step "Starting trainer A on GPU 0 (live-only profile)"
    } else {
        Write-Step "Starting trainer A on GPU 0 (hybrid)"
    }
    $aOut = Join-Path $runDir "trainer_a.out.log"
    $aErr = Join-Path $runDir "trainer_a.err.log"
    $procA = Start-Trainer `
        -PythonExe $pythonExe `
        -GpuId "0" `
        -TrainerId "trainer_a" `
        -Seed 42 `
        -StdOutPath $aOut `
        -StdErrPath $aErr `
        -LocalUrl $LocalUrl `
        -LiveUrl $LiveUrl `
        -EnvPerTrainer $EnvPerTrainer `
        -TotalTimestepsLocal $TotalTimestepsLocal `
        -TotalTimestepsLive $TotalTimestepsLive `
        -EvalFreqLive $EvalFreqLive `
        -SaveFreqLive $SaveFreqLive `
        -EvalEpisodesLive $EvalEpisodesLive `
        -ModelDir $ModelDir `
        -LogDir $LogDir `
        -LiveOnly:$LiveOnly `
        -NoSubproc:$NoSubproc

    if ($LiveOnly.IsPresent) {
        Write-Step "Starting trainer B on GPU 1 (live-only profile)"
    } else {
        Write-Step "Starting trainer B on GPU 1 (hybrid)"
    }
    $bOut = Join-Path $runDir "trainer_b.out.log"
    $bErr = Join-Path $runDir "trainer_b.err.log"
    $procB = Start-Trainer `
        -PythonExe $pythonExe `
        -GpuId "1" `
        -TrainerId "trainer_b" `
        -Seed 1042 `
        -StdOutPath $bOut `
        -StdErrPath $bErr `
        -LocalUrl $LocalUrl `
        -LiveUrl $LiveUrl `
        -EnvPerTrainer $EnvPerTrainer `
        -TotalTimestepsLocal $TotalTimestepsLocal `
        -TotalTimestepsLive $TotalTimestepsLive `
        -EvalFreqLive $EvalFreqLive `
        -SaveFreqLive $SaveFreqLive `
        -EvalEpisodesLive $EvalEpisodesLive `
        -ModelDir $ModelDir `
        -LogDir $LogDir `
        -LiveOnly:$LiveOnly `
        -NoSubproc:$NoSubproc

    Write-Host "Trainer A PID: $($procA.Id)"
    Write-Host "Trainer B PID: $($procB.Id)"
    Write-Host "Promote interval (seconds): $PromoteIntervalSeconds"
    Write-Host "Live eval/save cadence: eval_freq=$EvalFreqLive save_freq=$SaveFreqLive eval_episodes=$EvalEpisodesLive"
    Write-Host "Logs:"
    Write-Host "  $aOut"
    Write-Host "  $aErr"
    Write-Host "  $bOut"
    Write-Host "  $bErr"

    Write-Step "Waiting for trainers to finish"
    $lastPromote = [DateTime]::UtcNow.AddYears(-1)
    while ($true) {
        $procA.Refresh()
        $procB.Refresh()

        if ($PromoteIntervalSeconds -gt 0 -and (
            ([DateTime]::UtcNow - $lastPromote).TotalSeconds -ge $PromoteIntervalSeconds
        )) {
            Write-Step "Periodic model promotion run"
            try {
                # During training, promote from current eval-best models.
                $periodicOk = Invoke-Selection `
                    -PythonExe $pythonExe `
                    -LocalUrl $LocalUrl `
                    -LiveUrl $LiveUrl `
                    -CandidateGlob ".\models_web\hybrid\trainer_*\live\best_model.zip" `
                    -MaxCandidates 4 `
                    -LiveOnly:$LiveOnly
                if (-not $periodicOk) {
                    # Early in training, eval-best may not exist yet. Fallback to checkpoints.
                    $periodicOk = Invoke-Selection `
                        -PythonExe $pythonExe `
                        -LocalUrl $LocalUrl `
                        -LiveUrl $LiveUrl `
                        -CandidateGlob ".\models_web\hybrid\trainer_*\live\checkpoints\ppo_live_*_steps.zip" `
                        -MaxCandidates 6 `
                        -LiveOnly:$LiveOnly
                }
                if ($periodicOk) {
                    $lastPromote = [DateTime]::UtcNow
                } else {
                    Write-Host "Periodic promotion skipped." -ForegroundColor Yellow
                }
            }
            catch {
                Write-Host "Periodic promotion failed: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        }

        if ($procA.HasExited -and $procB.HasExited) {
            break
        }
        Start-Sleep -Seconds 5
    }

    $procA.Refresh()
    $procB.Refresh()
    if ($procA.ExitCode -ne 0 -or $procB.ExitCode -ne 0) {
        throw "One or both trainers failed. Exit codes: trainer_a=$($procA.ExitCode), trainer_b=$($procB.ExitCode)"
    }

    Write-Step "Selecting best model (consistency metric)"
    # Final promotion: prefer final models, fallback to live best if needed.
    $finalOk = Invoke-Selection `
        -PythonExe $pythonExe `
        -LocalUrl $LocalUrl `
        -LiveUrl $LiveUrl `
        -CandidateGlob ".\models_web\hybrid\trainer_*\final_model.zip" `
        -MaxCandidates 4 `
        -LiveOnly:$LiveOnly
    if (-not $finalOk) {
        Write-Host "Final-model selection failed, trying live/best_model fallback..." -ForegroundColor Yellow
        $finalOk = Invoke-Selection `
            -PythonExe $pythonExe `
            -LocalUrl $LocalUrl `
            -LiveUrl $LiveUrl `
            -CandidateGlob ".\models_web\hybrid\trainer_*\live\best_model.zip" `
            -MaxCandidates 4 `
            -LiveOnly:$LiveOnly
    }
    if (-not $finalOk) {
        Write-Host "Live/best selection failed, trying checkpoint fallback..." -ForegroundColor Yellow
        $finalOk = Invoke-Selection `
            -PythonExe $pythonExe `
            -LocalUrl $LocalUrl `
            -LiveUrl $LiveUrl `
            -CandidateGlob ".\models_web\hybrid\trainer_*\live\checkpoints\ppo_live_*_steps.zip" `
            -MaxCandidates 10 `
            -LiveOnly:$LiveOnly
    }
    if (-not $finalOk) {
        throw "Model selection failed."
    }

    Write-Step "Done"
    Write-Host "Best model: .\models_web\hybrid\best_model.zip"
    Write-Host "Selection report: .\models_web\hybrid\selection_report.json"
}
catch {
    Write-Host ""
    Write-Host "Launcher failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
