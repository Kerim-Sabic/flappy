# Web RL Hybrid Pipeline (Windows, Parallel, Adaptive)

This setup supports both hybrid and **live-only TapCap PPO**.

For your current target, use the live-only path:
1. Train only on `https://play.tapcapgame.com/`
2. Run dual trainers (GPU0 + GPU1)
3. Select best by consistency metric on live eval
4. Watch with visible playback and optional auto-reload

## 1) Setup

Run once in project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_web_rl_windows.ps1
```

It will:
- create `.\venv_web`
- install Python deps + Playwright Chromium
- patch/create local webgame
- install npm deps for `.\webgame`
- verify `window.__rl` state + randomization API

## 2) Start local pretrain game server

In a separate terminal:

```powershell
cd .\webgame
npm run start -- --port 5173
```

URL:

`http://127.0.0.1:5173`

## 3) Live-only training (single trainer)

```powershell
.\venv_web\Scripts\python.exe .\train_tapcap_hybrid.py `
  --profile tapcap_live_strict `
  --stage live `
  --trainer-id solo `
  --live-url https://play.tapcapgame.com/ `
  --n-envs-live 1 `
  --vec-mode dummy `
  --total-timesteps-live 5000000
```

Important flags:
- `--profile {none,tapcap_live_strict}`
- `--stage {local,live,hybrid}`
- `--vec-mode {auto,dummy,subproc}`
- `--n-envs`, `--n-envs-live`
- `--trainer-id`
- `--device` (`auto`, `cuda`, `cuda:0`, etc.)
- `--net-arch {small,medium,large}`
- `--state-stack`
- `--action-repeat`
- `--health-log-interval`
- `--health-jsonl-path`
- `--resume <model.zip>`

## 4) Dual trainer mode (2 GPUs, parallel search, live-only default)

Runs trainer A on GPU 0 and trainer B on GPU 1, then selects best:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch_dual_trainers.ps1
```

Default behavior:
- `EnvPerTrainer=1` (stable Windows baseline, total 2 envs across both trainers)
- live-only (`-LiveOnly` default on)
- `--profile tapcap_live_strict` for each trainer
- `--vec-mode dummy` for each trainer
- best model promoted to:
  - `.\models_web\hybrid\best_model.zip`
  - `.\models_web\hybrid\best_vecnormalize.pkl`

Optional continuous promotion while trainers run:

```powershell
powershell -ExecutionPolicy Bypass -File .\launch_dual_trainers.ps1 -PromoteIntervalSeconds 300
```

## 5) Manual model selection

```powershell
.\venv_web\Scripts\python.exe .\select_best_tapcap_model.py `
  --model-glob ".\models_web\hybrid\*\final_model.zip" `
  --eval-suite live `
  --selection-metric consistency `
  --eval-episodes 12
```

Selection report:

`.\models_web\hybrid\selection_report.json`

## 6) Watch trained PPO on live TapCap

```powershell
.\venv_web\Scripts\python.exe .\play_tapcap_ppo.py `
  --url https://play.tapcapgame.com/ `
  --model-path .\models_web\hybrid\best_model.zip `
  --vecnorm-path .\models_web\hybrid\best_vecnormalize.pkl `
  --auto-reload-model `
  --reload-seconds 60 `
  --status-interval-seconds 30
```

## 7) Training dashboard

Health JSON log is written per stage, for example:

`.\logs_web\hybrid\trainer_a\live\health.jsonl`

TensorBoard:

```powershell
.\venv_web\Scripts\python.exe -m tensorboard.main --logdir .\logs_web\hybrid --port 6006
```

## 8) Heuristic adaptive bot (non-PPO)

Still available:

```powershell
.\venv_web\Scripts\python.exe .\play_tapcap_vision.py --url https://play.tapcapgame.com/
```

## Test / smoke commands

```powershell
.\venv_web\Scripts\python.exe -m unittest tests.test_tapcap_state
.\venv_web\Scripts\python.exe -m unittest tests.test_env_lifecycle
```

The lifecycle test is skipped automatically if local server is not running.

## Troubleshooting

### Playwright sync API / asyncio loop errors
Use provided env classes (`TapCapStateEnv`, `LocalTapCapStateEnv`) and scripts; they isolate Playwright calls in worker threads.

### Live training unstable with many envs
- use `--vec-mode dummy` on Windows live runs
- keep `--n-envs-live 1` per trainer first
- only try `--vec-mode subproc` after a stable soak test
- keep `--auto-fallback-envs` enabled

### CUDA unavailable

```powershell
.\venv_web\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.\venv_web\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### Node/npm missing
Install Node.js LTS (`https://nodejs.org`) and rerun setup script.
