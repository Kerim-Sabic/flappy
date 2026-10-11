# Flappy Bird RL (Windows 11, Gymnasium + SB3 + PyTorch CUDA)

This project sets up a reproducible PPO training pipeline for `flappy-bird-gymnasium` on Windows using:
- Gymnasium
- Stable-Baselines3
- PyTorch with CUDA wheels

The Flappy Bird environment ID used here is: `FlappyBird-v0`.

## 1) Setup

Run from PowerShell in this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

What setup does:
- Creates and activates `.\venv`
- Installs dependencies from `requirements.txt`
- Clones `https://github.com/markub3327/flappy-bird-gymnasium` into `.\external\flappy-bird-gymnasium`
- Installs the repo in editable mode
- Prints Python/pip/torch/CUDA info
- Verifies `FlappyBird-v0` can be created and stepped

## 2) Train PPO

```powershell
.\venv\Scripts\python.exe .\train_ppo.py --total-timesteps 5000000
```

Useful options:
- `--n-envs 16` (default)
- `--seed 42` (default)
- `--eval-freq 50000` (default)
- `--save-freq 200000` (default)
- `--subproc` (optional; Windows default is `DummyVecEnv`)

Outputs:
- TensorBoard logs in `.\logs`
- Checkpoints in `.\models\checkpoints\`
- Best model in `.\models\best_model.zip`
- Final model in `.\models\final_model.zip`

TensorBoard:

```powershell
.\venv\Scripts\python.exe -m tensorboard.main --logdir .\logs
```

## 3) Watch the Agent Play

```powershell
.\venv\Scripts\python.exe .\play.py
```

`play.py` loads:
1. `.\models\best_model.zip` (preferred)
2. `.\models\final_model.zip` (fallback)

## Troubleshooting

### `torch.cuda.is_available()` is `False`

1. Update/install NVIDIA driver and reboot.
2. Reinstall CUDA wheels for PyTorch:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. Re-check CUDA:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

### Environment ID errors (`FlappyBird-v0` not found)

The package must be imported for registration side-effects. To list registered Flappy-like IDs:

```powershell
.\venv\Scripts\python.exe -c "import gymnasium as gym; import flappy_bird_gymnasium; print([spec.id for spec in gym.envs.registry.values() if 'flappy' in spec.id.lower()])"
```

### `SubprocVecEnv` issues on Windows

Use the default `DummyVecEnv` (do not pass `--subproc`).  
`DummyVecEnv` is the safe default on Windows and avoids multiprocessing edge cases.
