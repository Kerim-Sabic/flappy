from __future__ import annotations

import argparse
import os
import platform
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from web_flappy_env import WebFlappyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on local web Flappy game.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5173")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default=r".\models_web")
    parser.add_argument("--log-dir", type=str, default=r".\logs_web")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv (advanced).")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run browser headless during training (default: true).",
    )
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wait_for_url(url: str, timeout_s: float = 25.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"Webgame URL is not reachable: {url}. "
        f"Start the server first (for example: cd .\\webgame && npm run start -- --port 5173). "
        f"Last error: {last_error}"
    )


def make_env_fn(url: str, headless: bool, step_dt: float, seed: int, rank: int):
    def _init():
        env = WebFlappyEnv(url=url, headless=headless, step_dt=step_dt)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(url: str, headless: bool, step_dt: float, n_envs: int, seed: int, use_subproc: bool):
    env_fns = [
        make_env_fn(url=url, headless=headless, step_dt=step_dt, seed=seed, rank=i)
        for i in range(n_envs)
    ]
    if use_subproc:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


def main() -> int:
    args = parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    if args.n_envs > 2:
        print("Warning: --n-envs > 2 launches many browser instances and can be very heavy on Windows.")

    wait_for_url(args.url)
    set_global_seeds(args.seed)

    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training URL: {args.url}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    use_subproc = args.subproc
    if platform.system() == "Windows" and not args.subproc:
        print("Windows detected: using DummyVecEnv by default.")
    if platform.system() == "Windows" and args.subproc:
        print("Windows detected and --subproc enabled: using SubprocVecEnv with spawn.")

    train_env = None
    eval_env = None
    model = None
    try:
        train_env = build_vec_env(
            url=args.url,
            headless=args.headless,
            step_dt=args.step_dt,
            n_envs=args.n_envs,
            seed=args.seed,
            use_subproc=use_subproc,
        )
        eval_env = build_vec_env(
            url=args.url,
            headless=True,
            step_dt=args.step_dt,
            n_envs=1,
            seed=args.seed + 10_000,
            use_subproc=False,
        )

        checkpoint_freq = max(args.save_freq // args.n_envs, 1)
        eval_freq = max(args.eval_freq // args.n_envs, 1)
        print(f"Checkpoint callback frequency (steps): {checkpoint_freq}")
        print(f"Eval callback frequency (steps): {eval_freq}")

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_web_flappy",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir / "eval"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )
        callbacks = CallbackList([checkpoint_callback, eval_callback])

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=str(log_dir),
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_web_flappy",
            progress_bar=False,
        )

        final_model = model_dir / "final_model"
        model.save(str(final_model))
        print(f"Saved final model to: {final_model.with_suffix('.zip')}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if model is not None:
            interrupted_model = model_dir / "interrupted_model"
            model.save(str(interrupted_model))
            print(f"Saved interrupted model to: {interrupted_model.with_suffix('.zip')}")
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
