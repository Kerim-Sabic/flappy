from __future__ import annotations

import argparse
import os
import platform
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage

DEFAULT_ENV_ID = "FlappyBird-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on flappy-bird-gymnasium.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=str, default=r".\models")
    parser.add_argument("--log-dir", type=str, default=r".\logs")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--save-freq", type=int, default=200_000)
    parser.add_argument("--subproc", action="store_true", help="Use SubprocVecEnv.")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    return parser.parse_args()


def get_registered_env_ids() -> list[str]:
    ids: set[str] = set()
    try:
        for key, spec in gym.envs.registry.items():
            if hasattr(spec, "id"):
                ids.add(spec.id)
            elif isinstance(key, str):
                ids.add(key)
    except Exception:
        pass
    return sorted(ids)


def format_env_id_error(env_id: str, original_error: Exception | None = None) -> str:
    ids = get_registered_env_ids()
    candidates = [name for name in ids if "flappy" in name.lower()]
    if not candidates:
        candidates_text = "none found"
    else:
        candidates_text = ", ".join(candidates)

    message = f"Could not create Gymnasium env '{env_id}'."
    if original_error is not None:
        message += f" Original error: {original_error}"
    message += f" Registered ids containing 'flappy': {candidates_text}."
    message += " Ensure 'import flappy_bird_gymnasium' runs before gym.make(...)."
    return message


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_image_observation_space(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Box) and len(space.shape) == 3


def channels_last(shape: tuple[int, ...]) -> bool:
    if len(shape) != 3:
        return False
    return shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4)


def make_env_fn(env_id: str, seed: int, rank: int, render_mode: str | None = None):
    def _init():
        try:
            env = gym.make(env_id, render_mode=render_mode, audio_on=False)
        except Exception as exc:
            raise RuntimeError(format_env_id_error(env_id, exc)) from exc
        env = Monitor(env)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def build_vec_env(env_id: str, n_envs: int, seed: int, use_subproc: bool):
    env_fns = [make_env_fn(env_id=env_id, seed=seed, rank=i, render_mode=None) for i in range(n_envs)]
    if use_subproc:
        return SubprocVecEnv(env_fns, start_method="spawn")
    return DummyVecEnv(env_fns)


def main() -> int:
    args = parse_args()

    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    try:
        import flappy_bird_gymnasium  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to import flappy_bird_gymnasium. Run setup_windows.ps1 first."
        ) from exc

    model_dir = Path(args.model_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    registered_ids = get_registered_env_ids()
    if args.env_id not in registered_ids:
        raise RuntimeError(format_env_id_error(args.env_id))

    try:
        probe_env = gym.make(args.env_id, render_mode=None, audio_on=False)
    except Exception as exc:
        raise RuntimeError(format_env_id_error(args.env_id, exc)) from exc

    obs_space = probe_env.observation_space
    action_space = probe_env.action_space
    probe_env.close()

    image_obs = is_image_observation_space(obs_space)
    policy = "CnnPolicy" if image_obs else "MlpPolicy"

    set_global_seeds(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Environment ID: {args.env_id}")
    print(f"Observation space: {obs_space}")
    print(f"Action space: {action_space}")
    print(f"Selected policy: {policy}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    use_subproc = args.subproc
    if platform.system() == "Windows" and not args.subproc:
        print("Windows detected: using DummyVecEnv by default.")
    if platform.system() == "Windows" and args.subproc:
        print("Windows detected and --subproc enabled: using SubprocVecEnv (spawn).")

    train_env = None
    eval_env = None
    model = None

    try:
        train_env = build_vec_env(
            env_id=args.env_id, n_envs=args.n_envs, seed=args.seed, use_subproc=use_subproc
        )
        eval_env = build_vec_env(env_id=args.env_id, n_envs=1, seed=args.seed + 10_000, use_subproc=False)

        if image_obs:
            if channels_last(obs_space.shape):
                train_env = VecTransposeImage(train_env)
                eval_env = VecTransposeImage(eval_env)
                print("Applied VecTransposeImage for channel-last observations.")

            train_env = VecFrameStack(train_env, n_stack=4)
            eval_env = VecFrameStack(eval_env, n_stack=4)
            print("Applied VecFrameStack(n_stack=4).")

        checkpoint_freq = max(args.save_freq // args.n_envs, 1)
        eval_freq = max(args.eval_freq // args.n_envs, 1)
        print(f"Checkpoint callback frequency (steps): {checkpoint_freq}")
        print(f"Eval callback frequency (steps): {eval_freq}")

        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_flappy",
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
            policy=policy,
            env=train_env,
            verbose=1,
            seed=args.seed,
            device=device,
            tensorboard_log=str(log_dir),
            n_steps=512,
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_flappy",
            progress_bar=True,
        )

        final_model_path = model_dir / "final_model"
        model.save(str(final_model_path))
        print(f"Saved final model to: {final_model_path.with_suffix('.zip')}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        if model is not None:
            interrupted_model_path = model_dir / "interrupted_model"
            model.save(str(interrupted_model_path))
            print(f"Saved interrupted model to: {interrupted_model_path.with_suffix('.zip')}")
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
