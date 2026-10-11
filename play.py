from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO

DEFAULT_ENV_ID = "FlappyBird-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Flappy Bird with a trained PPO model.")
    parser.add_argument("--model-dir", type=str, default=r".\models")
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio", action="store_true", help="Enable game audio.")
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
    candidates_text = ", ".join(candidates) if candidates else "none found"

    message = f"Could not create Gymnasium env '{env_id}'."
    if original_error is not None:
        message += f" Original error: {original_error}"
    message += f" Registered ids containing 'flappy': {candidates_text}."
    message += " Ensure 'import flappy_bird_gymnasium' runs before gym.make(...)."
    return message


def resolve_model_path(model_dir: Path) -> Path:
    best_model = model_dir / "best_model.zip"
    final_model = model_dir / "final_model.zip"

    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model

    raise FileNotFoundError(
        f"No model found. Expected '{best_model}' or '{final_model}'. Train first with train_ppo.py."
    )


def main() -> int:
    args = parse_args()

    try:
        import flappy_bird_gymnasium  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to import flappy_bird_gymnasium. Run setup_windows.ps1 first."
        ) from exc

    model_dir = Path(args.model_dir)
    model_path = resolve_model_path(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_path}")
    print(f"Inference device: {device}")
    model = PPO.load(str(model_path), device=device)

    env = None
    try:
        try:
            env = gym.make(args.env_id, render_mode="human", audio_on=args.audio)
        except Exception as exc:
            raise RuntimeError(format_env_id_error(args.env_id, exc)) from exc

        obs, info = env.reset(seed=args.seed)
        episode_idx = 1
        episode_reward = 0.0
        print("Running episodes. Press Ctrl+C to stop.")

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)

            if terminated or truncated:
                score = info.get("score", "n/a") if isinstance(info, dict) else "n/a"
                print(f"Episode {episode_idx}: reward={episode_reward:.2f}, score={score}")
                obs, info = env.reset()
                episode_reward = 0.0
                episode_idx += 1
    except KeyboardInterrupt:
        print("Playback stopped by user.")
    except Exception as exc:
        print(f"Playback failed: {exc}")
        print("Rendering tips for Windows:")
        print("1) Run in a local desktop session (headless/RDP sessions may fail to open a pygame window).")
        print("2) Update NVIDIA drivers and reboot.")
        print("3) Ensure pygame is installed in this venv: .\\venv\\Scripts\\python.exe -m pip show pygame")
        print("4) Re-run setup_windows.ps1 to repair dependencies.")
        return 1
    finally:
        if env is not None:
            env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
