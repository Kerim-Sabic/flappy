from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import torch
from stable_baselines3 import PPO

from web_flappy_env import WebFlappyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch trained PPO agent play the local web Flappy game.")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5173")
    parser.add_argument("--model-dir", type=str, default=r".\models_web")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-dt", type=float, default=0.05)
    return parser.parse_args()


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
        f"Start it first (for example: cd .\\webgame && npm run start -- --port 5173). "
        f"Last error: {last_error}"
    )


def resolve_model_path(model_dir: Path) -> Path:
    best_model = model_dir / "best_model.zip"
    final_model = model_dir / "final_model.zip"
    if best_model.exists():
        return best_model
    if final_model.exists():
        return final_model
    raise FileNotFoundError(
        f"No model found. Expected '{best_model}' or '{final_model}'. Train first with train_web_ppo.py."
    )


def main() -> int:
    args = parse_args()
    wait_for_url(args.url)

    model_path = resolve_model_path(Path(args.model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_path}")
    print(f"Inference device: {device}")

    model = PPO.load(str(model_path), device=device)
    env = WebFlappyEnv(
        url=args.url,
        headless=False,
        step_dt=args.step_dt,
        render_mode="human",
    )

    try:
        obs, info = env.reset(seed=args.seed)
        _ = info
        episode = 1
        episode_reward = 0.0
        print("Running agent with visible browser. Press Ctrl+C to stop.")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            if terminated or truncated:
                score = info.get("score", "n/a")
                print(f"Episode {episode}: reward={episode_reward:.2f}, score={score}")
                obs, info = env.reset()
                episode += 1
                episode_reward = 0.0
    except KeyboardInterrupt:
        print("Playback stopped by user.")
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
