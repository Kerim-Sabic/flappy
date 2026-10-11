from __future__ import annotations

import argparse
import glob
from collections import Counter, deque
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tapcap_state_env import TapCapStateEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play live TapCap using a trained PPO state-policy.")
    parser.add_argument("--url", type=str, default="https://play.tapcapgame.com/")
    parser.add_argument("--model-path", type=str, default=r".\models_web\hybrid\best_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default=r".\models_web\hybrid\best_vecnormalize.pkl")
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--state-stack", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--auto-reload-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reload-seconds", type=float, default=60.0)
    parser.add_argument("--status-interval-seconds", type=float, default=30.0)
    return parser.parse_args()


def wait_for_url(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=4) as response:
                if 200 <= response.status < 500:
                    return
        except urllib.error.HTTPError as exc:
            if 200 <= int(exc.code) < 500:
                return
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"URL not reachable: {url}. Last error: {last_error}")


def resolve_model(path: Path) -> Path:
    seen: set[str] = set()
    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            return
        seen.add(key)
        if candidate.exists():
            candidates.append(candidate)

    # Primary + common promoted paths.
    add_candidate(path)
    add_candidate(Path(r".\models_web\hybrid\best_model.zip"))
    add_candidate(Path(r".\models_web\hybrid\final_model.zip"))
    add_candidate(Path(r".\models_web\hybrid\best_model_smoke_liveonly.zip"))
    add_candidate(Path(r".\models_web\hybrid\best_model_smoke.zip"))

    # If promoted model not available yet, use the freshest trainer/live artifact.
    patterns = [
        r".\models_web\hybrid\trainer_*\live\best_model.zip",
        r".\models_web\hybrid\trainer_*\live\checkpoints\ppo_live_*_steps.zip",
        r".\models_web\hybrid\trainer_*\final_model.zip",
        r".\models_web\hybrid\*\final_model.zip",
        r".\models_web\hybrid\top_models\rank_*.zip",
    ]
    glob_hits: list[Path] = []
    for pattern in patterns:
        for hit in glob.glob(pattern):
            p = Path(hit)
            if p.exists():
                glob_hits.append(p)
    glob_hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in glob_hits:
        add_candidate(p)

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No playable model found. Expected at least one of: "
        f"{path}, .\\models_web\\hybrid\\best_model.zip, .\\models_web\\hybrid\\trainer_*\\live\\best_model.zip"
    )


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def file_mtime(path: Path) -> float:
    if path.exists():
        return path.stat().st_mtime
    return -1.0


def resolve_vecnorm(path: Path, model_path: Path) -> Path | None:
    if path.exists():
        return path

    candidates: list[Path] = [
        model_path.parent / "best_vecnormalize.pkl",
        model_path.parent / "final_vecnormalize.pkl",
        model_path.parent / "live_vecnormalize.pkl",
        model_path.with_name(model_path.stem + "_vecnormalize.pkl"),
        Path(r".\models_web\hybrid\best_vecnormalize.pkl"),
        Path(r".\models_web\hybrid\final_vecnormalize.pkl"),
        Path(r".\models_web\hybrid\best_vecnormalize_smoke_liveonly.pkl"),
        Path(r".\models_web\hybrid\best_vecnormalize_smoke.pkl"),
    ]

    # SB3 checkpoint format support.
    m = re.match(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps$", model_path.stem)
    if m:
        prefix = m.group("prefix")
        steps = m.group("steps")
        candidates.extend(
            [
                model_path.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl",
                model_path.parent / f"{prefix}_{steps}_steps.pkl",
                model_path.parent.parent / "live_vecnormalize.pkl",
                model_path.parent.parent / "final_vecnormalize.pkl",
            ]
        )
        step_n = int(steps)
        nearest_ckpt = []
        for p in model_path.parent.glob(f"{prefix}_vecnormalize_*_steps.pkl"):
            mm = re.match(rf"^{re.escape(prefix)}_vecnormalize_(\d+)_steps$", p.stem)
            if mm:
                nearest_ckpt.append((abs(int(mm.group(1)) - step_n), int(mm.group(1)), p))
        if nearest_ckpt:
            nearest_ckpt.sort(key=lambda x: (x[0], -x[1]))
            candidates.append(nearest_ckpt[0][2])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_eval_env(args: argparse.Namespace, vecnorm_path: Path | None):
    base_env = DummyVecEnv(
        [
            lambda: TapCapStateEnv(
                url=args.url,
                headless=False,
                step_dt=args.step_dt,
                state_stack=args.state_stack,
                action_repeat=args.action_repeat,
                render_mode="human",
            )
        ]
    )
    if vecnorm_path is not None and vecnorm_path.exists():
        wrapped_env = VecNormalize.load(str(vecnorm_path), base_env)
        wrapped_env.training = False
        wrapped_env.norm_reward = False
        return wrapped_env
    return base_env


def main() -> int:
    args = parse_args()
    wait_for_url(args.url)
    device = resolve_device(args.device)

    model_path = resolve_model(Path(args.model_path))
    requested_vecnorm_path = Path(args.vecnorm_path)
    vecnorm_path = resolve_vecnorm(requested_vecnorm_path, model_path)

    print(f"Loading model: {model_path}")
    print(f"Device: {device}")
    if vecnorm_path is not None and vecnorm_path.exists():
        print(f"Using vecnormalize: {vecnorm_path}")
    else:
        print("VecNormalize file not found, running without normalization wrapper.")

    eval_env = build_eval_env(args, vecnorm_path)
    model = PPO.load(str(model_path), device=device)
    model_mtime = file_mtime(model_path)
    vecnorm_mtime = file_mtime(vecnorm_path) if vecnorm_path is not None else -1.0
    last_reload_check = time.time()
    obs = eval_env.reset()
    episode = 1
    episode_reward = 0.0
    episode_rewards: deque[float] = deque(maxlen=50)
    episode_passes: deque[float] = deque(maxlen=50)
    reason_counter: Counter[str] = Counter()
    last_status = time.time()

    print("Running live PPO policy on TapCap. Press Ctrl+C to stop.")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            episode_reward += float(reward[0])
            if bool(done[0]):
                meta = info[0]
                print(
                    f"Episode {episode}: reward={episode_reward:.3f}, "
                    f"passes={meta.get('episode_passes', 'n/a')}, reason={meta.get('done_reason', 'n/a')}"
                )
                episode_rewards.append(float(episode_reward))
                episode_passes.append(float(meta.get("episode_passes", 0.0)))
                reason = str(meta.get("done_reason", "")).strip()
                if reason:
                    reason_counter[reason] += 1
                episode += 1
                episode_reward = 0.0
                if args.auto_reload_model and (time.time() - last_reload_check) >= max(1.0, args.reload_seconds):
                    last_reload_check = time.time()
                    candidate_model_path = resolve_model(Path(args.model_path))
                    candidate_vecnorm_path = resolve_vecnorm(Path(args.vecnorm_path), candidate_model_path)
                    candidate_model_mtime = file_mtime(candidate_model_path)
                    candidate_vecnorm_mtime = (
                        file_mtime(candidate_vecnorm_path) if candidate_vecnorm_path is not None else -1.0
                    )
                    if (
                        candidate_model_path != model_path
                        or candidate_model_mtime != model_mtime
                        or candidate_vecnorm_mtime != vecnorm_mtime
                    ):
                        print(f"Reloading model from disk: {candidate_model_path}")
                        candidate_env = None
                        try:
                            candidate_env = build_eval_env(args, candidate_vecnorm_path)
                            candidate_model = PPO.load(str(candidate_model_path), device=device)
                            candidate_obs = candidate_env.reset()

                            eval_env.close()
                            eval_env = candidate_env
                            candidate_env = None
                            model = candidate_model
                            obs = candidate_obs
                            model_path = candidate_model_path
                            vecnorm_path = candidate_vecnorm_path
                            model_mtime = candidate_model_mtime
                            vecnorm_mtime = candidate_vecnorm_mtime
                            print("Reload complete.")
                            continue
                        except Exception as exc:  # noqa: BLE001
                            print(f"Reload skipped due to error: {exc}")
                            if candidate_env is not None:
                                candidate_env.close()
                obs = eval_env.reset()

            if (time.time() - last_status) >= max(5.0, args.status_interval_seconds):
                mean_reward = (sum(episode_rewards) / len(episode_rewards)) if episode_rewards else 0.0
                mean_passes = (sum(episode_passes) / len(episode_passes)) if episode_passes else 0.0
                print(
                    f"[status] episodes={episode - 1} mean_reward={mean_reward:.3f} "
                    f"mean_passes={mean_passes:.2f} reasons={dict(reason_counter)} model={model_path.name}"
                )
                last_status = time.time()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
