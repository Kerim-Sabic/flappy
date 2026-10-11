from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from local_tapcap_state_env import LocalTapCapStateEnv
from tapcap_state_env import TapCapStateEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate TapCap PPO models and promote best by consistency metric."
    )
    parser.add_argument(
        "--model-glob",
        type=str,
        default=r".\models_web\hybrid\*\final_model.zip",
        help="Glob pattern for candidate model zip files.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="If > 0, evaluate only the newest N candidate models by mtime.",
    )
    parser.add_argument("--local-url", type=str, default="http://127.0.0.1:5173")
    parser.add_argument("--live-url", type=str, default="https://play.tapcapgame.com/")
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-suite", choices=["local", "live", "both"], default="live")
    parser.add_argument("--selection-metric", choices=["consistency"], default="consistency")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--state-stack", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--logs-root", type=str, default=r".\logs_web\hybrid")
    parser.add_argument(
        "--min-eval-episodes",
        type=int,
        default=0,
        help="Skip candidate models whose inferred trainer eval history has fewer episodes than this threshold.",
    )
    parser.add_argument("--output-best-model", type=str, default=r".\models_web\hybrid\best_model.zip")
    parser.add_argument("--output-best-vecnorm", type=str, default=r".\models_web\hybrid\best_vecnormalize.pkl")
    parser.add_argument("--output-report", type=str, default=r".\models_web\hybrid\selection_report.json")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def find_candidates(pattern: str) -> list[Path]:
    matches = [Path(p) for p in glob.glob(pattern)]
    return sorted([p for p in matches if p.exists()])


def find_vecnorm_for_model(model_path: Path) -> Path | None:
    checks: list[Path] = [
        model_path.parent / "final_vecnormalize.pkl",
        model_path.with_name(model_path.stem + "_vecnormalize.pkl"),
        model_path.parent / "live_vecnormalize.pkl",
        model_path.parent / "local_vecnormalize.pkl",
    ]

    # SB3 checkpoint callback format: <prefix>_<steps>_steps.zip
    # with vecnorm saved as: <prefix>_vecnormalize_<steps>_steps.pkl
    m = re.match(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps$", model_path.stem)
    if m:
        prefix = m.group("prefix")
        steps = m.group("steps")
        checks.extend(
            [
                model_path.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl",
                model_path.parent / f"{prefix}_{steps}_steps.pkl",
                model_path.parent.parent / "live_vecnormalize.pkl",
                model_path.parent.parent / "final_vecnormalize.pkl",
            ]
        )
        # If exact vecnormalize snapshot is not present, fall back to nearest
        # checkpoint vecnormalize from the same prefix.
        step_n = int(steps)
        vecnorm_candidates = []
        for p in model_path.parent.glob(f"{prefix}_vecnormalize_*_steps.pkl"):
            mm = re.match(rf"^{re.escape(prefix)}_vecnormalize_(\d+)_steps$", p.stem)
            if mm:
                vecnorm_candidates.append((abs(int(mm.group(1)) - step_n), int(mm.group(1)), p))
        if vecnorm_candidates:
            vecnorm_candidates.sort(key=lambda x: (x[0], -x[1]))
            checks.append(vecnorm_candidates[0][2])

    seen: set[str] = set()
    for p in checks:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p
    return None


def infer_trainer_id_from_model_path(model_path: Path) -> str:
    for part in model_path.parts:
        if part.lower().startswith("trainer_"):
            return part
    return model_path.parent.name


def infer_eval_episode_count(model_path: Path, logs_root: Path) -> int:
    trainer_id = infer_trainer_id_from_model_path(model_path)
    eval_npz = logs_root / trainer_id / "live" / "eval" / "evaluations.npz"
    if not eval_npz.exists():
        return 0
    try:
        data = np.load(str(eval_npz))
        results = data.get("results")
        if results is None:
            return 0
        arr = np.asarray(results)
        if arr.ndim == 2:
            return int(arr.shape[0] * arr.shape[1])
        return int(arr.size)
    except Exception:
        return 0


def evaluate_single_suite(
    model_path: Path,
    vecnorm_path: Path | None,
    suite: str,
    episodes: int,
    args: argparse.Namespace,
    device: str,
) -> dict[str, float]:
    if suite == "local":
        def _env_factory():
            return Monitor(
                LocalTapCapStateEnv(
                    url=args.local_url,
                    headless=args.headless,
                    step_dt=args.step_dt,
                    action_repeat=args.action_repeat,
                    state_stack=args.state_stack,
                    dr_level="moderate",
                    curriculum_total_resets=2500,
                )
            )
    else:
        def _env_factory():
            return Monitor(
                TapCapStateEnv(
                    url=args.live_url,
                    headless=args.headless,
                    step_dt=args.step_dt,
                    action_repeat=args.action_repeat,
                    state_stack=args.state_stack,
                )
            )

    raw_env = DummyVecEnv([_env_factory])
    eval_env = raw_env
    if vecnorm_path is not None and vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), raw_env)
        eval_env.training = False
        eval_env.norm_reward = False

    model = PPO.load(str(model_path), device=device)
    rewards: list[float] = []
    passes: list[float] = []
    early_deaths = 0

    obs = eval_env.reset()
    ep_reward = 0.0
    while len(rewards) < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        ep_reward += float(reward[0])
        if bool(done[0]):
            meta = info[0]
            rewards.append(ep_reward)
            passes.append(float(meta.get("episode_passes", meta.get("score", 0.0))))
            if float(meta.get("episode_steps", 0.0)) < 45.0:
                early_deaths += 1
            ep_reward = 0.0
            obs = eval_env.reset()

    eval_env.close()

    rewards_np = np.asarray(rewards, dtype=np.float32)
    passes_np = np.asarray(passes, dtype=np.float32)
    median_reward = float(np.median(rewards_np))
    peak_reward = float(np.max(rewards_np))
    std_reward = float(np.std(rewards_np))
    median_passes = float(np.median(passes_np)) if passes_np.size else 0.0
    early_rate = float(early_deaths / max(1, episodes))

    consistency_metric = (
        median_reward
        + (0.25 * peak_reward)
        + (0.65 * median_passes)
        - (0.35 * std_reward)
        - (1.25 * early_rate)
    )

    return {
        "metric": consistency_metric,
        "median_reward": median_reward,
        "peak_reward": peak_reward,
        "std_reward": std_reward,
        "median_passes": median_passes,
        "early_death_rate": early_rate,
    }


def evaluate_model(model_path: Path, vecnorm_path: Path | None, args: argparse.Namespace, device: str) -> dict:
    if args.eval_suite == "local":
        local = evaluate_single_suite(model_path, vecnorm_path, "local", args.eval_episodes, args, device)
        final_metric = local["metric"]
        return {"model": str(model_path), "vecnorm": str(vecnorm_path) if vecnorm_path else "", "local": local, "metric": final_metric}

    if args.eval_suite == "live":
        live = evaluate_single_suite(model_path, vecnorm_path, "live", args.eval_episodes, args, device)
        final_metric = live["metric"]
        return {"model": str(model_path), "vecnorm": str(vecnorm_path) if vecnorm_path else "", "live": live, "metric": final_metric}

    local = evaluate_single_suite(model_path, vecnorm_path, "local", args.eval_episodes, args, device)
    live = evaluate_single_suite(model_path, vecnorm_path, "live", args.eval_episodes, args, device)
    final_metric = (0.30 * local["metric"]) + (0.70 * live["metric"])
    return {
        "model": str(model_path),
        "vecnorm": str(vecnorm_path) if vecnorm_path else "",
        "local": local,
        "live": live,
        "metric": final_metric,
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    logs_root = Path(args.logs_root)
    candidates = find_candidates(args.model_glob)
    if not candidates:
        raise FileNotFoundError(f"No candidate models found for pattern: {args.model_glob}")

    if args.max_candidates > 0 and len(candidates) > args.max_candidates:
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[: args.max_candidates]

    print(f"Found {len(candidates)} candidates.")
    print(f"Eval suite: {args.eval_suite}, episodes/model: {args.eval_episodes}, device: {device}")

    report_rows = []
    for idx, model_path in enumerate(candidates, start=1):
        eval_episode_count = infer_eval_episode_count(model_path, logs_root)
        if args.min_eval_episodes > 0 and eval_episode_count < args.min_eval_episodes:
            print(
                f"[{idx}/{len(candidates)}] skipping {model_path} "
                f"(eval episodes {eval_episode_count} < min {args.min_eval_episodes})"
            )
            continue

        vecnorm_path = find_vecnorm_for_model(model_path)
        print(f"[{idx}/{len(candidates)}] evaluating: {model_path}")
        row = evaluate_model(model_path, vecnorm_path, args, device=device)
        row["eval_episode_count"] = int(eval_episode_count)
        report_rows.append(row)
        print(f"  metric={row['metric']:.4f}")

    if not report_rows:
        raise RuntimeError("No eligible models remained after min-eval-episodes filtering.")

    report_rows.sort(key=lambda x: float(x["metric"]), reverse=True)
    best = report_rows[0]
    best_model_path = Path(best["model"])
    best_vecnorm_path = Path(best["vecnorm"]) if best.get("vecnorm") else None

    output_best_model = Path(args.output_best_model)
    output_best_model.parent.mkdir(parents=True, exist_ok=True)
    output_best_model.write_bytes(best_model_path.read_bytes())
    print(f"Promoted best model: {best_model_path} -> {output_best_model}")

    output_best_vecnorm = Path(args.output_best_vecnorm)
    if best_vecnorm_path is not None and best_vecnorm_path.exists():
        output_best_vecnorm.parent.mkdir(parents=True, exist_ok=True)
        output_best_vecnorm.write_bytes(best_vecnorm_path.read_bytes())
        print(f"Promoted vecnormalize: {best_vecnorm_path} -> {output_best_vecnorm}")

    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    top_n = max(1, int(args.top_n))
    top_dir = output_report.parent / "top_models"
    top_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(report_rows[:top_n], start=1):
        src = Path(row["model"])
        dst = top_dir / f"rank_{rank}_{src.name}"
        dst.write_bytes(src.read_bytes())

    payload = {
        "selection_metric": args.selection_metric,
        "eval_suite": args.eval_suite,
        "eval_episodes": args.eval_episodes,
        "top_n": top_n,
        "timestamp_unix": time_now_unix(),
        "best": best,
        "all": report_rows,
    }
    output_report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved selection report: {output_report}")
    return 0


def time_now_unix() -> float:
    import time

    return time.time()


if __name__ == "__main__":
    raise SystemExit(main())
