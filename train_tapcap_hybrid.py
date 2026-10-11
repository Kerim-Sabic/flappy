from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from local_tapcap_state_env import LocalTapCapStateEnv
from tapcap_state_env import TapCapStateEnv


class EntCoefDecayCallback(BaseCallback):
    def __init__(self, start: float, end: float, total_stage_timesteps: int) -> None:
        super().__init__(verbose=0)
        self.start = float(start)
        self.end = float(end)
        self.total_stage_timesteps = max(1, int(total_stage_timesteps))
        self._stage_start_steps = 0

    def _on_training_start(self) -> None:
        self._stage_start_steps = int(self.model.num_timesteps)
        self.model.ent_coef = self.start

    def _on_step(self) -> bool:
        elapsed = max(0, int(self.model.num_timesteps) - self._stage_start_steps)
        progress = min(1.0, float(elapsed) / float(self.total_stage_timesteps))
        current = self.start + ((self.end - self.start) * progress)
        self.model.ent_coef = float(current)
        return True


class TrainingHealthCallback(BaseCallback):
    def __init__(
        self,
        interval_seconds: float,
        output_path: Path,
        stage_name: str,
        trainer_id: str,
    ) -> None:
        super().__init__(verbose=0)
        self.interval_seconds = max(5.0, float(interval_seconds))
        self.output_path = output_path
        self.stage_name = stage_name
        self.trainer_id = trainer_id

        self._start_wall = 0.0
        self._last_emit_wall = 0.0
        self._start_timesteps = 0
        self._num_envs = 1

        self._current_ep_rewards: list[float] = []
        self._current_ep_lengths: list[int] = []
        self._episode_rewards: deque[float] = deque(maxlen=200)
        self._episode_lengths: deque[int] = deque(maxlen=200)
        self._episodes_finished = 0

        self._interval_reason_counter: Counter[str] = Counter()
        self._reason_window: deque[str] = deque(maxlen=128)
        self._interval_flip_ratios: list[float] = []
        self._interval_refresh_counts: list[float] = []
        self._interval_restart_clicks: list[float] = []
        self._interval_runtime_recoveries: list[float] = []

    @staticmethod
    def _as_float_or_none(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._start_wall = time.time()
        self._last_emit_wall = self._start_wall
        self._start_timesteps = int(self.model.num_timesteps)
        self._num_envs = int(getattr(self.training_env, "num_envs", 1))
        self._current_ep_rewards = [0.0 for _ in range(self._num_envs)]
        self._current_ep_lengths = [0 for _ in range(self._num_envs)]

    def _emit(self) -> None:
        now = time.time()
        elapsed = max(1e-6, now - self._start_wall)
        timesteps = int(self.model.num_timesteps)
        fps = float((timesteps - self._start_timesteps) / elapsed)

        mean_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0
        mean_length = float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0

        cuda_alloc_mb = 0.0
        cuda_max_alloc_mb = 0.0
        if torch.cuda.is_available() and "cuda" in str(self.model.device):
            try:
                cuda_alloc_mb = float(torch.cuda.memory_allocated() / (1024.0 * 1024.0))
                cuda_max_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            except Exception:
                cuda_alloc_mb = 0.0
                cuda_max_alloc_mb = 0.0

        reason_hist_interval = dict(self._interval_reason_counter)
        reason_hist_window = dict(Counter(self._reason_window))

        payload = {
            "timestamp_unix": now,
            "stage": self.stage_name,
            "trainer": self.trainer_id,
            "timesteps": timesteps,
            "fps": fps,
            "episodes_finished": int(self._episodes_finished),
            "mean_reward_recent": mean_reward,
            "mean_ep_len_recent": mean_length,
            "done_reason_hist_interval": reason_hist_interval,
            "done_reason_hist_window": reason_hist_window,
            "mean_flip_ratio": float(np.mean(self._interval_flip_ratios)) if self._interval_flip_ratios else 0.0,
            "mean_refresh_count": float(np.mean(self._interval_refresh_counts)) if self._interval_refresh_counts else 0.0,
            "mean_restart_clicks": float(np.mean(self._interval_restart_clicks)) if self._interval_restart_clicks else 0.0,
            "mean_runtime_recovery_count": (
                float(np.mean(self._interval_runtime_recoveries)) if self._interval_runtime_recoveries else 0.0
            ),
            "cuda_mem_alloc_mb": cuda_alloc_mb,
            "cuda_mem_max_alloc_mb": cuda_max_alloc_mb,
            "device": str(self.model.device),
        }

        print(
            "[health] "
            f"stage={self.stage_name} ts={timesteps} fps={fps:.1f} "
            f"ep={self._episodes_finished} mean_r={mean_reward:.3f} mean_len={mean_length:.1f} "
            f"reasons={reason_hist_interval} cuda_mb={cuda_alloc_mb:.1f}/{cuda_max_alloc_mb:.1f}"
        )

        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

        self._interval_reason_counter.clear()
        self._interval_flip_ratios.clear()
        self._interval_refresh_counts.clear()
        self._interval_restart_clicks.clear()
        self._interval_runtime_recoveries.clear()
        self._last_emit_wall = now

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if not isinstance(infos, (list, tuple)):
            infos = []

        rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1) if rewards is not None else np.zeros((0,))
        dones_arr = np.asarray(dones, dtype=bool).reshape(-1) if dones is not None else np.zeros((0,), dtype=bool)

        for idx in range(self._num_envs):
            reward_i = float(rewards_arr[idx]) if idx < rewards_arr.shape[0] else 0.0
            done_i = bool(dones_arr[idx]) if idx < dones_arr.shape[0] else False
            info_i = infos[idx] if idx < len(infos) and isinstance(infos[idx], dict) else {}

            self._current_ep_rewards[idx] += reward_i
            self._current_ep_lengths[idx] += 1

            flip = self._as_float_or_none(info_i.get("flip_ratio"))
            if flip is not None:
                self._interval_flip_ratios.append(flip)

            refresh_count = self._as_float_or_none(info_i.get("refresh_count"))
            if refresh_count is not None:
                self._interval_refresh_counts.append(refresh_count)

            restart_clicks = self._as_float_or_none(info_i.get("restart_clicks"))
            if restart_clicks is not None:
                self._interval_restart_clicks.append(restart_clicks)

            runtime_recovery_count = self._as_float_or_none(info_i.get("runtime_recovery_count"))
            if runtime_recovery_count is not None:
                self._interval_runtime_recoveries.append(runtime_recovery_count)

            if done_i:
                self._episodes_finished += 1
                self._episode_rewards.append(self._current_ep_rewards[idx])
                self._episode_lengths.append(self._current_ep_lengths[idx])

                reason = str(info_i.get("done_reason", "")).strip()
                if reason:
                    self._interval_reason_counter[reason] += 1
                    self._reason_window.append(reason)

                self._current_ep_rewards[idx] = 0.0
                self._current_ep_lengths[idx] = 0

        if (time.time() - self._last_emit_wall) >= self.interval_seconds:
            self._emit()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid trainer: local domain-randomized pretrain + live TapCap fine-tune "
            "with parallel envs and consistency-focused reward shaping."
        )
    )
    parser.add_argument("--stage", choices=["local", "live", "hybrid"], default="hybrid")
    parser.add_argument("--local-url", type=str, default="http://127.0.0.1:5173")
    parser.add_argument("--live-url", type=str, default="https://play.tapcapgame.com/")
    parser.add_argument("--total-timesteps-local", type=int, default=1_500_000)
    parser.add_argument("--total-timesteps-live", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-envs-local", type=int, default=0)
    parser.add_argument("--n-envs-live", type=int, default=1)
    parser.add_argument("--profile", choices=["none", "tapcap_live_strict"], default="none")
    parser.add_argument("--vec-mode", choices=["auto", "dummy", "subproc"], default="auto")
    parser.add_argument("--state-stack", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--min-action-hold", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainer-id", type=str, default="solo")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument("--subproc", action="store_true", help="Legacy alias: same as --vec-mode subproc")
    parser.add_argument(
        "--auto-fallback-envs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically reduce env count if env startup fails.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-dt", type=float, default=0.05)
    parser.add_argument("--dr-level", choices=["low", "moderate", "high"], default="moderate")
    parser.add_argument("--curriculum-total-resets", type=int, default=2500)
    parser.add_argument("--model-dir", type=str, default=r".\models_web\hybrid")
    parser.add_argument("--log-dir", type=str, default=r".\logs_web\hybrid")
    parser.add_argument("--resume", type=str, default="", help="Optional model .zip to resume from.")
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef-local", type=float, default=0.01)
    parser.add_argument("--ent-coef-live", type=float, default=0.003)
    parser.add_argument("--ent-coef-live-end", type=float, default=0.0005)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--norm-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--net-arch", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--health-log-interval", type=float, default=45.0)
    parser.add_argument(
        "--health-jsonl-path",
        type=str,
        default="",
        help="Optional path template for health JSONL. Supports {trainer} and {stage}.",
    )
    return parser.parse_args()


def linear_schedule(initial: float):
    def _schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * float(initial)

    return _schedule


def apply_profile_overrides(args: argparse.Namespace) -> None:
    if args.subproc and args.vec_mode == "auto":
        args.vec_mode = "subproc"

    if args.profile != "tapcap_live_strict":
        return

    args.stage = "live"
    args.vec_mode = "dummy"
    args.n_envs_live = 1
    args.state_stack = 4
    args.action_repeat = 2
    args.min_action_hold = 2
    args.step_dt = 0.05
    args.n_steps = 1024
    args.batch_size = 512
    args.gamma = 0.995
    args.gae_lambda = 0.95
    args.lr = 3e-4
    args.target_kl = 0.02
    args.ent_coef_live = 0.003
    args.ent_coef_live_end = 0.0005
    args.net_arch = "large"
    # Keep eval/checkpoint cadence low in live-only mode so promotion has
    # usable artifacts early even with low on-site FPS.
    args.eval_freq = 64
    args.eval_episodes = 2
    args.save_freq = 256
    args.headless = True
    print("Applied profile: tapcap_live_strict (live-only, stability-focused defaults).")


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def resolve_use_subproc(args: argparse.Namespace, stage_name: str, n_envs: int) -> bool:
    if n_envs <= 1:
        return False

    mode = args.vec_mode
    if mode == "dummy":
        return False
    if mode == "subproc":
        return True

    # auto mode: keep live stable on Windows.
    if platform.system() == "Windows" and stage_name == "live":
        return False
    return True


def build_vec_env(env_fns: list, use_subproc: bool):
    if use_subproc and len(env_fns) > 1:
        start_method = "spawn" if platform.system() == "Windows" else "fork"
        return SubprocVecEnv(env_fns, start_method=start_method)
    return DummyVecEnv(env_fns)


def make_local_env_fn(
    rank: int,
    base_seed: int,
    local_url: str,
    headless: bool,
    step_dt: float,
    state_stack: int,
    action_repeat: int,
    dr_level: str,
    curriculum_total_resets: int,
):
    def _init():
        env = LocalTapCapStateEnv(
            url=local_url,
            headless=headless,
            step_dt=step_dt,
            state_stack=state_stack,
            action_repeat=action_repeat,
            dr_level=dr_level,
            curriculum_total_resets=curriculum_total_resets,
        )
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env

    return _init


def make_live_env_fn(
    rank: int,
    base_seed: int,
    live_url: str,
    headless: bool,
    step_dt: float,
    state_stack: int,
    action_repeat: int,
    min_action_hold: int,
):
    def _init():
        env = TapCapStateEnv(
            url=live_url,
            headless=headless,
            step_dt=step_dt,
            state_stack=state_stack,
            action_repeat=action_repeat,
            min_action_hold=min_action_hold,
        )
        env = Monitor(env)
        env.reset(seed=base_seed + rank)
        return env

    return _init


def build_stage_envs(
    stage_name: str,
    n_envs: int,
    seed: int,
    args: argparse.Namespace,
):
    if stage_name == "local":
        env_fns = [
            make_local_env_fn(
                rank=i,
                base_seed=seed,
                local_url=args.local_url,
                headless=args.headless,
                step_dt=args.step_dt,
                state_stack=args.state_stack,
                action_repeat=args.action_repeat,
                dr_level=args.dr_level,
                curriculum_total_resets=args.curriculum_total_resets,
            )
            for i in range(n_envs)
        ]
        eval_fn = make_local_env_fn(
            rank=10_000,
            base_seed=seed,
            local_url=args.local_url,
            headless=True,
            step_dt=args.step_dt,
            state_stack=args.state_stack,
            action_repeat=args.action_repeat,
            dr_level=args.dr_level,
            curriculum_total_resets=args.curriculum_total_resets,
        )
    else:
        env_fns = [
            make_live_env_fn(
                rank=i,
                base_seed=seed,
                live_url=args.live_url,
                headless=args.headless,
                step_dt=args.step_dt,
                state_stack=args.state_stack,
                action_repeat=args.action_repeat,
                min_action_hold=args.min_action_hold,
            )
            for i in range(n_envs)
        ]
        eval_fn = make_live_env_fn(
            rank=10_000,
            base_seed=seed,
            live_url=args.live_url,
            headless=True,
            step_dt=args.step_dt,
            state_stack=args.state_stack,
            action_repeat=args.action_repeat,
            min_action_hold=args.min_action_hold,
        )

    use_subproc = resolve_use_subproc(args=args, stage_name=stage_name, n_envs=n_envs)
    train_env = build_vec_env(env_fns, use_subproc=use_subproc)
    eval_env = build_vec_env([eval_fn], use_subproc=False)
    return train_env, eval_env, use_subproc


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_wrap_vecnormalize(
    train_env,
    eval_env,
    vecnorm_path: Path | None,
    norm_obs: bool,
    norm_reward: bool,
):
    if vecnorm_path is not None and vecnorm_path.exists():
        train_env = VecNormalize.load(str(vecnorm_path), train_env)
        train_env.training = True
        train_env.norm_reward = norm_reward
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        return train_env, eval_env

    train_env = VecNormalize(
        train_env,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=norm_obs,
        norm_reward=False,
        training=False,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    return train_env, eval_env


def resolve_health_log_path(args: argparse.Namespace, stage_name: str, stage_log_dir: Path) -> Path:
    if not args.health_jsonl_path:
        return stage_log_dir / "health.jsonl"
    rendered = args.health_jsonl_path.format(trainer=args.trainer_id, stage=stage_name)
    return Path(rendered)


def resolve_net_arch(preset: str) -> list[int]:
    if preset == "small":
        return [128, 128]
    if preset == "large":
        return [512, 512, 256]
    return [256, 256, 128]


def train_stage(
    stage_name: str,
    total_timesteps: int,
    n_envs: int,
    args: argparse.Namespace,
    model_dir: Path,
    log_dir: Path,
    initial_model_path: Path | None,
    initial_vecnorm_path: Path | None,
    seed: int,
    device: str,
) -> tuple[Path, Path]:
    stage_model_dir = model_dir / stage_name
    stage_log_dir = log_dir / stage_name
    stage_ckpt_dir = stage_model_dir / "checkpoints"
    stage_eval_dir = stage_log_dir / "eval"
    stage_model_dir.mkdir(parents=True, exist_ok=True)
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    stage_ckpt_dir.mkdir(parents=True, exist_ok=True)
    stage_eval_dir.mkdir(parents=True, exist_ok=True)

    active_n_envs = max(1, int(n_envs))
    train_env = None
    eval_env = None
    model = None
    use_subproc = False
    last_error: Exception | None = None

    while active_n_envs >= 1:
        try:
            train_env, eval_env, use_subproc = build_stage_envs(
                stage_name=stage_name,
                n_envs=active_n_envs,
                seed=seed,
                args=args,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()
            train_env = None
            eval_env = None
            if (not args.auto_fallback_envs) or active_n_envs == 1:
                raise RuntimeError(
                    f"Failed to start {stage_name} envs with n_envs={active_n_envs}: {exc}"
                ) from exc
            next_n = max(1, active_n_envs // 2)
            print(
                f"[{stage_name}] env startup failed at n_envs={active_n_envs}: {exc}. "
                f"Retrying with n_envs={next_n}."
            )
            active_n_envs = next_n

    if train_env is None or eval_env is None:
        raise RuntimeError(f"Could not initialize {stage_name} environments. Last error: {last_error}")

    train_env, eval_env = maybe_wrap_vecnormalize(
        train_env=train_env,
        eval_env=eval_env,
        vecnorm_path=initial_vecnorm_path,
        norm_obs=args.norm_obs,
        norm_reward=args.norm_reward,
    )

    stage_ent_coef = args.ent_coef_local if stage_name == "local" else args.ent_coef_live
    checkpoint_freq = max(args.save_freq // active_n_envs, 1)
    eval_freq = max(args.eval_freq // active_n_envs, 1)
    health_path = resolve_health_log_path(args, stage_name, stage_log_dir)
    print(
        f"[{stage_name}] n_envs={active_n_envs}, vec_mode={args.vec_mode}, "
        f"use_subproc={use_subproc}, steps={total_timesteps}, "
        f"checkpoint_freq={checkpoint_freq}, eval_freq={eval_freq}, ent_coef={stage_ent_coef}, "
        f"health_log={health_path}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(stage_ckpt_dir),
        name_prefix=f"ppo_{stage_name}_{args.trainer_id}",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(stage_model_dir),
        log_path=str(stage_eval_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=args.eval_episodes,
    )
    health_callback = TrainingHealthCallback(
        interval_seconds=args.health_log_interval,
        output_path=health_path,
        stage_name=stage_name,
        trainer_id=args.trainer_id,
    )

    callback_items: list[BaseCallback] = [checkpoint_callback, eval_callback, health_callback]
    if stage_name == "live":
        callback_items.append(
            EntCoefDecayCallback(
                start=float(stage_ent_coef),
                end=float(args.ent_coef_live_end),
                total_stage_timesteps=total_timesteps,
            )
        )
    callbacks = CallbackList(callback_items)

    policy_kwargs = dict(net_arch=resolve_net_arch(args.net_arch))

    if initial_model_path is not None and initial_model_path.exists():
        print(f"[{stage_name}] Resuming from model: {initial_model_path}")
        model = PPO.load(str(initial_model_path), env=train_env, device=device)
        model.ent_coef = float(stage_ent_coef)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=seed,
            device=device,
            tensorboard_log=str(stage_log_dir),
            learning_rate=linear_schedule(args.lr),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=stage_ent_coef,
            clip_range=args.clip_range,
            target_kl=args.target_kl,
            policy_kwargs=policy_kwargs,
        )
        model.ent_coef = float(stage_ent_coef)

    policy_device = next(model.policy.parameters()).device
    print(f"[{stage_name}] model policy device: {policy_device}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=f"{stage_name}_{args.trainer_id}",
        progress_bar=False,
        reset_num_timesteps=False,
    )

    final_model_path = stage_model_dir / f"{stage_name}_final_model"
    model.save(str(final_model_path))
    print(f"[{stage_name}] saved model: {final_model_path.with_suffix('.zip')}")

    vecnorm_path = stage_model_dir / f"{stage_name}_vecnormalize.pkl"
    if isinstance(train_env, VecNormalize):
        train_env.save(str(vecnorm_path))
        print(f"[{stage_name}] saved vecnormalize: {vecnorm_path}")
    else:
        vecnorm_path = stage_model_dir / f"{stage_name}_vecnormalize.none"
        vecnorm_path.write_text("no_vecnormalize", encoding="utf-8")

    eval_env.close()
    train_env.close()
    return final_model_path.with_suffix(".zip"), vecnorm_path


def main() -> int:
    args = parse_args()
    apply_profile_overrides(args)

    if args.n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    n_envs_local = args.n_envs_local if args.n_envs_local > 0 else args.n_envs
    n_envs_live = args.n_envs_live if args.n_envs_live > 0 else args.n_envs

    model_dir = Path(args.model_dir) / args.trainer_id
    log_dir = Path(args.log_dir) / args.trainer_id
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in {"local", "hybrid"}:
        wait_for_url(args.local_url)
    if args.stage in {"live", "hybrid"}:
        wait_for_url(args.live_url)

    set_global_seeds(args.seed)
    device = resolve_device(args.device)

    use_subproc_local = resolve_use_subproc(args=args, stage_name="local", n_envs=n_envs_local)
    use_subproc_live = resolve_use_subproc(args=args, stage_name="live", n_envs=n_envs_live)

    print(f"Trainer: {args.trainer_id}")
    print(f"Stage: {args.stage}")
    print(f"Device: {device}")
    print(f"Vec mode: {args.vec_mode}")
    print(f"Live env count: {n_envs_live} (expected browsers ~ {n_envs_live})")
    print(f"Subproc local/live: {use_subproc_local}/{use_subproc_live}")
    print(f"Network preset: {args.net_arch} -> {resolve_net_arch(args.net_arch)}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"CUDA[{idx}]: {torch.cuda.get_device_name(idx)}")

    start_model = Path(args.resume) if args.resume else None
    start_vecnorm: Path | None = None

    if start_model is not None and start_model.exists():
        possible = start_model.with_name(start_model.stem.replace(".zip", "") + "_vecnormalize.pkl")
        if possible.exists():
            start_vecnorm = possible

    latest_model: Path | None = start_model
    latest_vecnorm: Path | None = start_vecnorm

    if args.stage in {"local", "hybrid"}:
        latest_model, latest_vecnorm = train_stage(
            stage_name="local",
            total_timesteps=args.total_timesteps_local,
            n_envs=n_envs_local,
            args=args,
            model_dir=model_dir,
            log_dir=log_dir,
            initial_model_path=latest_model,
            initial_vecnorm_path=latest_vecnorm,
            seed=args.seed,
            device=device,
        )

    if args.stage in {"live", "hybrid"}:
        live_seed = args.seed + 101
        latest_model, latest_vecnorm = train_stage(
            stage_name="live",
            total_timesteps=args.total_timesteps_live,
            n_envs=n_envs_live,
            args=args,
            model_dir=model_dir,
            log_dir=log_dir,
            initial_model_path=latest_model,
            initial_vecnorm_path=latest_vecnorm,
            seed=live_seed,
            device=device,
        )

    if latest_model is None:
        raise RuntimeError("No model was trained.")

    final_model = model_dir / "final_model.zip"
    final_vecnorm = model_dir / "final_vecnormalize.pkl"
    final_model.write_bytes(latest_model.read_bytes())
    if latest_vecnorm is not None and latest_vecnorm.exists() and latest_vecnorm.suffix == ".pkl":
        final_vecnorm.write_bytes(latest_vecnorm.read_bytes())
    print(f"Final model: {final_model}")
    if final_vecnorm.exists():
        print(f"Final vecnormalize: {final_vecnorm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
