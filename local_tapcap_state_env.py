from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from tapcap_state import BASE_STATE_DIM, update_stacked_state


def _dr_code(level: str) -> float:
    low = level.lower().strip()
    if low == "low":
        return 0.2
    if low == "high":
        return 0.8
    return 0.5


@dataclass
class LocalTapCapConfig:
    url: str = "http://127.0.0.1:5173"
    headless: bool = True
    step_dt: float = 0.05
    action_repeat: int = 2
    state_stack: int = 4
    dr_level: str = "moderate"
    curriculum_total_resets: int = 2500
    timeout_ms: int = 15_000
    max_retries: int = 20
    retry_sleep_ms: float = 100.0
    viewport_width: int = 420
    viewport_height: int = 720


class LocalTapCapStateEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        url: str = "http://127.0.0.1:5173",
        headless: bool = True,
        step_dt: float = 0.05,
        action_repeat: int = 2,
        state_stack: int = 4,
        dr_level: str = "moderate",
        curriculum_total_resets: int = 2500,
        timeout_ms: int = 15_000,
        max_retries: int = 20,
        retry_sleep_ms: float = 100.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if step_dt <= 0:
            raise ValueError("step_dt must be > 0")
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if state_stack < 1:
            raise ValueError("state_stack must be >= 1")

        self.config = LocalTapCapConfig(
            url=url,
            headless=False if render_mode == "human" else headless,
            step_dt=step_dt,
            action_repeat=action_repeat,
            state_stack=state_stack,
            dr_level=dr_level,
            curriculum_total_resets=max(100, int(curriculum_total_resets)),
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_sleep_ms=retry_sleep_ms,
        )
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.config.state_stack * BASE_STATE_DIM,),
            dtype=np.float32,
        )

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._executor: ThreadPoolExecutor | None = None

        self._history: deque[np.ndarray] = deque(maxlen=self.config.state_stack)
        self._prev_state: dict[str, Any] | None = None
        self._last_score: float = 0.0
        self._prev_action: int = 0
        self._episode_steps: int = 0
        self._reset_count: int = 0

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="local_tapcap_env")

    def _run_on_worker(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_executor()
        assert self._executor is not None
        future = self._executor.submit(fn, *args, **kwargs)
        return future.result()

    def _ensure_browser(self) -> None:
        self._run_on_worker(self._ensure_browser_in_worker)

    def _ensure_browser_in_worker(self) -> None:
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        if self._browser is None:
            self._browser = self._playwright.chromium.launch(headless=self.config.headless)
        if self._context is None:
            self._context = self._browser.new_context(
                viewport={
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height,
                },
                device_scale_factor=1,
            )
        if self._page is None or self._page.is_closed():
            self._page = self._context.new_page()

    def _goto_game_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        self._page.goto(self.config.url, wait_until="domcontentloaded", timeout=self.config.timeout_ms)
        self._page.wait_for_function(
            "() => window.__rl && "
            "typeof window.__rl.reset === 'function' && "
            "typeof window.__rl.flap === 'function' && "
            "typeof window.__rl.getState === 'function' && "
            "typeof window.__rl.isDone === 'function'",
            timeout=self.config.timeout_ms,
        )

    def _wait_step_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        self._page.wait_for_timeout(self.config.step_dt * 1000.0)

    def _flap_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        self._page.evaluate("() => window.__rl.flap()")

    def _reset_with_curriculum_in_worker(self, seed: int | None, progress: float) -> dict[str, Any]:
        self._ensure_browser_in_worker()
        assert self._page is not None
        seed_value = int(seed) if seed is not None else None
        payload = {
            "seed": seed_value,
            "drLevel": self.config.dr_level,
            "curriculumProgress": float(np.clip(progress, 0.0, 1.0)),
        }
        self._page.evaluate("(cfg) => window.__rl.reset(cfg)", payload)
        state = self._page.evaluate("() => window.__rl.getState()")
        return state

    def _get_state_in_worker(self) -> dict[str, Any]:
        self._ensure_browser_in_worker()
        assert self._page is not None
        return self._page.evaluate("() => window.__rl.getState()")

    def _is_done_in_worker(self) -> bool:
        self._ensure_browser_in_worker()
        assert self._page is not None
        return bool(self._page.evaluate("() => window.__rl.isDone()"))

    def _get_randomization_in_worker(self) -> dict[str, float]:
        self._ensure_browser_in_worker()
        assert self._page is not None
        raw = self._page.evaluate(
            "() => (window.__rl && typeof window.__rl.getRandomization === 'function') "
            "? window.__rl.getRandomization() : null"
        )
        if not isinstance(raw, dict):
            return {}
        out: dict[str, float] = {}
        for key, value in raw.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def _safe_worker_call(self, fn: Callable[..., Any], *args: Any) -> Any:
        last_error: Exception | None = None
        for _ in range(max(1, self.config.max_retries)):
            try:
                return self._run_on_worker(fn, *args)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"LocalTapCap env worker call failed: {last_error}") from last_error
        raise RuntimeError("LocalTapCap env worker call failed without error details.")

    def _state_to_vec(
        self,
        state: dict[str, Any],
        prev_state: dict[str, Any] | None,
        dr: dict[str, float],
        progress: float,
    ) -> np.ndarray:
        y = float(state.get("y", 0.0))
        vy = float(state.get("vy", 0.0))
        dx = float(state.get("nextPipeDx", 0.0))
        gap_dy = float(state.get("gapDy", 0.0))
        alive = 1.0 if bool(state.get("alive", True)) else 0.0

        width = max(1.0, float(self.config.viewport_width))
        height = max(1.0, float(self.config.viewport_height))

        prev_dx = dx if prev_state is None else float(prev_state.get("nextPipeDx", dx))
        obstacle_speed = (prev_dx - dx) / max(1e-6, self.config.step_dt)

        gravity = float(dr.get("gravity", 900.0))
        flap_velocity = float(abs(dr.get("flapVelocity", -300.0)))
        pipe_speed = float(dr.get("pipeSpeed", 150.0))
        pipe_gap = float(dr.get("pipeGap", 170.0))
        spawn_interval = float(dr.get("pipeSpawnInterval", 1.35))
        gap_margin = float(dr.get("gapMargin", 120.0))

        map_sig = np.array(
            [
                np.clip(gravity / 1400.0, 0.0, 1.0),
                np.clip(flap_velocity / 500.0, 0.0, 1.0),
                np.clip(pipe_speed / 260.0, 0.0, 1.0),
                np.clip(pipe_gap / 260.0, 0.0, 1.0),
                np.clip(spawn_interval / 2.2, 0.0, 1.0),
                np.clip(gap_margin / 220.0, 0.0, 1.0),
                np.clip(progress, 0.0, 1.0),
                _dr_code(self.config.dr_level),
            ],
            dtype=np.float32,
        )

        return np.array(
            [
                np.clip(y / height, 0.0, 1.0),
                np.clip(vy / 700.0, -3.0, 3.0),
                np.clip(dx / width, -1.5, 2.5),
                np.clip(gap_dy / height, -1.5, 1.5),
                np.clip(obstacle_speed / 500.0, -3.0, 3.0),
                alive,
                *map_sig.tolist(),
            ],
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options

        self._safe_worker_call(self._goto_game_in_worker)

        progress = float(min(1.0, self._reset_count / max(1, self.config.curriculum_total_resets)))
        state = self._safe_worker_call(self._reset_with_curriculum_in_worker, seed, progress)
        dr = self._safe_worker_call(self._get_randomization_in_worker)

        vec = self._state_to_vec(state, prev_state=None, dr=dr, progress=progress)
        self._history.clear()
        obs = update_stacked_state(self._history, vec, self.config.state_stack)

        self._prev_state = state
        self._last_score = float(state.get("score", 0.0))
        self._prev_action = 0
        self._episode_steps = 0
        self._reset_count += 1

        info = {
            "score": self._last_score,
            "curriculum_progress": progress,
            "dr_level": self.config.dr_level,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_int = int(action)
        reward = 0.0
        done = False
        latest_state: dict[str, Any] = self._prev_state or {}
        latest_dr: dict[str, float] = {}
        progress = float(min(1.0, self._reset_count / max(1, self.config.curriculum_total_resets)))

        if action_int != self._prev_action:
            reward -= 0.01
        self._prev_action = action_int

        for _ in range(self.config.action_repeat):
            if action_int == 1:
                self._safe_worker_call(self._flap_in_worker)

            self._safe_worker_call(self._wait_step_in_worker)
            latest_state = self._safe_worker_call(self._get_state_in_worker)
            latest_dr = self._safe_worker_call(self._get_randomization_in_worker)
            self._episode_steps += 1

            alive = bool(latest_state.get("alive", True))
            score = float(latest_state.get("score", self._last_score))
            score_delta = max(0.0, score - self._last_score)

            if alive:
                reward += 0.05
            if score_delta > 0:
                reward += 1.0 * score_delta
                reward += 0.5 * score_delta
            self._last_score = score

            gap_dy = float(latest_state.get("gapDy", 0.0))
            if abs(gap_dy) / max(1.0, float(self.config.viewport_height)) > 0.24:
                reward -= 0.02

            done_flag = self._safe_worker_call(self._is_done_in_worker)
            if done_flag or (not alive):
                done = True
                break

        if done:
            reward -= 1.0

        vec = self._state_to_vec(latest_state, prev_state=self._prev_state, dr=latest_dr, progress=progress)
        obs = update_stacked_state(self._history, vec, self.config.state_stack)
        self._prev_state = latest_state

        reward = float(np.clip(reward, -2.0, 2.0))
        info = {
            "score": self._last_score,
            "episode_steps": self._episode_steps,
            "curriculum_progress": progress,
            "dr_level": self.config.dr_level,
        }
        return obs, reward, done, False, info

    def render(self) -> None:
        return None

    def _close_in_worker(self) -> None:
        if self._page is not None:
            try:
                self._page.close()
            except Exception:  # noqa: BLE001
                pass
            self._page = None

        if self._context is not None:
            try:
                self._context.close()
            except Exception:  # noqa: BLE001
                pass
            self._context = None

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:  # noqa: BLE001
                pass
            self._browser = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:  # noqa: BLE001
                pass
            self._playwright = None

    def close(self) -> None:
        if self._executor is not None:
            try:
                self._run_on_worker(self._close_in_worker)
            except Exception:  # noqa: BLE001
                pass
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        super().close()

