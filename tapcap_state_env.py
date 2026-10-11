from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from tapcap_state import (
    BASE_STATE_DIM,
    TapCapDetection,
    build_state_vector,
    extract_tapcap_state,
    map_key_from_signature,
    parse_tapcap_detection,
    refresh_and_enter_run,
    tap,
    update_stacked_state,
)


@dataclass
class TapCapEnvConfig:
    url: str = "https://play.tapcapgame.com/"
    headless: bool = True
    step_dt: float = 0.05
    action_repeat: int = 2
    state_stack: int = 4
    min_action_hold: int = 2
    timeout_ms: int = 20_000
    max_retries: int = 12
    max_reset_retries: int = 3
    retry_sleep_ms: float = 120.0
    reload_wait_ms: float = 900.0
    viewport_width: int = 430
    viewport_height: int = 932


class RunState:
    MENU = "MENU"
    READY = "READY"
    RUNNING = "RUNNING"
    GAME_OVER = "GAME_OVER"
    RECOVERING = "RECOVERING"


class TapCapStateEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        url: str = "https://play.tapcapgame.com/",
        headless: bool = True,
        step_dt: float = 0.05,
        action_repeat: int = 2,
        state_stack: int = 4,
        min_action_hold: int = 2,
        timeout_ms: int = 20_000,
        max_retries: int = 12,
        max_reset_retries: int = 3,
        retry_sleep_ms: float = 120.0,
        reload_wait_ms: float = 900.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if step_dt <= 0:
            raise ValueError("step_dt must be > 0")
        if action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if state_stack < 1:
            raise ValueError("state_stack must be >= 1")
        if min_action_hold < 0:
            raise ValueError("min_action_hold must be >= 0")

        self.config = TapCapEnvConfig(
            url=url,
            headless=False if render_mode == "human" else headless,
            step_dt=step_dt,
            action_repeat=action_repeat,
            state_stack=state_stack,
            min_action_hold=min_action_hold,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            max_reset_retries=max_reset_retries,
            retry_sleep_ms=retry_sleep_ms,
            reload_wait_ms=reload_wait_ms,
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
        self._reason_window: deque[str] = deque(maxlen=64)
        self._recent_actions: deque[int] = deque(maxlen=12)

        self._prev_det: TapCapDetection | None = None
        self._episode_steps = 0
        self._episode_passes = 0
        self._prev_action = 0
        self._steps_since_action_change = 99

        self._prev_obstacle_x: float | None = None
        self._last_obstacle_x_for_running: float | None = None
        self._running_motion_streak = 0
        self._confirmed_running = False

        self._overlay_persist = 0
        self._menu_persist = 0
        self._vision_lost_persist = 0
        self._inactive_persist = 0
        self._nonprogress_steps = 0
        self._missing_frames = 0

        self._run_state = RunState.RECOVERING
        self._refresh_count = 0
        self._restart_clicks = 0
        self._runtime_recovery_count = 0

    def _default_detection(self) -> TapCapDetection:
        return TapCapDetection(
            bird_y=None,
            gap_y=None,
            obstacle_x=None,
            hud_visible=False,
            start_visible=True,
            replay_visible=False,
            width=430,
            height=932,
            map_sig=np.zeros((8,), dtype=np.float32),
        )

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tapcap_env")

    def _run_on_worker(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._ensure_executor()
        assert self._executor is not None
        future = self._executor.submit(fn, *args, **kwargs)
        return future.result()

    def _close_resources_in_worker(self) -> None:
        if self._page is not None:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None

        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

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

    def _recreate_browser_in_worker(self) -> None:
        self._close_resources_in_worker()
        self._ensure_browser_in_worker()

    def _attempt_runtime_recovery(self) -> None:
        self._runtime_recovery_count += 1
        try:
            self._run_on_worker(self._recreate_browser_in_worker)
        except Exception:
            pass

    def _refresh_run_in_worker(self) -> tuple[TapCapDetection, dict[str, int]]:
        self._ensure_browser_in_worker()
        assert self._page is not None
        det, meta = refresh_and_enter_run(
            page=self._page,
            url=self.config.url,
            reload_wait_ms=self.config.reload_wait_ms,
            max_attempts=12,
            hard_refreshes=3,
            warmup_taps=0,
            warmup_interval_ms=170.0,
            return_meta=True,
        )
        return det, meta

    def _wait_step_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        self._page.wait_for_timeout(self.config.step_dt * 1000.0)

    def _tap_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        tap(self._page)

    def _extract_det_in_worker(self) -> TapCapDetection | None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        raw = extract_tapcap_state(
            self._page,
            retries=self.config.max_retries,
            retry_sleep_ms=self.config.retry_sleep_ms,
        )
        return parse_tapcap_detection(raw)

    def _safe_extract_det(self) -> TapCapDetection | None:
        try:
            return self._run_on_worker(self._extract_det_in_worker)
        except Exception:
            return None

    def _classify_state(self, det: TapCapDetection | None) -> str:
        if det is None:
            return RunState.RECOVERING
        if det.replay_visible:
            return RunState.GAME_OVER
        if det.start_visible:
            return RunState.READY if det.hud_visible else RunState.MENU
        if det.hud_visible and det.bird_y is not None:
            return RunState.RUNNING
        if det.hud_visible:
            return RunState.READY
        return RunState.RECOVERING

    def _update_running_confirmation(self, det: TapCapDetection | None, state: str) -> None:
        if det is None:
            return

        if state == RunState.RUNNING:
            moving = (
                self._last_obstacle_x_for_running is not None
                and det.obstacle_x is not None
                and det.obstacle_x < (self._last_obstacle_x_for_running - 0.4)
            )
            if moving:
                self._running_motion_streak += 1
            else:
                self._running_motion_streak = 0

            if self._running_motion_streak >= 2:
                self._confirmed_running = True

        if det.obstacle_x is not None:
            self._last_obstacle_x_for_running = det.obstacle_x

    def _compute_pass_increment(self, det: TapCapDetection) -> int:
        if det.obstacle_x is None:
            return 0

        line = 0.27 * det.width
        inc = 0
        if self._prev_obstacle_x is not None and self._prev_obstacle_x > line >= det.obstacle_x:
            inc = 1
        self._prev_obstacle_x = det.obstacle_x
        return inc

    def _terminal_reason(self, det: TapCapDetection | None, state: str) -> str:
        if det is None:
            self._vision_lost_persist += 1
        else:
            self._vision_lost_persist = 0

        if state == RunState.GAME_OVER:
            self._overlay_persist += 1
        else:
            self._overlay_persist = 0

        if state in {RunState.MENU, RunState.READY} and self._episode_steps > 10:
            self._menu_persist += 1
        else:
            self._menu_persist = 0

        if self._confirmed_running and state != RunState.RUNNING:
            self._inactive_persist += 1
        else:
            self._inactive_persist = 0

        if state != RunState.RUNNING and not self._confirmed_running:
            self._nonprogress_steps += 1
        else:
            self._nonprogress_steps = 0

        if self._overlay_persist >= 2:
            return "death_overlay"
        if self._inactive_persist >= 2:
            return "death_inactive_confirmed"
        if self._vision_lost_persist >= 3 or self._missing_frames >= 3:
            return "vision_lost"
        if self._menu_persist >= 2 and self._episode_steps > 20 and not self._confirmed_running:
            return "menu_abort"
        if self._nonprogress_steps >= 80:
            return "run_stuck"
        return ""

    def _build_obs_from_det(self, det: TapCapDetection) -> np.ndarray:
        vec = build_state_vector(det, previous=self._prev_det, step_dt=self.config.step_dt)
        obs = update_stacked_state(self._history, vec, self.config.state_stack)
        self._prev_det = det
        return obs

    def _reason_counts_window(self) -> dict[str, int]:
        return dict(Counter(self._reason_window))

    def _base_info(self, det: TapCapDetection | None, done_reason: str, runtime_error: int) -> dict[str, Any]:
        active_det = det if det is not None else self._default_detection()
        return {
            "episode_steps": self._episode_steps,
            "episode_passes": self._episode_passes,
            "map_key": map_key_from_signature(active_det.map_sig),
            "done_reason": done_reason,
            "run_state": self._run_state,
            "confirmed_running": float(self._confirmed_running),
            "refresh_count": int(self._refresh_count),
            "restart_clicks": int(self._restart_clicks),
            "reason_counts_window": self._reason_counts_window(),
            "runtime_error": int(runtime_error),
            "runtime_recovery_count": int(self._runtime_recovery_count),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options

        runtime_error = 0
        last_error: Exception | None = None
        det: TapCapDetection | None = None

        for _attempt in range(max(1, self.config.max_reset_retries)):
            try:
                det, meta = self._run_on_worker(self._refresh_run_in_worker)
                self._refresh_count += int(meta.get("refresh_count", 0))
                self._restart_clicks += int(meta.get("restart_clicks", 0))
                break
            except Exception as exc:
                runtime_error = 1
                last_error = exc
                self._attempt_runtime_recovery()
                continue

        if det is None:
            det = self._default_detection()

        self._history.clear()
        self._prev_det = None
        obs = self._build_obs_from_det(det)

        self._episode_steps = 0
        self._episode_passes = 0
        self._missing_frames = 0
        self._prev_action = 0
        self._steps_since_action_change = 99
        self._prev_obstacle_x = det.obstacle_x
        self._last_obstacle_x_for_running = det.obstacle_x
        self._running_motion_streak = 0
        self._confirmed_running = False
        self._overlay_persist = 0
        self._menu_persist = 0
        self._vision_lost_persist = 0
        self._inactive_persist = 0
        self._nonprogress_steps = 0
        self._run_state = self._classify_state(det)
        self._recent_actions.clear()

        info = self._base_info(det=det, done_reason="", runtime_error=runtime_error)
        if runtime_error and last_error is not None:
            info["reset_error"] = str(last_error)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward = 0.0
        done = False
        done_reason = ""
        runtime_error = 0
        latest_det: TapCapDetection | None = None

        requested_action = int(action)
        action_int = requested_action
        if requested_action != self._prev_action:
            if self._steps_since_action_change < self.config.min_action_hold:
                action_int = self._prev_action
                self._steps_since_action_change += 1
            else:
                reward -= 0.006
                self._steps_since_action_change = 0
                self._prev_action = requested_action
                action_int = requested_action
        else:
            self._steps_since_action_change += 1

        self._recent_actions.append(action_int)
        flip_ratio = 0.0
        if len(self._recent_actions) >= 6:
            actions = list(self._recent_actions)
            flips = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])
            flip_ratio = float(flips / max(1, len(actions) - 1))
            if flip_ratio > 0.6:
                reward -= 0.08 * (flip_ratio - 0.6)

        try:
            for _ in range(self.config.action_repeat):
                if action_int == 1:
                    try:
                        self._run_on_worker(self._tap_in_worker)
                    except Exception:
                        runtime_error = 1
                        self._attempt_runtime_recovery()

                try:
                    self._run_on_worker(self._wait_step_in_worker)
                except Exception:
                    runtime_error = 1
                    self._attempt_runtime_recovery()

                det = self._safe_extract_det()
                latest_det = det
                self._episode_steps += 1

                if det is None:
                    self._missing_frames += 1
                    if self._missing_frames >= 3:
                        done = True
                        done_reason = "vision_lost"
                        break
                    continue

                self._missing_frames = 0
                self._run_state = self._classify_state(det)
                self._update_running_confirmation(det, self._run_state)

                active = self._run_state == RunState.RUNNING and self._confirmed_running
                if active:
                    reward += 0.05

                inc = self._compute_pass_increment(det)
                if inc > 0:
                    reward += 1.0 * float(inc)
                    reward += 0.5 * float(inc)
                self._episode_passes += inc

                if det.bird_y is not None and active:
                    corridor_y = det.gap_y if det.gap_y is not None else (0.52 * float(det.height))
                    dev = abs((det.bird_y - corridor_y) / max(1.0, float(det.height)))
                    if dev > 0.20:
                        reward -= 0.06 * (dev - 0.20)
                    if dev > 0.35:
                        reward -= 0.10 * (dev - 0.35)

                done_reason = self._terminal_reason(det, self._run_state)
                if done_reason:
                    done = True
                    break

        except Exception:
            runtime_error = 1
            self._attempt_runtime_recovery()
            done = True
            done_reason = "vision_lost"

        if latest_det is None:
            latest_det = self._prev_det if self._prev_det is not None else self._default_detection()

        obs = self._build_obs_from_det(latest_det)

        if done:
            reward -= 1.0
            if done_reason:
                self._reason_window.append(done_reason)

        reward = float(np.clip(reward, -2.0, 2.0))
        info = self._base_info(det=latest_det, done_reason=done_reason, runtime_error=runtime_error)
        info["flip_ratio"] = float(flip_ratio)
        info["inactive_frames"] = int(self._inactive_persist)
        return obs, reward, done, False, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        if self._executor is not None:
            try:
                self._run_on_worker(self._close_resources_in_worker)
            except Exception:
                pass
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        super().close()
