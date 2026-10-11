from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


@dataclass
class WebFlappyConfig:
    url: str = "http://127.0.0.1:5173"
    headless: bool = True
    step_dt: float = 0.05
    timeout_ms: int = 15_000
    max_retries: int = 25
    retry_sleep: float = 0.1
    viewport_width: int = 420
    viewport_height: int = 720


class WebFlappyEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        url: str = "http://127.0.0.1:5173",
        headless: bool = True,
        step_dt: float = 0.05,
        timeout_ms: int = 15_000,
        max_retries: int = 25,
        retry_sleep: float = 0.1,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if step_dt <= 0:
            raise ValueError("step_dt must be > 0")

        self.config = WebFlappyConfig(
            url=url,
            headless=headless if render_mode != "human" else False,
            step_dt=step_dt,
            timeout_ms=timeout_ms,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5,),
            dtype=np.float32,
        )

        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._executor: ThreadPoolExecutor | None = None

        self._last_score: float = 0.0
        self._episode_steps: int = 0
        self._alive: bool = True

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright_sync")

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
                }
            )
        if self._page is None or self._page.is_closed():
            self._page = self._context.new_page()

    def _goto_game(self) -> None:
        self._run_on_worker(self._goto_game_in_worker)

    def _goto_game_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        last_error: Exception | None = None
        for _ in range(self.config.max_retries):
            try:
                self._page.goto(
                    self.config.url,
                    wait_until="domcontentloaded",
                    timeout=self.config.timeout_ms,
                )
                self._page.wait_for_function(
                    "() => window.__rl && "
                    "typeof window.__rl.reset === 'function' && "
                    "typeof window.__rl.flap === 'function' && "
                    "typeof window.__rl.getState === 'function' && "
                    "typeof window.__rl.isDone === 'function'",
                    timeout=self.config.timeout_ms,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self.config.retry_sleep)

        raise RuntimeError(
            f"Failed to load RL API at {self.config.url} after "
            f"{self.config.max_retries} retries. Last error: {last_error}"
        )

    def _rl_eval(
        self, expression: str, arg: Any | None = None, *, retries: int | None = None
    ) -> Any:
        effective_retries = retries if retries is not None else self.config.max_retries
        return self._run_on_worker(self._rl_eval_in_worker, expression, arg, effective_retries)

    def _rl_eval_in_worker(self, expression: str, arg: Any | None, retries: int) -> Any:
        self._ensure_browser_in_worker()
        assert self._page is not None
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                if arg is None:
                    return self._page.evaluate(expression)
                return self._page.evaluate(expression, arg)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self.config.retry_sleep)
        raise RuntimeError(f"Playwright evaluate failed: {expression}. Last error: {last_error}")

    def _wait_step(self) -> None:
        self._run_on_worker(self._wait_step_in_worker)

    def _wait_step_in_worker(self) -> None:
        self._ensure_browser_in_worker()
        assert self._page is not None
        self._page.wait_for_timeout(self.config.step_dt * 1000.0)

    @staticmethod
    def _state_to_obs(state: dict[str, Any]) -> np.ndarray:
        return np.array(
            [
                float(state.get("y", 0.0)),
                float(state.get("vy", 0.0)),
                float(state.get("nextPipeDx", 0.0)),
                float(state.get("gapDy", 0.0)),
                float(state.get("score", 0.0)),
            ],
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        _ = options

        self._goto_game()

        seed_arg = int(seed) if seed is not None else None
        self._rl_eval("(seed) => window.__rl.reset(seed)", seed_arg)
        self._rl_eval("() => window.__rl.getState()")  # warm-up call
        state = self._rl_eval("() => window.__rl.getState()")

        self._last_score = float(state.get("score", 0.0))
        self._episode_steps = 0
        self._alive = bool(state.get("alive", True))
        obs = self._state_to_obs(state)
        info = {"score": self._last_score, "alive": self._alive}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if int(action) == 1:
            self._rl_eval("() => window.__rl.flap()")

        self._wait_step()

        state = self._rl_eval("() => window.__rl.getState()")
        done_flag = bool(self._rl_eval("() => window.__rl.isDone()"))

        score = float(state.get("score", 0.0))
        alive = bool(state.get("alive", True))
        done = done_flag or (not alive)

        reward = 0.0
        if alive:
            reward += 0.1

        if score > self._last_score:
            reward += score - self._last_score

        if done:
            reward -= 1.0

        self._last_score = score
        self._alive = alive
        self._episode_steps += 1

        obs = self._state_to_obs(state)
        info = {
            "score": score,
            "alive": alive,
            "episode_steps": self._episode_steps,
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
