from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

from tapcap_state import extract_tapcap_state


@dataclass
class Detection:
    bird_y: float | None
    gap_y: float | None
    obstacle_x: float | None
    hud_visible: bool
    start_button_visible: bool
    replay_visible: bool
    width: int
    height: int
    map_sig: np.ndarray


@dataclass
class ControllerParams:
    target_offset: float = 0.0
    deadzone: float = 6.0
    vel_gain: float = 1.2
    tap_cooldown_ms: float = 95.0

    def clipped(self) -> "ControllerParams":
        return ControllerParams(
            target_offset=float(np.clip(self.target_offset, -24.0, 30.0)),
            deadzone=float(np.clip(self.deadzone, 3.0, 22.0)),
            vel_gain=float(np.clip(self.vel_gain, 0.15, 3.5)),
            tap_cooldown_ms=float(np.clip(self.tap_cooldown_ms, 62.0, 280.0)),
        )


@dataclass
class MapProfile:
    params: ControllerParams
    metric: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive vision bot for TopCap Flappy-like game without changing game code. "
            "Reads state from rendered canvas and adapts per-map control parameters."
        )
    )
    parser.add_argument("--url", type=str, default="https://play.tapcapgame.com/")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run browser headless (default: false).",
    )
    parser.add_argument("--tick-ms", type=float, default=58.0, help="Control loop period in milliseconds.")
    parser.add_argument("--deadzone", type=float, default=6.0, help="Initial deadzone.")
    parser.add_argument("--target-offset", type=float, default=0.0, help="Initial vertical target offset.")
    parser.add_argument("--tap-cooldown-ms", type=float, default=95.0, help="Initial tap cooldown in ms.")
    parser.add_argument("--vel-gain", type=float, default=1.2, help="Initial velocity gain.")
    parser.add_argument("--map-change-threshold", type=float, default=28.0, help="Map change L2 threshold.")
    parser.add_argument("--max-taps-per-second", type=int, default=8, help="Hard tap rate cap.")
    parser.add_argument(
        "--recover-cooldown-ms",
        type=float,
        default=900.0,
        help="Minimum delay between hard recoveries.",
    )
    parser.add_argument(
        "--reload-wait-ms",
        type=float,
        default=900.0,
        help="Wait after page refresh before start sequence.",
    )
    parser.add_argument("--warmup-taps", type=int, default=0, help="Extra taps right after startup.")
    parser.add_argument("--warmup-interval-ms", type=float, default=170.0, help="Warmup tap interval.")
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable online parameter adaptation between episodes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--best-params-path",
        type=str,
        default=r".\models_web\tapcap_bot_best.json",
        help="Path to save/load best parameters and map profiles.",
    )
    parser.add_argument(
        "--resume-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from saved best parameters if available.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Optional runtime limit in seconds (0 means run until Ctrl+C).",
    )
    return parser.parse_args()


def canvas_bbox(page: Page) -> dict[str, float] | None:
    canvas = page.query_selector("canvas")
    if canvas is None:
        return None
    return canvas.bounding_box()


def click_canvas_norm(page: Page, nx: float, ny: float) -> None:
    box = canvas_bbox(page)
    if box is not None:
        x = box["x"] + nx * box["width"]
        y = box["y"] + ny * box["height"]
    else:
        vp = page.viewport_size or {"width": 430, "height": 932}
        x = vp["width"] * nx
        y = vp["height"] * ny
    page.mouse.click(x, y)


def tap(page: Page) -> None:
    click_canvas_norm(page, 0.50, 0.55)


def as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def params_from_obj(raw: dict[str, object], fallback: ControllerParams) -> ControllerParams:
    return ControllerParams(
        target_offset=float(raw.get("target_offset", fallback.target_offset)),
        deadzone=float(raw.get("deadzone", fallback.deadzone)),
        vel_gain=float(raw.get("vel_gain", fallback.vel_gain)),
        tap_cooldown_ms=float(raw.get("tap_cooldown_ms", fallback.tap_cooldown_ms)),
    ).clipped()


def map_key_from_sig(sig: np.ndarray) -> str:
    if sig.size < 8 or not np.any(sig):
        return "unknown"
    bins = np.array([15.0, 15.0, 15.0, 9.0, 9.0, 9.0, 12.0, 12.0], dtype=np.float32)
    q = np.round(sig / bins).astype(np.int32)
    return "|".join(str(int(v)) for v in q.tolist())


def load_saved_state(
    best_params_path: Path,
    fallback_params: ControllerParams,
) -> tuple[ControllerParams, float, dict[str, MapProfile]]:
    best_params = fallback_params
    best_metric = -1e9
    map_profiles: dict[str, MapProfile] = {}

    if not best_params_path.exists():
        return best_params, best_metric, map_profiles

    raw = json.loads(best_params_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return best_params, best_metric, map_profiles

    best_params = params_from_obj(raw, fallback_params)
    best_metric = float(raw.get("metric", -1e9))

    raw_profiles = raw.get("map_profiles", {})
    if isinstance(raw_profiles, dict):
        for key, value in raw_profiles.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            prof_params = params_from_obj(value, best_params)
            prof_metric = float(value.get("metric", -1e9))
            map_profiles[key] = MapProfile(params=prof_params, metric=prof_metric)

    return best_params, best_metric, map_profiles


def save_state(
    best_params_path: Path,
    best_params: ControllerParams,
    best_metric: float,
    map_profiles: dict[str, MapProfile],
) -> None:
    sorted_items = sorted(map_profiles.items(), key=lambda item: item[1].metric, reverse=True)
    trimmed = sorted_items[:32]
    payload_profiles: dict[str, dict[str, float]] = {}
    for key, prof in trimmed:
        payload_profiles[key] = {
            "target_offset": prof.params.target_offset,
            "deadzone": prof.params.deadzone,
            "vel_gain": prof.params.vel_gain,
            "tap_cooldown_ms": prof.params.tap_cooldown_ms,
            "metric": prof.metric,
        }

    payload: dict[str, object] = {
        "target_offset": best_params.target_offset,
        "deadzone": best_params.deadzone,
        "vel_gain": best_params.vel_gain,
        "tap_cooldown_ms": best_params.tap_cooldown_ms,
        "metric": best_metric,
        "updated_at_unix": time.time(),
        "map_profiles": payload_profiles,
    }

    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_active_run(state: Detection) -> bool:
    return state.hud_visible and (not state.start_button_visible) and (state.bird_y is not None)


def parse_detection(features: dict[str, object] | None) -> Detection | None:
    if features is None:
        return None

    width = int(features.get("width", 430))
    height = int(features.get("height", 765))
    bird_y = as_float(features.get("birdY"))
    gap_y = as_float(features.get("gapY"))
    obstacle_x = as_float(features.get("obstacleX"))
    hud_visible = bool(features.get("hudVisible", False))
    start_button_visible = bool(features.get("startVisible", False))
    replay_visible = bool(features.get("replayVisible", False))

    raw_sig = features.get("mapSig")
    if isinstance(raw_sig, list):
        map_sig = np.asarray(raw_sig, dtype=np.float32)
    else:
        map_sig = np.zeros((8,), dtype=np.float32)

    return Detection(
        bird_y=bird_y,
        gap_y=gap_y,
        obstacle_x=obstacle_x,
        hud_visible=hud_visible,
        start_button_visible=start_button_visible,
        replay_visible=replay_visible,
        width=width,
        height=height,
        map_sig=map_sig,
    )


def read_canvas_features(page: Page) -> dict[str, object] | None:
    return extract_tapcap_state(page)


def try_click_replay(page: Page) -> bool:
    try:
        replay = page.query_selector("#replayBtn")
        if replay is not None and replay.is_visible():
            replay.click(timeout=500)
            return True
    except PlaywrightError:
        return False
    return False


def refresh_and_start(
    page: Page,
    url: str,
    reload_wait_ms: float,
    warmup_taps: int,
    warmup_interval_ms: float,
) -> Detection | None:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_timeout(reload_wait_ms)

    last_state: Detection | None = None
    for _ in range(8):
        features = read_canvas_features(page)
        state = parse_detection(features)
        if state is None:
            page.wait_for_timeout(130)
            continue

        last_state = state
        if is_active_run(state):
            break

        if state.replay_visible:
            try_click_replay(page)
            page.wait_for_timeout(180)

        sx = as_float(features.get("startCenterX")) if features is not None else None
        sy = as_float(features.get("startCenterY")) if features is not None else None
        if state.start_button_visible:
            if sx is not None and sy is not None:
                click_canvas_norm(page, sx, sy)
                page.wait_for_timeout(110)
                click_canvas_norm(page, sx, sy)
            else:
                click_canvas_norm(page, 0.50, 0.86)
                page.wait_for_timeout(110)
                click_canvas_norm(page, 0.50, 0.86)
        else:
            tap(page)
            page.wait_for_timeout(110)

        tap(page)
        page.wait_for_timeout(130)

        retry_features = read_canvas_features(page)
        retry_state = parse_detection(retry_features)
        if retry_state is not None:
            last_state = retry_state
            if is_active_run(retry_state):
                break

    for _ in range(max(0, warmup_taps)):
        tap(page)
        page.wait_for_timeout(max(90.0, warmup_interval_ms))

    return last_state


def propose_next_params(
    rng: np.random.Generator,
    current: ControllerParams,
    best: ControllerParams,
    improved: bool,
    episode_duration: float,
    episode_index: int,
    had_map_change: bool,
) -> ControllerParams:
    base = best if improved else current
    scale = max(0.35, 1.7 / (1.0 + 0.08 * episode_index))

    candidate = ControllerParams(
        target_offset=base.target_offset + float(rng.normal(0.0, 1.8 * scale)),
        deadzone=base.deadzone + float(rng.normal(0.0, 0.95 * scale)),
        vel_gain=base.vel_gain + float(rng.normal(0.0, 0.22 * scale)),
        tap_cooldown_ms=base.tap_cooldown_ms + float(rng.normal(0.0, 6.5 * scale)),
    )

    if episode_duration < 2.7:
        candidate.target_offset -= 1.8
        candidate.deadzone -= 0.7
        candidate.vel_gain += 0.12
        candidate.tap_cooldown_ms -= 8.0
    elif episode_duration > 7.0:
        candidate.deadzone += 0.30
        candidate.tap_cooldown_ms += 2.5

    if had_map_change:
        candidate.deadzone += 0.5
        candidate.tap_cooldown_ms += 4.0

    return candidate.clipped()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    tick_ms = max(28.0, float(args.tick_ms))
    recover_cooldown_ms = max(500.0, float(args.recover_cooldown_ms))
    reload_wait_ms = max(250.0, float(args.reload_wait_ms))
    warmup_interval_ms = max(90.0, float(args.warmup_interval_ms))
    max_taps_per_second = max(2, int(args.max_taps_per_second))
    map_change_threshold = max(6.0, float(args.map_change_threshold))

    initial_params = ControllerParams(
        target_offset=float(args.target_offset),
        deadzone=float(args.deadzone),
        vel_gain=float(args.vel_gain),
        tap_cooldown_ms=float(args.tap_cooldown_ms),
    ).clipped()

    best_params_path = Path(args.best_params_path)
    best_metric = -1e9
    map_profiles: dict[str, MapProfile] = {}
    if args.resume_best and best_params_path.exists():
        try:
            saved_params, saved_metric, saved_profiles = load_saved_state(best_params_path, initial_params)
            sane = (
                saved_metric >= 4.0
                and abs(saved_params.target_offset) <= 14.0
                and 3.5 <= saved_params.deadzone <= 14.0
                and 70.0 <= saved_params.tap_cooldown_ms <= 180.0
                and 0.2 <= saved_params.vel_gain <= 2.8
            )
            if sane:
                initial_params = saved_params
                best_metric = saved_metric
                map_profiles = saved_profiles
                print(
                    f"Loaded saved params from {best_params_path} "
                    f"(metric={best_metric:.2f}, map_profiles={len(map_profiles)})."
                )
            else:
                print(
                    f"Ignoring saved params from {best_params_path} "
                    f"(metric={saved_metric:.2f}) because they look unstable."
                )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to load {best_params_path}: {exc}")

    current_params = initial_params
    best_params = initial_params

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 430, "height": 932},
            device_scale_factor=1,
        )
        page = context.new_page()

        refresh_and_start(
            page=page,
            url=args.url,
            reload_wait_ms=reload_wait_ms,
            warmup_taps=int(args.warmup_taps),
            warmup_interval_ms=warmup_interval_ms,
        )

        print("Adaptive bot running. Press Ctrl+C to stop.")

        start_time = time.time()
        episode_index = 1
        episode_start = time.time()
        episode_passes = 0
        episode_had_map_change = False

        missing_bird_frames = 0
        start_button_frames = 0
        replay_frames = 0
        stagnant_frames = 0
        last_tap_time = 0.0
        last_ui_click_time = 0.0
        last_recover_time = 0.0
        last_log_time = 0.0
        tap_times: list[float] = []

        prev_obstacle_x: float | None = None
        prev_motion_bird_y: float | None = None
        prev_motion_obstacle_x: float | None = None
        ema_bird_y: float | None = None
        ema_gap_y: float | None = None
        prev_ema_bird_y: float | None = None
        tap_armed = False

        map_sig_ema: np.ndarray | None = None
        map_baseline: np.ndarray | None = None
        current_map_key = "unknown"
        map_diff = 0.0
        map_change_count = 0
        map_shift_frames = 0
        map_grace_until = 0.0
        last_map_change_time = 0.0

        try:
            while True:
                loop_start = time.time()

                features = read_canvas_features(page)
                state = parse_detection(features)
                if state is None:
                    page.wait_for_timeout(140)
                    continue

                now = time.time()
                episode_age = now - episode_start

                if state.bird_y is None:
                    missing_bird_frames += 1
                else:
                    missing_bird_frames = 0

                if state.start_button_visible:
                    start_button_frames += 1
                else:
                    start_button_frames = 0

                if state.replay_visible:
                    replay_frames += 1
                else:
                    replay_frames = 0

                motion_score = 0.0
                if state.bird_y is not None and prev_motion_bird_y is not None:
                    motion_score += abs(state.bird_y - prev_motion_bird_y)
                if state.obstacle_x is not None and prev_motion_obstacle_x is not None:
                    motion_score += 1.45 * abs(state.obstacle_x - prev_motion_obstacle_x)
                if state.start_button_visible:
                    motion_score *= 0.2

                prev_motion_bird_y = state.bird_y
                prev_motion_obstacle_x = state.obstacle_x

                if motion_score < 1.7:
                    stagnant_frames += 1
                else:
                    stagnant_frames = 0

                if state.bird_y is not None:
                    if ema_bird_y is None:
                        ema_bird_y = state.bird_y
                    else:
                        ema_bird_y = 0.62 * ema_bird_y + 0.38 * state.bird_y

                if state.gap_y is not None:
                    if ema_gap_y is None:
                        ema_gap_y = state.gap_y
                    else:
                        ema_gap_y = 0.68 * ema_gap_y + 0.32 * state.gap_y

                if state.obstacle_x is not None:
                    pass_line = 0.27 * state.width
                    if prev_obstacle_x is not None and prev_obstacle_x > pass_line >= state.obstacle_x:
                        episode_passes += 1
                    prev_obstacle_x = state.obstacle_x
                else:
                    prev_obstacle_x = None

                if state.map_sig.size == 8 and np.any(state.map_sig):
                    if map_sig_ema is None:
                        map_sig_ema = state.map_sig.copy()
                    else:
                        map_sig_ema = 0.88 * map_sig_ema + 0.12 * state.map_sig

                    active_now = is_active_run(state)
                    if map_baseline is None and active_now and episode_age > 0.8:
                        map_baseline = map_sig_ema.copy()
                        current_map_key = map_key_from_sig(map_baseline)

                    if map_baseline is not None:
                        map_diff = float(np.linalg.norm(map_sig_ema - map_baseline))
                        if active_now and map_diff < (0.55 * map_change_threshold):
                            map_baseline = 0.995 * map_baseline + 0.005 * map_sig_ema
                            map_diff = float(np.linalg.norm(map_sig_ema - map_baseline))
                        if active_now and map_diff > map_change_threshold:
                            map_shift_frames += 1
                        else:
                            map_shift_frames = max(0, map_shift_frames - 1)

                        map_changed = (
                            active_now
                            and map_shift_frames >= 8
                            and (now - last_map_change_time) > 2.0
                            and episode_age > 3.0
                        )
                        if map_changed:
                            map_baseline = map_sig_ema.copy()
                            map_shift_frames = 0
                            last_map_change_time = now
                            map_grace_until = now + 1.5
                            map_change_count += 1
                            episode_had_map_change = True
                            ema_gap_y = None
                            prev_ema_bird_y = None
                            stagnant_frames = 0
                            tap_armed = False

                            new_key = map_key_from_sig(map_baseline)
                            if new_key != current_map_key:
                                current_map_key = new_key
                                prof = map_profiles.get(current_map_key)
                                if prof is not None and prof.metric >= 3.0:
                                    current_params = prof.params
                                    print(
                                        f"[Episode {episode_index}] map change -> profile {current_map_key} "
                                        f"(metric={prof.metric:.2f}) params={current_params}"
                                    )
                                else:
                                    current_params = ControllerParams(
                                        target_offset=current_params.target_offset - 0.8,
                                        deadzone=current_params.deadzone + 0.8,
                                        vel_gain=current_params.vel_gain,
                                        tap_cooldown_ms=current_params.tap_cooldown_ms - 4.0,
                                    ).clipped()
                                    print(
                                        f"[Episode {episode_index}] map change -> unknown profile {current_map_key}, "
                                        f"using safe retune params={current_params}"
                                    )

                if args.max_seconds > 0 and (now - start_time) >= args.max_seconds:
                    print(f"Reached --max-seconds={args.max_seconds}, exiting.")
                    break

                # UI assist: if menu detected, click start softly instead of spam.
                if (
                    state.start_button_visible
                    and (now - last_ui_click_time) > 0.9
                    and (now - last_recover_time) > 0.5
                    and features is not None
                ):
                    sx = as_float(features.get("startCenterX"))
                    sy = as_float(features.get("startCenterY"))
                    if sx is not None and sy is not None:
                        click_canvas_norm(page, sx, sy)
                    else:
                        click_canvas_norm(page, 0.50, 0.86)
                    last_ui_click_time = now

                fail_replay = replay_frames >= 1 and episode_age > 1.0
                fail_menu = (
                    start_button_frames >= 2
                    and episode_age > 1.5
                    and (motion_score < 4.0 or (not state.hud_visible))
                )
                fail_missing_bird = missing_bird_frames > 11 and episode_age > 1.5
                fail_stagnant = stagnant_frames > 24 and episode_age > 2.0

                if now < map_grace_until:
                    fail_menu = False
                    fail_stagnant = False

                need_recover = fail_replay or fail_menu or fail_missing_bird or fail_stagnant
                if need_recover and (now - last_recover_time) * 1000.0 >= recover_cooldown_ms:
                    duration = now - episode_start
                    metric = duration + 0.50 * episode_passes
                    improved = metric > best_metric

                    if improved:
                        best_metric = metric
                        best_params = current_params

                    if current_map_key != "unknown":
                        prev_prof = map_profiles.get(current_map_key)
                        if prev_prof is None or metric > prev_prof.metric:
                            map_profiles[current_map_key] = MapProfile(params=current_params, metric=metric)

                    try:
                        save_state(best_params_path, best_params, best_metric, map_profiles)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Warning: failed to save params to {best_params_path}: {exc}")

                    print(
                        f"[Episode {episode_index}] duration={duration:.2f}s passes={episode_passes} "
                        f"metric={metric:.2f} best={best_metric:.2f} map={current_map_key} "
                        f"mapDiff={map_diff:.2f} params={current_params}"
                    )

                    if args.adaptive:
                        current_params = propose_next_params(
                            rng=rng,
                            current=current_params,
                            best=best_params,
                            improved=improved,
                            episode_duration=duration,
                            episode_index=episode_index,
                            had_map_change=episode_had_map_change,
                        )
                        print(f"[Episode {episode_index}] next params={current_params}")

                    refresh_and_start(
                        page=page,
                        url=args.url,
                        reload_wait_ms=reload_wait_ms,
                        warmup_taps=int(args.warmup_taps),
                        warmup_interval_ms=warmup_interval_ms,
                    )
                    last_recover_time = time.time()

                    episode_index += 1
                    episode_start = time.time()
                    episode_passes = 0
                    episode_had_map_change = False
                    missing_bird_frames = 0
                    start_button_frames = 0
                    replay_frames = 0
                    stagnant_frames = 0
                    last_tap_time = 0.0
                    tap_times = []
                    prev_obstacle_x = None
                    prev_motion_bird_y = None
                    prev_motion_obstacle_x = None
                    ema_bird_y = None
                    ema_gap_y = None
                    prev_ema_bird_y = None
                    tap_armed = False
                    map_sig_ema = None
                    map_baseline = None
                    current_map_key = "unknown"
                    map_diff = 0.0
                    map_shift_frames = 0
                    map_grace_until = 0.0
                    continue

                active = is_active_run(state)
                if active and ema_bird_y is not None:
                    base_target = 0.43 * state.height
                    if ema_gap_y is not None:
                        if state.obstacle_x is not None:
                            near = float(
                                np.clip(
                                    (0.65 * state.width - state.obstacle_x) / (0.45 * state.width),
                                    0.0,
                                    1.0,
                                )
                            )
                            gap_blend = 0.30 + (0.55 * near)
                        else:
                            gap_blend = 0.42
                        target_y = ((1.0 - gap_blend) * base_target) + (
                            gap_blend * (ema_gap_y + current_params.target_offset)
                        )
                    else:
                        target_y = base_target + current_params.target_offset

                    vy = 0.0 if prev_ema_bird_y is None else (ema_bird_y - prev_ema_bird_y)
                    error = ema_bird_y - target_y
                    drive = error + current_params.vel_gain * max(0.0, vy)

                    hi = current_params.deadzone
                    lo = max(2.0, 0.48 * current_params.deadzone)
                    if drive > hi:
                        tap_armed = True
                    elif drive < lo:
                        tap_armed = False

                    tap_times = [t for t in tap_times if (now - t) <= 1.0]
                    rate_ok = len(tap_times) < max_taps_per_second
                    emergency_low = ema_bird_y > (0.60 * state.height)
                    effective_cooldown = 68.0 if emergency_low else current_params.tap_cooldown_ms
                    cooldown_ok = (now - last_tap_time) * 1000.0 >= effective_cooldown
                    rising_too_fast = (vy < -2.2) and (ema_bird_y < target_y - 5.0)
                    startup_pulse = (
                        episode_age < 1.05
                        and ema_bird_y > (0.56 * state.height)
                        and ((now - last_tap_time) * 1000.0 >= 220.0)
                    )

                    do_tap = (
                        (tap_armed and rate_ok and cooldown_ok and not rising_too_fast)
                        or (emergency_low and rate_ok and cooldown_ok)
                        or (startup_pulse and rate_ok)
                    )
                    if do_tap:
                        tap(page)
                        last_tap_time = now
                        tap_times.append(now)
                elif state.hud_visible and (not state.start_button_visible) and state.bird_y is not None:
                    # Tutorial / early-run nudge if no active control lock yet.
                    if (now - last_tap_time) > 0.40 and episode_age < 1.2:
                        tap(page)
                        last_tap_time = now

                prev_ema_bird_y = ema_bird_y

                if now - last_log_time >= 1.0:
                    print(
                        f"ep={episode_index} bird={state.bird_y} gap={state.gap_y} obs={state.obstacle_x} "
                        f"hud={state.hud_visible} start={state.start_button_visible} replay={state.replay_visible} "
                        f"miss={missing_bird_frames} stg={stagnant_frames} motion={motion_score:.2f} "
                        f"map={current_map_key} mapDiff={map_diff:.2f} mapShift={map_shift_frames} "
                        f"mapChanges={map_change_count} taps/s={len(tap_times)}"
                    )
                    last_log_time = now

                elapsed_ms = (time.time() - loop_start) * 1000.0
                sleep_ms = tick_ms - elapsed_ms
                if sleep_ms > 0:
                    page.wait_for_timeout(sleep_ms)
        except KeyboardInterrupt:
            print("Stopped by user.")
        finally:
            context.close()
            browser.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
