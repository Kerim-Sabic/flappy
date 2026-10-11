from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page

MAP_SIG_DIM = 8
BASE_STATE_DIM = 6 + MAP_SIG_DIM


@dataclass
class TapCapDetection:
    bird_y: float | None
    gap_y: float | None
    obstacle_x: float | None
    hud_visible: bool
    start_visible: bool
    replay_visible: bool
    width: int
    height: int
    map_sig: np.ndarray


def map_key_from_signature(sig: np.ndarray) -> str:
    if sig.size < MAP_SIG_DIM or not np.any(sig):
        return "unknown"
    bins = np.array([15.0, 15.0, 15.0, 9.0, 9.0, 9.0, 12.0, 12.0], dtype=np.float32)
    q = np.round(sig / bins).astype(np.int32)
    return "|".join(str(int(v)) for v in q.tolist())


def is_active_run(det: TapCapDetection) -> bool:
    return det.hud_visible and (not det.start_visible) and (det.bird_y is not None)


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


def try_click_replay_button(page: Page) -> bool:
    try:
        replay = page.query_selector("#replayBtn")
        if replay is not None and replay.is_visible():
            replay.click(timeout=500)
            return True
    except PlaywrightError:
        return False
    return False


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_tapcap_detection(features: dict[str, object] | None) -> TapCapDetection | None:
    if features is None:
        return None

    width = int(features.get("width", 430))
    height = int(features.get("height", 765))
    bird_y = _as_float(features.get("birdY"))
    gap_y = _as_float(features.get("gapY"))
    obstacle_x = _as_float(features.get("obstacleX"))
    hud_visible = bool(features.get("hudVisible", False))
    start_visible = bool(features.get("startVisible", False))
    replay_visible = bool(features.get("replayVisible", False))

    raw_sig = features.get("mapSig")
    if isinstance(raw_sig, list):
        map_sig = np.asarray(raw_sig, dtype=np.float32)
        if map_sig.shape != (MAP_SIG_DIM,):
            map_sig = np.resize(map_sig, (MAP_SIG_DIM,)).astype(np.float32)
    else:
        map_sig = np.zeros((MAP_SIG_DIM,), dtype=np.float32)

    return TapCapDetection(
        bird_y=bird_y,
        gap_y=gap_y,
        obstacle_x=obstacle_x,
        hud_visible=hud_visible,
        start_visible=start_visible,
        replay_visible=replay_visible,
        width=width,
        height=height,
        map_sig=map_sig,
    )


TAPCAP_EXTRACT_SCRIPT = """
() => new Promise((resolve) => {
  const canvas = document.querySelector('canvas');
  if (!canvas) {
    resolve(null);
    return;
  }

  const srcW = canvas.width | 0;
  const srcH = canvas.height | 0;
  if (srcW <= 0 || srcH <= 0) {
    resolve(null);
    return;
  }

  const done = () => {
    const maxW = 430;
    const scale = Math.min(1, maxW / srcW);
    const w = Math.max(120, Math.round(srcW * scale));
    const h = Math.max(200, Math.round(srcH * scale));

    let tmp = window.__botTmpCanvas;
    if (!tmp) {
      tmp = document.createElement('canvas');
      window.__botTmpCanvas = tmp;
    }
    if (tmp.width !== w || tmp.height !== h) {
      tmp.width = w;
      tmp.height = h;
    }

    let tctx = window.__botTmpCtx;
    if (!tctx || tctx.canvas !== tmp) {
      tctx = tmp.getContext('2d', { willReadFrequently: true });
      window.__botTmpCtx = tctx;
    }
    if (!tctx) {
      resolve(null);
      return;
    }

    tctx.imageSmoothingEnabled = false;
    tctx.clearRect(0, 0, w, h);
    tctx.drawImage(canvas, 0, 0, w, h);

    let img;
    try {
      img = tctx.getImageData(0, 0, w, h).data;
    } catch (_) {
      resolve(null);
      return;
    }

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const idx = (x, y) => ((y * w + x) << 2);
    const p = (x, y) => {
      const xx = clamp(x | 0, 0, w - 1);
      const yy = clamp(y | 0, 0, h - 1);
      const i = idx(xx, yy);
      return [img[i], img[i + 1], img[i + 2]];
    };
    const grayAt = (x, y) => {
      const c = p(x, y);
      return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    };
    const quantile = (arr, qv) => {
      if (!arr.length) return 0;
      arr.sort((a, b) => a - b);
      const n = arr.length;
      const id = Math.min(n - 1, Math.max(0, Math.floor(qv * (n - 1))));
      return arr[id];
    };

    let birdCount = 0;
    let birdSumY = 0;
    const bx0 = Math.floor(0.02 * w);
    const bx1 = Math.floor(0.62 * w);
    const by0 = Math.floor(0.08 * h);
    const by1 = Math.floor(0.92 * h);
    for (let y = by0; y < by1; y += 2) {
      for (let x = bx0; x < bx1; x += 2) {
        const [r, g, b] = p(x, y);
        if (g > 118 && r > 72 && b < 150 && (g - r) > 6 && g > (b + 5)) {
          birdCount += 1;
          birdSumY += y;
        }
      }
    }
    const birdY = birdCount >= 55 ? (birdSumY / birdCount) : null;

    let startCount = 0;
    let startSumX = 0;
    let startSumY = 0;
    const sx0 = Math.floor(0.16 * w);
    const sx1 = Math.floor(0.84 * w);
    const sy0 = Math.floor(0.68 * h);
    const sy1 = Math.floor(0.96 * h);
    for (let y = sy0; y < sy1; y += 2) {
      for (let x = sx0; x < sx1; x += 2) {
        const [r, g, b] = p(x, y);
        if (g > 96 && (g - r) > 12 && (g - b) > 8 && r < 170) {
          startCount += 1;
          startSumX += x;
          startSumY += y;
        }
      }
    }
    const startVisible = startCount > 120;
    const startCenterX = startVisible ? (startSumX / startCount) / w : null;
    const startCenterY = startVisible ? (startSumY / startCount) / h : null;

    let hudCount = 0;
    const hx0 = Math.floor(0.02 * w);
    const hx1 = Math.floor(0.36 * w);
    const hy0 = Math.floor(0.01 * h);
    const hy1 = Math.floor(0.16 * h);
    for (let y = hy0; y < hy1; y += 2) {
      for (let x = hx0; x < hx1; x += 2) {
        const [r, g, b] = p(x, y);
        if (r > 150 && g > 120 && b < 140) {
          hudCount += 1;
        }
      }
    }
    const hudVisible = hudCount > 18;

    let replayVisible = false;
    try {
      const replay = document.querySelector('#replayBtn');
      if (replay) {
        const st = getComputedStyle(replay);
        const rc = replay.getBoundingClientRect();
        replayVisible = (
          st.display !== 'none' &&
          st.visibility !== 'hidden' &&
          parseFloat(st.opacity || '1') > 0.05 &&
          rc.width > 5 &&
          rc.height > 5
        );
      }
    } catch (_) {
      replayVisible = false;
    }

    const ry0 = Math.floor(0.12 * h);
    const ry1 = Math.floor(0.92 * h);
    const rx0 = Math.floor(0.36 * w);
    const rx1 = Math.floor(0.98 * w);
    const graySamples = [];
    for (let y = ry0; y < ry1; y += 6) {
      for (let x = rx0; x < rx1; x += 6) {
        graySamples.push(grayAt(x, y));
      }
    }
    const q22 = quantile(graySamples, 0.22);
    const darkThr = Math.max(24, Math.min(145, q22 - 8));

    const yMin = Math.floor(0.10 * h);
    const yMax = Math.floor(0.94 * h);
    const searchX0 = Math.floor(0.42 * w);
    let obstacleX = null;
    for (let x = searchX0; x < w; x += 1) {
      let dark = 0;
      let edge = 0;
      let rows = 0;
      for (let y = yMin; y < yMax; y += 2) {
        rows += 1;
        const gy = grayAt(x, y);
        if (gy < darkThr) {
          dark += 1;
        }
        if (x > searchX0) {
          const gl = grayAt(x - 1, y);
          if (Math.abs(gy - gl) > 24) {
            edge += 1;
          }
        }
      }
      const darkFrac = dark / Math.max(1, rows);
      const edgeFrac = edge / Math.max(1, rows);
      const score = 0.72 * darkFrac + 0.28 * edgeFrac;
      if (darkFrac > 0.16 && score > 0.24) {
        obstacleX = x;
        break;
      }
    }

    let gapY = null;
    if (obstacleX !== null) {
      const x0 = Math.max(0, obstacleX - 2);
      const x1 = Math.min(w - 1, obstacleX + 2);
      const lo = Math.floor(0.12 * h);
      const hi = Math.floor(0.94 * h);
      let bestLen = 0;
      let bestStart = 0;
      let curLen = 0;
      let curStart = 0;
      for (let y = lo; y < hi; y += 1) {
        let blocked = false;
        for (let x = x0; x <= x1; x += 1) {
          if (grayAt(x, y) < darkThr) {
            blocked = true;
            break;
          }
        }
        if (!blocked) {
          if (curLen === 0) {
            curStart = y;
          }
          curLen += 1;
          if (curLen > bestLen) {
            bestLen = curLen;
            bestStart = curStart;
          }
        } else {
          curLen = 0;
        }
      }
      if (bestLen > Math.floor(0.09 * h)) {
        gapY = bestStart + bestLen / 2;
      }
    }

    const patches = [
      [Math.floor(0.01 * w), Math.floor(0.06 * w)],
      [Math.floor(0.07 * w), Math.floor(0.12 * w)],
      [Math.floor(0.13 * w), Math.floor(0.17 * w)],
    ];
    const my0 = Math.floor(0.16 * h);
    const my1 = Math.floor(0.62 * h);
    let n = 0;
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    let sumR2 = 0;
    let sumG2 = 0;
    let sumB2 = 0;
    const grays = [];
    for (const patch of patches) {
      const px0 = patch[0];
      const px1 = patch[1];
      for (let y = my0; y < my1; y += 3) {
        for (let x = px0; x < px1; x += 3) {
          const [r, g, b] = p(x, y);
          n += 1;
          sumR += r;
          sumG += g;
          sumB += b;
          sumR2 += r * r;
          sumG2 += g * g;
          sumB2 += b * b;
          grays.push(0.299 * r + 0.587 * g + 0.114 * b);
        }
      }
    }
    const meanR = n ? (sumR / n) : 0;
    const meanG = n ? (sumG / n) : 0;
    const meanB = n ? (sumB / n) : 0;
    const stdR = n ? Math.sqrt(Math.max(0, sumR2 / n - meanR * meanR)) : 0;
    const stdG = n ? Math.sqrt(Math.max(0, sumG2 / n - meanG * meanG)) : 0;
    const stdB = n ? Math.sqrt(Math.max(0, sumB2 / n - meanB * meanB)) : 0;
    const q25 = quantile(grays.slice(), 0.25);
    const q60 = quantile(grays.slice(), 0.60);
    const mapSig = [meanR, meanG, meanB, stdR, stdG, stdB, q25, q60];

    resolve({
      width: w,
      height: h,
      birdY: birdY,
      gapY: gapY,
      obstacleX: obstacleX,
      hudVisible: hudVisible,
      startVisible: startVisible,
      startCenterX: startCenterX,
      startCenterY: startCenterY,
      replayVisible: replayVisible,
      mapSig: mapSig,
    });
  };

  requestAnimationFrame(() => requestAnimationFrame(done));
});
"""


def extract_tapcap_state(
    page: Page,
    retries: int = 4,
    retry_sleep_ms: float = 120.0,
) -> dict[str, object] | None:
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            raw = page.evaluate(TAPCAP_EXTRACT_SCRIPT)
            if isinstance(raw, dict):
                return raw
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        page.wait_for_timeout(retry_sleep_ms)
    if last_error is not None:
        raise RuntimeError(f"extract_tapcap_state failed after {retries} retries: {last_error}") from last_error
    return None


def refresh_and_enter_run(
    page: Page,
    url: str,
    reload_wait_ms: float = 900.0,
    max_attempts: int = 10,
    hard_refreshes: int = 2,
    warmup_taps: int = 0,
    warmup_interval_ms: float = 170.0,
    return_meta: bool = False,
) -> TapCapDetection | tuple[TapCapDetection, dict[str, int]]:
    meta = {"refresh_count": 0, "restart_clicks": 0}
    state: TapCapDetection | None = None
    for refresh_idx in range(max(1, hard_refreshes)):
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(reload_wait_ms)
        meta["refresh_count"] += 1

        for _ in range(max(1, max_attempts)):
            raw = extract_tapcap_state(page)
            state = parse_tapcap_detection(raw)
            if state is None:
                page.wait_for_timeout(120)
                continue

            if is_active_run(state):
                break

            if state.replay_visible:
                if try_click_replay_button(page):
                    meta["restart_clicks"] += 1
                page.wait_for_timeout(180)

            acted = False
            sx = _as_float(raw.get("startCenterX")) if raw is not None else None
            sy = _as_float(raw.get("startCenterY")) if raw is not None else None
            if state.start_visible:
                if sx is not None and sy is not None:
                    click_canvas_norm(page, sx, sy)
                    meta["restart_clicks"] += 1
                    page.wait_for_timeout(90)
                    click_canvas_norm(page, sx, sy)
                    meta["restart_clicks"] += 1
                else:
                    click_canvas_norm(page, 0.50, 0.86)
                    meta["restart_clicks"] += 1
                    page.wait_for_timeout(90)
                    click_canvas_norm(page, 0.50, 0.86)
                    meta["restart_clicks"] += 1
                acted = True

            if not acted:
                tap(page)
                meta["restart_clicks"] += 1
                page.wait_for_timeout(95)
                tap(page)
                meta["restart_clicks"] += 1

            page.wait_for_timeout(120)
            post_raw = extract_tapcap_state(page)
            post_state = parse_tapcap_detection(post_raw)
            if post_state is not None and is_active_run(post_state):
                state = post_state
                break

        if state is not None and is_active_run(state):
            break
        if refresh_idx < max(1, hard_refreshes) - 1:
            page.wait_for_timeout(max(200.0, reload_wait_ms * 0.5))

    if state is None:
        raise RuntimeError("Failed to detect game state during refresh_and_enter_run.")
    if not is_active_run(state):
        raise RuntimeError("Failed to enter active TapCap run after bounded refresh attempts.")

    for _ in range(max(0, warmup_taps)):
        tap(page)
        meta["restart_clicks"] += 1
        page.wait_for_timeout(max(90.0, warmup_interval_ms))

    if return_meta:
        return state, meta
    return state


def build_state_vector(
    current: TapCapDetection,
    previous: TapCapDetection | None,
    step_dt: float,
) -> np.ndarray:
    h = max(1.0, float(current.height))
    w = max(1.0, float(current.width))

    if current.bird_y is not None:
        bird_y = float(current.bird_y)
    elif previous is not None and previous.bird_y is not None:
        bird_y = float(previous.bird_y)
    else:
        bird_y = 0.5 * h

    prev_bird_y = float(previous.bird_y) if previous and previous.bird_y is not None else float(bird_y)
    bird_vy = (float(bird_y) - float(prev_bird_y)) / max(1e-6, step_dt)

    obs_x = current.obstacle_x if current.obstacle_x is not None else (previous.obstacle_x if previous else w)
    prev_obs_x = previous.obstacle_x if previous and previous.obstacle_x is not None else obs_x
    obstacle_speed = (float(prev_obs_x) - float(obs_x)) / max(1e-6, step_dt)
    next_obstacle_dx = float(obs_x) - (0.28 * w)

    if current.gap_y is not None:
        gap_dy = float(current.gap_y) - float(bird_y)
    elif previous and previous.gap_y is not None:
        gap_dy = float(previous.gap_y) - float(bird_y)
    else:
        gap_dy = 0.0

    alive = 1.0 if is_active_run(current) else 0.0

    map_sig = np.asarray(current.map_sig, dtype=np.float32)
    if map_sig.shape != (MAP_SIG_DIM,):
        map_sig = np.resize(map_sig, (MAP_SIG_DIM,)).astype(np.float32)

    map_norm = np.array(
        [
            np.clip(map_sig[0] / 255.0, 0.0, 1.0),
            np.clip(map_sig[1] / 255.0, 0.0, 1.0),
            np.clip(map_sig[2] / 255.0, 0.0, 1.0),
            np.clip(map_sig[3] / 128.0, 0.0, 1.0),
            np.clip(map_sig[4] / 128.0, 0.0, 1.0),
            np.clip(map_sig[5] / 128.0, 0.0, 1.0),
            np.clip(map_sig[6] / 255.0, 0.0, 1.0),
            np.clip(map_sig[7] / 255.0, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    vec = np.array(
        [
            np.clip(float(bird_y) / h, 0.0, 1.0),
            np.clip(bird_vy / 700.0, -3.0, 3.0),
            np.clip(next_obstacle_dx / w, -1.5, 2.5),
            np.clip(gap_dy / h, -1.5, 1.5),
            np.clip(obstacle_speed / 500.0, -3.0, 3.0),
            alive,
            *map_norm.tolist(),
        ],
        dtype=np.float32,
    )
    return vec


def extract_features_from_rgb(frame: np.ndarray) -> dict[str, object]:
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("Expected RGB frame with shape (H, W, 3).")

    rgb = frame[:, :, :3].astype(np.int16)
    h, w = rgb.shape[0], rgb.shape[1]
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    bird_mask = (g > 118) & (r > 72) & (b < 150) & ((g - r) > 6) & (g > (b + 5))
    bird_mask[: int(0.08 * h), :] = False
    bird_mask[:, int(0.62 * w) :] = False
    ys = np.where(bird_mask)[0]
    bird_y = float(np.median(ys)) if ys.size >= 55 else None

    sx0, sx1 = int(0.16 * w), int(0.84 * w)
    sy0, sy1 = int(0.68 * h), int(0.96 * h)
    start_roi = rgb[sy0:sy1, sx0:sx1]
    if start_roi.size > 0:
        sr = start_roi[:, :, 0]
        sg = start_roi[:, :, 1]
        sb = start_roi[:, :, 2]
        start_mask = (sg > 96) & ((sg - sr) > 12) & ((sg - sb) > 8) & (sr < 170)
        start_visible = int(start_mask.sum()) > 120
        if start_visible:
            yy, xx = np.where(start_mask)
            start_center_x = float((sx0 + np.mean(xx)) / w)
            start_center_y = float((sy0 + np.mean(yy)) / h)
        else:
            start_center_x = None
            start_center_y = None
    else:
        start_visible = False
        start_center_x = None
        start_center_y = None

    hx0, hx1 = int(0.02 * w), int(0.36 * w)
    hy0, hy1 = int(0.01 * h), int(0.16 * h)
    hud_roi = rgb[hy0:hy1, hx0:hx1]
    if hud_roi.size > 0:
        hr = hud_roi[:, :, 0]
        hg = hud_roi[:, :, 1]
        hb = hud_roi[:, :, 2]
        hud_visible = int(((hr > 150) & (hg > 120) & (hb < 140)).sum()) > 18
    else:
        hud_visible = False

    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    sample = gray[int(0.12 * h) : int(0.92 * h), int(0.36 * w) : int(0.98 * w)]
    if sample.size > 0:
        dark_thr = float(np.clip(np.percentile(sample, 22) - 8.0, 24.0, 145.0))
    else:
        dark_thr = 70.0

    y0, y1 = int(0.10 * h), int(0.94 * h)
    search_x0 = int(0.42 * w)
    obstacle_x: float | None = None
    for x in range(search_x0, w):
        col = gray[y0:y1:2, x]
        if col.size == 0:
            continue
        dark_frac = float(np.mean(col < dark_thr))
        if x > search_x0:
            prev_col = gray[y0:y1:2, x - 1]
            edge_frac = float(np.mean(np.abs(col - prev_col) > 24.0))
        else:
            edge_frac = 0.0
        score = 0.72 * dark_frac + 0.28 * edge_frac
        if dark_frac > 0.16 and score > 0.24:
            obstacle_x = float(x)
            break

    gap_y: float | None = None
    if obstacle_x is not None:
        ox = int(obstacle_x)
        x0 = max(0, ox - 2)
        x1 = min(w - 1, ox + 2)
        lo = int(0.12 * h)
        hi = int(0.94 * h)
        blocked = np.any(gray[lo:hi, x0 : x1 + 1] < dark_thr, axis=1)
        open_rows = ~blocked
        best_len = 0
        best_start = 0
        cur_len = 0
        cur_start = 0
        for idx, is_open in enumerate(open_rows.tolist()):
            if is_open:
                if cur_len == 0:
                    cur_start = idx
                cur_len += 1
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
            else:
                cur_len = 0
        if best_len > int(0.09 * h):
            gap_y = float(lo + best_start + best_len / 2.0)

    patch_ranges = [
        (int(0.01 * w), int(0.06 * w)),
        (int(0.07 * w), int(0.12 * w)),
        (int(0.13 * w), int(0.17 * w)),
    ]
    py0, py1 = int(0.16 * h), int(0.62 * h)
    patches = []
    for x0, x1 in patch_ranges:
        roi = rgb[py0:py1:3, x0:x1:3]
        if roi.size > 0:
            patches.append(roi.reshape(-1, 3))
    if patches:
        stacked = np.concatenate(patches, axis=0).astype(np.float32)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        gray_patch = (0.299 * stacked[:, 0] + 0.587 * stacked[:, 1] + 0.114 * stacked[:, 2]).astype(np.float32)
        q25, q60 = np.percentile(gray_patch, [25, 60]).astype(np.float32)
        map_sig = [
            float(mean[0]),
            float(mean[1]),
            float(mean[2]),
            float(std[0]),
            float(std[1]),
            float(std[2]),
            float(q25),
            float(q60),
        ]
    else:
        map_sig = [0.0] * 8

    return {
        "width": int(w),
        "height": int(h),
        "birdY": bird_y,
        "gapY": gap_y,
        "obstacleX": obstacle_x,
        "hudVisible": bool(hud_visible),
        "startVisible": bool(start_visible),
        "startCenterX": start_center_x,
        "startCenterY": start_center_y,
        "replayVisible": False,
        "mapSig": map_sig,
    }


def update_stacked_state(
    history: deque[np.ndarray],
    new_state: np.ndarray,
    state_stack: int,
) -> np.ndarray:
    while len(history) < state_stack:
        history.append(new_state.copy())
    history.append(new_state.copy())
    while len(history) > state_stack:
        history.popleft()
    return np.concatenate(list(history), axis=0).astype(np.float32)
