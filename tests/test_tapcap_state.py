from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tapcap_state import BASE_STATE_DIM, build_state_vector, extract_features_from_rgb, parse_tapcap_detection


ROOT = Path(__file__).resolve().parents[1]


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


class TapCapStateTests(unittest.TestCase):
    def test_menu_frame_detects_start_button(self) -> None:
        frame = load_rgb(ROOT / "tapcap_initial.png")
        feat = extract_features_from_rgb(frame)
        self.assertTrue(bool(feat["startVisible"]))
        self.assertIsNotNone(feat["birdY"])

    def test_ingame_frame_detects_obstacle_and_no_start_button(self) -> None:
        frame = load_rgb(ROOT / "tapcap_after_taps.png")
        feat = extract_features_from_rgb(frame)
        self.assertFalse(bool(feat["startVisible"]))
        self.assertIsNotNone(feat["obstacleX"])
        self.assertIsNotNone(feat["gapY"])

    def test_map_signature_is_stable_on_same_frame(self) -> None:
        frame = load_rgb(ROOT / "tapcap_after_start_click.png")
        feat_a = extract_features_from_rgb(frame)
        feat_b = extract_features_from_rgb(frame)
        sig_a = np.asarray(feat_a["mapSig"], dtype=np.float32)
        sig_b = np.asarray(feat_b["mapSig"], dtype=np.float32)
        self.assertEqual(sig_a.shape, (8,))
        self.assertTrue(np.allclose(sig_a, sig_b, atol=1e-6))

    def test_state_vector_shape_and_finiteness(self) -> None:
        frame_a = load_rgb(ROOT / "tapcap_after_start_click.png")
        frame_b = load_rgb(ROOT / "tapcap_after_taps.png")
        det_a = parse_tapcap_detection(extract_features_from_rgb(frame_a))
        det_b = parse_tapcap_detection(extract_features_from_rgb(frame_b))
        self.assertIsNotNone(det_a)
        self.assertIsNotNone(det_b)
        assert det_a is not None
        assert det_b is not None
        vec = build_state_vector(det_b, previous=det_a, step_dt=0.05)
        self.assertEqual(vec.shape, (BASE_STATE_DIM,))
        self.assertTrue(np.all(np.isfinite(vec)))


if __name__ == "__main__":
    unittest.main()
