from __future__ import annotations

import unittest
import urllib.request

from local_tapcap_state_env import LocalTapCapStateEnv


def url_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


@unittest.skipUnless(url_reachable("http://127.0.0.1:5173"), "Local webgame is not running on 127.0.0.1:5173")
class LocalEnvLifecycleTests(unittest.TestCase):
    def test_reset_step_close_smoke(self) -> None:
        env = LocalTapCapStateEnv(
            url="http://127.0.0.1:5173",
            headless=True,
            step_dt=0.03,
            action_repeat=1,
            state_stack=2,
            dr_level="moderate",
        )
        try:
            obs, _ = env.reset(seed=123)
            self.assertEqual(obs.shape[0], env.observation_space.shape[0])
            for _ in range(25):
                obs, reward, done, truncated, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.shape[0], env.observation_space.shape[0])
                self.assertFalse(truncated)
                self.assertTrue(isinstance(float(reward), float))
                if done:
                    obs, _ = env.reset()
                    self.assertEqual(obs.shape[0], env.observation_space.shape[0])
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
