"""
Smoke-test for VS050 Pick-and-Place environment.
Run from the repo root:
    python test_env.py
"""
import numpy as np

# ── Register the env ──────────────────────────────────────────────
import vs050_mujoco as _pkg  # noqa: F401  (registers VS050-PickAndPlace-v0)
import gymnasium as gym

# ── Tests ─────────────────────────────────────────────────────────

def test_basic():
    print("Creating environment …")
    env = gym.make("VS050-PickAndPlace-v0", render_mode=None)

    # ---- reset ----
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray), "obs must be ndarray"
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != {env.observation_space.shape}"
    )
    print(f"  reset OK  →  obs shape {obs.shape}")

    # ---- step ----
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        if terminated or truncated:
            obs, info = env.reset()

    print(f"  20 random steps OK")
    print(f"  last reward: {reward:.4f}")
    print(f"  last info: {info}")

    # ---- action space ----
    assert env.action_space.shape == (7,), "action space must be 7-dim"
    print(f"  action space: {env.action_space}")

    env.close()
    print("All checks passed! ✓")


def test_rgb_array():
    print("Testing rgb_array render …")
    env = gym.make("VS050-PickAndPlace-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame is not None, "render() must return an array in rgb_array mode"
    assert frame.ndim == 3 and frame.shape[2] == 3, "frame must be HxWx3"
    print(f"  rgb_array frame shape: {frame.shape}  ✓")
    env.close()

