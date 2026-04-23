#!/usr/bin/env python3
"""Test script to verify VS050-ReachPose-v0 environment works correctly."""

import gymnasium as gym
import numpy as np

import vs050_mujoco  # noqa: F401  (registers environments)


def test_env():
    print("Creating VS050-ReachPose-v0 environment...")
    env = gym.make("VS050-ReachPose-v0", render_mode=None)

    # Check observation space (flat Box, not Dict)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray), "obs must be ndarray"
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != {env.observation_space.shape}"
    )
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")

    # Test a few steps
    print("\nTesting steps...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        print(
            f"Step {i}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()

    env.close()
    print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    test_env()
