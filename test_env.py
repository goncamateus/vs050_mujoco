#!/usr/bin/env python3
"""
Test script to verify VS050-ReachPose-v0 environment works correctly.
"""

import numpy as np
import gymnasium as gym
import vs050_mujoco  # This registers the environments


def test_env():
    print("Creating VS050-ReachPose-v0 environment...")
    env = gym.make("VS050-ReachPose-v0", render_mode=None)

    # Check observation space (should be Dict for GoalEnv)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"Observation keys: {obs.keys()}")
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Achieved goal shape: {obs['achieved_goal'].shape}")
    print(f"Desired goal shape: {obs['desired_goal'].shape}")

    # Test a few steps
    print("\nTesting steps...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
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
