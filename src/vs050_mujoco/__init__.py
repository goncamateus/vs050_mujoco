"""
vs050_mujoco – Gymnasium environments for the DENSO VS050 robot arm.
"""

import gymnasium as gym

from .envs.pick_and_place_env import PickAndPlaceEnv
from .envs.reach_pose_env import ReachPoseEnv

gym.register(
    id="VS050-PickAndPlace-v0",
    entry_point="vs050_mujoco.envs.pick_and_place_env:PickAndPlaceEnv",
    max_episode_steps=500,
)

gym.register(
    id="VS050-ReachPose-v0",
    entry_point="vs050_mujoco.envs.reach_pose_env:ReachPoseEnv",
    max_episode_steps=500,
)

__all__ = ["PickAndPlaceEnv", "ReachPoseEnv"]
