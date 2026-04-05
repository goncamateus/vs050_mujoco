"""
ReachPose Gymnasium Environment for DENSO VS050 robot arm.

A 6-DoF robotic arm must reach a target 3D position with its end-effector.
Designed for use with Hindsight Experience Replay (HER):
  - Implements gymnasium.GoalEnv (Dict obs with achieved_goal / desired_goal).
  - Provides a reward function that HER can re-label.
"""
from __future__ import annotations

import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


# gymnasium.GoalEnv was removed in v1.0; provide minimal compatible interface.
class _GoalEnvCompat(gym.Env):
    """Compatibility shim for gymnasium.GoalEnv (removed in gymnasium 1.0+)."""
    def compute_reward(
        self, achieved_goal, desired_goal, info
    ) -> np.ndarray | float: ...


# Path to the MuJoCo XML scene
_SCENE_XML = os.path.join(os.path.dirname(__file__), "..", "models", "scene_reach.xml")

_N_ARM_JOINTS = 6
_HOME_QPOS = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float64)

# Reachable workspace volume (meters) — conservative envelope for the VS050
_WORKSPACE_LOW = np.array([-0.45, -0.45, 0.08], dtype=np.float64)
_WORKSPACE_HIGH = np.array([0.45, 0.45, 0.55], dtype=np.float64)

_SUCCESS_DIST = 0.04  # 4 cm


class ReachPoseEnv(_GoalEnvCompat):
    """
    Gymnasium GoalEnv for a reach-pose task with the DENSO VS050 arm.

    **Observation** (Dict):
        - observation (27): joint pos (6) + joint vel (6) + end-effector xyz (3)
                           + control target (6) + target-visual xyz (3) +
                           + target_site xyz (3) + padding (3)
        - achieved_goal (3): end-effector position in world frame
        - desired_goal (3): target position in world frame

    **Action** (6-dim float32, clipped to [-1, 1]):
        - delta joint targets (6) -> scaled by max_delta per joint

    **Reward** (sparse + shaped):
        r = -||achieved - desired|| + 10.0 when success

    **Termination**:
        - Success: end-effector within _SUCCESS_DIST of desired_goal
        - Truncation: step limit exceeded
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        max_delta_per_joint: list[float] | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Load model
        xml_path = os.path.abspath(_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self._target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker"
        )
        self._target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site"
        )

        # Joint addresses
        self._arm_qpos_addrs = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
            ]
            for i in range(_N_ARM_JOINTS)
        ]
        self._arm_qvel_addrs = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i+1}")
            ]
            for i in range(_N_ARM_JOINTS)
        ]

        # Actuator addresses
        self._arm_act_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint_{i+1}"
            )
            for i in range(_N_ARM_JOINTS)
        ]
        self._arm_ctrl_lo = np.array([
            self.model.actuator_ctrlrange[idx, 0] for idx in self._arm_act_ids
        ])
        self._arm_ctrl_hi = np.array([
            self.model.actuator_ctrlrange[idx, 1] for idx in self._arm_act_ids
        ])

        # Per-joint action scale
        self._max_delta = np.array(
            max_delta_per_joint if max_delta_per_joint else [0.08, 0.08, 0.08, 0.10, 0.10, 0.15],
            dtype=np.float64,
        )

        # ---- Observation spaces (GoalEnv Dict) ----
        obs_dim = 27  # joint_pos(6) + joint_vel(6) + ee_pos(3) + ctrl(6) + target_geom(3) + target_site(3)
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
            )
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_ARM_JOINTS,), dtype=np.float32
        )

        # MuJoCo viewer / renderer (lazy init)
        self._viewer = None
        self._renderer = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        mujoco.mj_resetData(self.model, self.data)

        # Arm to home
        for i, addr in enumerate(self._arm_qpos_addrs):
            self.data.qpos[addr] = _HOME_QPOS[i]
        for i, idx in enumerate(self._arm_act_ids):
            self.data.ctrl[idx] = _HOME_QPOS[i]

        # Sample a new goal inside the workspace
        rng = self.np_random
        goal = rng.uniform(_WORKSPACE_LOW, _WORKSPACE_HIGH)
        self._set_goal(goal)

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Delta joint targets
        arm_delta = action * self._max_delta
        current_targets = np.array([
            self.data.ctrl[idx] for idx in self._arm_act_ids
        ])
        new_targets = np.clip(
            current_targets + arm_delta, self._arm_ctrl_lo, self._arm_ctrl_hi
        )
        for i, idx in enumerate(self._arm_act_ids):
            self.data.ctrl[idx] = new_targets[i]

        # 5 physics steps per env step => 0.01 s
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        achieved = self._get_achieved_goal()
        desired = obs["desired_goal"]
        reward = self.compute_reward(achieved, desired, {})
        success = self._check_success(achieved, desired)

        terminated = success
        truncated = self._step_count >= self.max_episode_steps

        info = self._get_info()
        info["is_success"] = success

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                cam = self._viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = [0.0, 0.0, 0.3]
                cam.distance = 1.8
                cam.azimuth = 135.0
                cam.elevation = -20.0
                self._viewer.sync()
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    # ------------------------------------------------------------------
    # GoalEnv API
    # ------------------------------------------------------------------

    def _get_achieved_goal(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_obs(self) -> dict:
        joint_pos = np.array([self.data.qpos[a] for a in self._arm_qpos_addrs])
        joint_vel = np.array([self.data.qvel[a] for a in self._arm_qvel_addrs])
        ee_pos = self._get_achieved_goal()
        ctrl_targets = np.array([self.data.ctrl[idx] for idx in self._arm_act_ids])
        target_geom_xyz = self.data.geom_xpos[self._target_geom_id].copy()
        target_site_xyz = self.data.site_xpos[self._target_site_id].copy()

        # Pad to a fixed dimension so observations can grow without breaking trained policies
        padding = np.zeros(27 - len(joint_pos) - len(joint_vel) - len(ee_pos)
                           - len(ctrl_targets) - len(target_geom_xyz) - len(target_site_xyz),
                           dtype=np.float64)
        obs_vec = np.concatenate([
            joint_pos, joint_vel, ee_pos, ctrl_targets,
            target_geom_xyz, target_site_xyz, padding,
        ]).astype(np.float32)

        return dict(
            observation=obs_vec,
            achieved_goal=ee_pos.astype(np.float32),
            desired_goal=target_geom_xyz.astype(np.float32),
        )

    def _get_info(self) -> dict:
        return {
            "ee_pos": self._get_achieved_goal().tolist(),
            "goal": self.data.geom_xpos[self._target_geom_id].copy().tolist(),
            "step": self._step_count,
        }

    def compute_reward(
        self, achieved_goal, desired_goal, info
    ) -> np.ndarray | float:
        """
        Reward used by the GoalEnv API. Must accept both single and batched goals.
        Used by SB3 HER to re-label transitions with different desired goals.
        """
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        reward = -d
        if np.isscalar(d) or d.shape == ():
            if d < _SUCCESS_DIST:
                reward = 10.0  # sparse success bonus (single transition)
        else:
            reward = np.where(d < _SUCCESS_DIST, 10.0, reward)
        return reward

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_goal(self, goal: np.ndarray):
        """Place the goal marker geom and site at the given world position."""
        self.model.geom_pos[self._target_geom_id] = goal
        mujoco.mj_forward(self.model, self.data)

    def _check_success(self, achieved: np.ndarray, desired: np.ndarray) -> bool:
        return bool(np.linalg.norm(achieved - desired) < _SUCCESS_DIST)
