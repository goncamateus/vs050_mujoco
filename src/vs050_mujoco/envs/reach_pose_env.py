"""
ReachPose Gymnasium Environment for DENSO VS050 robot arm.

Refactored to inherit from gymnasium.envs.mujoco.MujocoEnv.
Uses potential-based reward shaping (Ng, 1999).
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv

_SCENE_XML = os.path.join(os.path.dirname(__file__), "..", "models", "scene_reach.xml")

_N_ARM_JOINTS = 6
_HOME_QPOS = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float64)

_WORKSPACE_LOW = np.array([-0.45, -0.45, 0.08], dtype=np.float64)
_WORKSPACE_HIGH = np.array([0.45, 0.45, 0.55], dtype=np.float64)

DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array([0.0, 0.0, 0.3]),
    "distance": 1.8,
    "azimuth": 135.0,
    "elevation": -20.0,
}


class ReachPoseEnv(MujocoEnv, utils.EzPickle):
    """
    VS050 reach-pose environment using MujocoEnv.

    Observation (flat Box):
        [qpos, qvel, ee_pos, goal_pos, ee_to_goal]

    Action (Box(-1, 1, (6,), float32)):
        Delta joint targets scaled per joint.

    Reward:
        Potential-based shaping + control cost + success bonus.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = _SCENE_XML,
        frame_skip: int = 5,
        default_camera_config: dict[str, float | np.ndarray] = DEFAULT_CAMERA_CONFIG,
        max_delta_per_joint: list[float] | None = None,
        gamma_shaping: float = 1.00,
        success_dist: float = 0.01,
        success_reward: float = 100.0,
        ctrl_cost_weight: float = 1e-3,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            max_delta_per_joint,
            gamma_shaping,
            success_dist,
            success_reward,
            ctrl_cost_weight,
            reset_noise_scale,
            **kwargs,
        )

        fullpath = os.path.abspath(xml_file)
        MujocoEnv.__init__(
            self,
            fullpath,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # Cache MuJoCo IDs
        self._ee_site_id = self._get_id(mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self._target_geom_id = self._get_id(mujoco.mjtObj.mjOBJ_GEOM, "target_marker")
        self._target_site_id = self._get_id(mujoco.mjtObj.mjOBJ_SITE, "target_site")

        self._arm_qpos_addrs = [
            self.model.jnt_qposadr[
                self._get_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i + 1}")
            ]
            for i in range(_N_ARM_JOINTS)
        ]
        self._arm_qvel_addrs = [
            self.model.jnt_dofadr[
                self._get_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i + 1}")
            ]
            for i in range(_N_ARM_JOINTS)
        ]
        self._arm_act_ids = [
            self._get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint_{i + 1}")
            for i in range(_N_ARM_JOINTS)
        ]

        # Override init state to home position
        self.init_qpos[self._arm_qpos_addrs] = _HOME_QPOS
        self.init_qvel[self._arm_qvel_addrs] = 0.0

        # Per-joint action scale
        self._max_delta = np.array(
            max_delta_per_joint
            if max_delta_per_joint
            else [0.08, 0.08, 0.08, 0.10, 0.10, 0.15],
            dtype=np.float64,
        )

        # Parameters
        self.gamma_shaping = gamma_shaping
        self.success_dist = success_dist
        self.success_reward = success_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale

        # Override action space: delta control in [-1, 1]^6
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_ARM_JOINTS,), dtype=np.float32
        )

        # Observation space: qpos + qvel + ee_pos(3) + goal_pos(3) + rel_pos(3)
        obs_dim = self.data.qpos.size + self.data.qvel.size + 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def _get_id(self, obj_type, name: str) -> int:
        """Get MuJoCo ID or raise ValueError."""
        id_ = mujoco.mj_name2id(self.model, obj_type, name)
        if id_ < 0:
            raise ValueError(f"Name '{name}' not found in model for type {obj_type}")
        return id_

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_goal_pos(self) -> np.ndarray:
        return self.data.geom_xpos[self._target_geom_id].copy()

    def _set_goal(self, goal: np.ndarray):
        self.model.geom_pos[self._target_geom_id] = goal
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        ee_pos = self._get_ee_pos()
        goal_pos = self._get_goal_pos()
        rel_pos = goal_pos - ee_pos
        return np.concatenate([qpos, qvel, ee_pos, goal_pos, rel_pos]).astype(
            np.float32
        )

    def _get_reset_info(self) -> dict:
        return {
            "ee_pos": self._get_ee_pos().tolist(),
            "goal": self._get_goal_pos().tolist(),
        }

    def reset_model(self):
        noise = self.np_random.uniform(
            -self._reset_noise_scale,
            self._reset_noise_scale,
            size=self.model.nq,
        )
        qpos = self.init_qpos + noise
        qvel = self.init_qvel + noise

        self.set_state(qpos, qvel)

        # Set initial ctrl targets to current joint positions
        self.data.ctrl[self._arm_act_ids] = np.clip(
            qpos[self._arm_qpos_addrs], self._arm_ctrl_lo, self._arm_ctrl_hi
        )

        # Sample new goal
        goal = self.np_random.uniform(_WORKSPACE_LOW, _WORKSPACE_HIGH)
        self._set_goal(goal)

        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Previous distance for shaping
        ee_pos_prev = self._get_ee_pos()
        goal_pos = self._get_goal_pos()
        d_prev = np.linalg.norm(ee_pos_prev - goal_pos)

        # Delta control
        arm_delta = action * self._max_delta
        current_targets = self.data.ctrl[self._arm_act_ids].copy()
        new_targets = np.clip(
            current_targets + arm_delta, self._arm_ctrl_lo, self._arm_ctrl_hi
        )

        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_act_ids] = new_targets

        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        ee_pos = self._get_ee_pos()
        d_curr = np.linalg.norm(ee_pos - goal_pos)

        # Potential-based shaping: F = gamma * Phi(s') - Phi(s), Phi = -d
        shaping = d_prev - self.gamma_shaping * d_curr
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        success = d_curr < self.success_dist
        success_bonus = self.success_reward if success else 0.0

        reward = shaping - ctrl_cost + success_bonus

        terminated = success
        truncated = False

        info = {
            "distance": d_curr,
            "shaping": shaping,
            "ctrl_cost": ctrl_cost,
            "success": success,
            "ee_pos": ee_pos.tolist(),
            "goal": goal_pos.tolist(),
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    @property
    def _arm_ctrl_lo(self) -> np.ndarray:
        return np.array(
            [self.model.actuator_ctrlrange[idx, 0] for idx in self._arm_act_ids],
            dtype=np.float64,
        )

    @property
    def _arm_ctrl_hi(self) -> np.ndarray:
        return np.array(
            [self.model.actuator_ctrlrange[idx, 1] for idx in self._arm_act_ids],
            dtype=np.float64,
        )
