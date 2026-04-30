"""ReachPose Gymnasium Environment for DENSO VS050 robot arm.

Refactored to inherit from gymnasium.envs.mujoco.MujocoEnv.
Uses potential-based reward shaping (Ng, 1999).
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv

# Path to the MuJoCo XML scene
_SCENE_XML = os.path.join(os.path.dirname(__file__), "..", "models", "scene_reach.xml")

# Robot joints (6-DoF arm)
_N_ARM_JOINTS = 6

# Home qpos for arm joints (from keyframe)
_HOME_QPOS = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float64)

# End-effector workspace limits (metres)
_WORKSPACE_LOW = np.array([-0.45, -0.45, 0.08], dtype=np.float64)
_WORKSPACE_HIGH = np.array([0.45, 0.45, 0.55], dtype=np.float64)

DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array([0.0, 0.0, 0.3]),
    "distance": 1.8,
    "azimuth": 135.0,
    "elevation": -20.0,
}

SUCCESS_REWARD = 100.0


class ReachPoseEnv(MujocoEnv, utils.EzPickle):
    """Reach-pose task with the DENSO VS050 arm.

    **Observation** (flat Box):
        - joint positions  (nq)
        - joint velocities (nv)
        - end-effector pos (3)
        - goal position    (3)
        - ee-to-goal vec   (3)

    **Action** (6-dim float32, clipped to [-1, 1]):
        - delta joint targets (6)  → scaled per joint by _max_delta

    **Reward** (dense):
        r = -d(ee → goal)
          - ctrl_cost
          + success_bonus

    **Termination**:
        - Success: end-effector within success_dist of goal
        - Truncation: handled by TimeLimit wrapper (max_episode_steps)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    # ===================================================================
    # Construction
    # ===================================================================

    def __init__(
        self,
        xml_file: str = _SCENE_XML,
        frame_skip: int = 5,
        default_camera_config: dict[str, float | np.ndarray] = DEFAULT_CAMERA_CONFIG,
        max_delta_per_joint: list[float] | None = None,
        success_dist: float = 0.01,
        success_reward: float = SUCCESS_REWARD,
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
        self._init_cache_ids()
        self._init_state(
            max_delta_per_joint,
            success_dist,
            success_reward,
            ctrl_cost_weight,
            reset_noise_scale,
        )

        obs_dim = self.data.qpos.size + self.data.qvel.size + 9
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_ARM_JOINTS,), dtype=np.float32
        )
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.reward_info = {}

    # -------------------------------------------------------------------
    # Sub-initialisers
    # -------------------------------------------------------------------

    def _init_cache_ids(self):
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

    def _init_state(
        self,
        max_delta_per_joint,
        success_dist,
        success_reward,
        ctrl_cost_weight,
        reset_noise_scale,
    ):
        self.init_qpos[self._arm_qpos_addrs] = _HOME_QPOS
        self.init_qvel[self._arm_qvel_addrs] = 0.0

        self._max_delta = np.array(
            max_delta_per_joint
            if max_delta_per_joint
            else [0.08, 0.08, 0.08, 0.10, 0.10, 0.15],
            dtype=np.float64,
        )
        self.success_dist = success_dist
        self.success_reward = success_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._step_count = 0

    # ===================================================================
    # MuJoCo helpers
    # ===================================================================

    def _get_id(self, obj_type, name: str) -> int:
        """Get MuJoCo ID or raise ValueError."""
        id_ = mujoco.mj_name2id(self.model, obj_type, name)
        if id_ < 0:
            raise ValueError(f"Name '{name}' not found in model for type {obj_type}")
        return id_

    # ===================================================================
    # Observation helpers
    # ===================================================================

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._get_ee_pos()
        goal_pos = self._get_goal_pos()
        return np.concatenate(
            [
                self.data.qpos.copy(),
                self.data.qvel.copy(),
                ee_pos,
                goal_pos,
                goal_pos - ee_pos,
            ]
        ).astype(np.float32)

    # ===================================================================
    # Kinematics helpers
    # ===================================================================

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_goal_pos(self) -> np.ndarray:
        return self.data.geom_xpos[self._target_geom_id].copy()

    def _set_goal(self, goal: np.ndarray):
        self.model.geom_pos[self._target_geom_id] = goal
        mujoco.mj_forward(self.model, self.data)

    # ===================================================================
    # Reward helpers
    # ===================================================================

    def _compute_reward(self, action: np.ndarray) -> float:
        ee_pos = self._get_ee_pos()
        goal_pos = self._get_goal_pos()
        dist = float(np.linalg.norm(ee_pos - goal_pos))
        ctrl_cost = float(self.ctrl_cost_weight * np.sum(np.square(action)))
        success = self._check_success(dist)

        reward = -dist - ctrl_cost + (self.success_reward if success else 0.0)

        self.reward_info["distance"] = dist
        self.reward_info["ctrl_cost"] = ctrl_cost
        self.reward_info["success"] = success

        return reward

    def _check_success(self, dist: float) -> bool:
        return dist < self.success_dist

    # ===================================================================
    # Info helpers
    # ===================================================================

    def _get_reset_info(self) -> dict:
        return {
            "ee_pos": self._get_ee_pos().tolist(),
            "goal": self._get_goal_pos().tolist(),
            "step": self._step_count,
        }

    # ===================================================================
    # Actuator helpers
    # ===================================================================

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

    # ===================================================================
    # Core Gymnasium API
    # ===================================================================

    def reset_model(self):
        self._reset_simulation()
        self._reset_arm()
        self._reset_goal()
        self._step_count = 0
        self.reward_info = {}
        return self._get_obs()

    def _reset_simulation(self):
        noise = self.np_random.uniform(
            -self._reset_noise_scale,
            self._reset_noise_scale,
            size=self.model.nq,
        )
        self.set_state(self.init_qpos + noise, self.init_qvel + noise)

    def _reset_arm(self):
        self.data.ctrl[self._arm_act_ids] = np.clip(
            self.data.qpos[self._arm_qpos_addrs],
            self._arm_ctrl_lo,
            self._arm_ctrl_hi,
        )

    def _reset_goal(self):
        goal = self.np_random.uniform(_WORKSPACE_LOW, _WORKSPACE_HIGH)
        self._set_goal(goal)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._build_control(action)
        self.do_simulation(ctrl, self.frame_skip)
        self._step_count += 1
        return self._build_step_return(action)

    def _build_control(self, action: np.ndarray) -> np.ndarray:
        arm_delta = action * self._max_delta
        current_targets = self.data.ctrl[self._arm_act_ids].copy()
        new_targets = np.clip(
            current_targets + arm_delta,
            self._arm_ctrl_lo,
            self._arm_ctrl_hi,
        )
        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_act_ids] = new_targets
        return ctrl

    def _build_step_return(self, action: np.ndarray):
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = bool(self.reward_info["success"])
        truncated = False

        info = self._get_reset_info()
        info.update(self.reward_info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
