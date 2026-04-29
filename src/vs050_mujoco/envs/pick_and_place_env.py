"""Pick-and-Place Gymnasium Environment for DENSO VS050 + Robotiq 2F-85.

Refactored to inherit from gymnasium.envs.mujoco.MujocoEnv.
Single-object variant.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

# Path to the MuJoCo XML scene
_SCENE_XML = os.path.join(
    os.path.dirname(__file__), "..", "models", "pick_and_place_scene.xml"
)

# Robot joints (6-DoF arm + 1 gripper tendon actuator = 7 total)
_N_ARM_JOINTS = 6
_N_ACTUATORS = 7  # 6 arm + 1 gripper

# Home qpos for arm joints (from keyframe)
_HOME_QPOS = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float64)

# Cage work-surface limits (XY, metres)
_XY_LOW = -0.55
_XY_HIGH = 0.55
_Z_OBJ = 0.03  # initial z for objects (on the wood floor)

# Object half-size (cube)
_OBJ_HALF = 0.025

# Target zone (should match target_site in XML)
_TARGET_POS = np.array([0.30, 0.30, _OBJ_HALF], dtype=np.float64)
_SUCCESS_DIST = 0.01  # 1 cm
_GRASP_DIST = 0.05  # 5 cm
_FINGER_THRESHOLD = 112  # gripper command threshold for "closed" state (0-255)

# Exclude zone around robot base
_ROBOT_EXCLUSION_R = 0.18

DEFAULT_CAMERA_CONFIG = {
    "lookat": np.array([0.0, 0.0, 0.5]),
    "distance": 3.0,
    "azimuth": 125.0,
    "elevation": -15.0,
}

SUCCESS_REWARD = 100.0


class PickAndPlaceEnv(MujocoEnv, utils.EzPickle):
    """Pick-and-place task with the DENSO VS050 arm and Robotiq 2F-85 gripper.

    **Observation** (23-dim float32):
        - joint positions  (6)
        - joint velocities (6)
        - gripper opening  (1)  – normalised in [0, 1]
        - object position  (3)
        - object quat      (4)
        - target position  (3)
                        = 23-dim

    **Action** (7-dim float32, clipped to [-1, 1]):
        - delta joint targets (6)  → scaled by 0.05 rad/step
        - gripper command     (1)  → mapped to [0, 255] (0=open, 1=closed)

    **Reward** (dense):
        r = -d(pinch → object)
          + grasp_bonus
          - d(object → target)
          + success_bonus

    **Termination**:
        - Success: object within _SUCCESS_DIST of target
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
        target_pos: np.ndarray | None = None,
        reset_noise_scale: float = 0.0,
        **kwargs,
    ):
        self._init_ezpickle(
            xml_file,
            frame_skip,
            default_camera_config,
            target_pos,
            reset_noise_scale,
            **kwargs,
        )
        self._init_mujoco_env(xml_file, frame_skip, default_camera_config, **kwargs)
        self._init_cache_ids()
        self._init_state(target_pos, reset_noise_scale)

        self._is_grasped = False  # track grasp state
        self._stage = "REACH"  # reward stage: REACH, GRASP, PLACE

        obs_dim = _N_ARM_JOINTS + _N_ARM_JOINTS + 1 + 3 + 4 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_ACTUATORS,), dtype=np.float32
        )
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.reward_info = {}

    # -------------------------------------------------------------------
    # Sub-initialisers
    # -------------------------------------------------------------------

    def _init_ezpickle(self, xml_file, frame_skip, camera, target_pos, noise, **kw):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            camera,
            target_pos,
            noise,
            **kw,
        )

    def _init_mujoco_env(self, xml_file, frame_skip, camera, **kw):
        fullpath = os.path.abspath(xml_file)
        MujocoEnv.__init__(
            self,
            fullpath,
            frame_skip,
            observation_space=None,
            default_camera_config=camera,
            **kw,
        )

    def _init_cache_ids(self):
        self._pinch_id = self._get_id(mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self._target_id = self._get_id(mujoco.mjtObj.mjOBJ_SITE, "target_site")

        self._obj_body_id = self._get_id(mujoco.mjtObj.mjOBJ_BODY, "object0")
        self._obj_qpos_addr = self.model.jnt_qposadr[
            self._get_id(mujoco.mjtObj.mjOBJ_JOINT, "object0_joint")
        ]
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
        self._gripper_act_id = self._get_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
        )
        self._arm_act_ids = [
            self._get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint_{i + 1}")
            for i in range(_N_ARM_JOINTS)
        ]

    def _init_state(self, target_pos, reset_noise_scale):
        self.init_qpos[self._arm_qpos_addrs] = _HOME_QPOS
        self.init_qvel[self._arm_qvel_addrs] = 0.0

        self._target_pos = (
            np.array(target_pos, dtype=np.float64)
            if target_pos is not None
            else _TARGET_POS.copy()
        )
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
        return np.concatenate(
            [
                self._get_joint_obs(),
                self._get_gripper_obs(),
                self._get_object_obs(),
                self._get_target_obs(),
            ]
        ).astype(np.float32)

    def _get_joint_obs(self) -> np.ndarray:
        pos = [self.data.qpos[a] for a in self._arm_qpos_addrs]
        vel = [self.data.qvel[a] for a in self._arm_qvel_addrs]
        return np.concatenate([np.array(pos), np.array(vel)])

    def _get_gripper_obs(self) -> np.ndarray:
        return np.array([self.data.ctrl[self._gripper_act_id] / 255.0])

    def _get_object_obs(self) -> np.ndarray:
        addr = self._obj_qpos_addr
        obj_pos = self.data.qpos[addr : addr + 3]
        obj_quat = self.data.qpos[addr + 3 : addr + 7]
        return np.concatenate([obj_pos, obj_quat])

    def _get_target_obs(self) -> np.ndarray:
        return self._target_pos

    # ===================================================================
    # Kinematics helpers
    # ===================================================================

    def _get_pinch_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._pinch_id].copy()

    def _get_obj_pos(self) -> np.ndarray:
        addr = self._obj_qpos_addr
        return self.data.qpos[addr : addr + 3].copy()

    # ===================================================================
    # Reward helpers
    # ===================================================================

    def _compute_reward(self) -> float:
        pinch_pos = self._get_pinch_pos()
        obj_pos = self._get_obj_pos()
        dist_pinch_obj = float(np.linalg.norm(pinch_pos - obj_pos))
        is_grasped = int(self._check_grasp(pinch_pos, obj_pos))
        dist_place = float(np.linalg.norm(obj_pos - self._target_pos))
        success = self._check_success(obj_pos)

        # Stage transition logic (monotonic)
        is_closed = float(self.data.ctrl[self._gripper_act_id]) > _FINGER_THRESHOLD
        if self._stage == "REACH" and dist_pinch_obj < _GRASP_DIST and is_closed:
            self._stage = "GRASP"
        elif self._stage == "GRASP" and is_grasped:
            self._stage = "PLACE"

        # Staged reward calculation
        if self._stage == "REACH":
            reward = -dist_pinch_obj
        elif self._stage == "GRASP":
            reward = -dist_pinch_obj + 0.1
        else:  # PLACE
            reward = -2.0 * dist_place + (0.2 if is_grasped else 0.0)

        self.reward_info["dist_pinch_obj"] = dist_pinch_obj
        self.reward_info["dist_place"] = dist_place
        self.reward_info["is_grasped"] = is_grasped
        self.reward_info["success"] = int(success)
        self.reward_info["stage"] = self._stage

        return SUCCESS_REWARD if success else reward

    def _check_grasp(self, pinch_pos: np.ndarray, obj_pos: np.ndarray) -> bool:
        is_close = float(np.linalg.norm(pinch_pos - obj_pos)) < _GRASP_DIST
        is_closed = float(self.data.ctrl[self._gripper_act_id]) > _FINGER_THRESHOLD
        return is_close and is_closed

    def _check_success(self, obj_pos: np.ndarray) -> bool:
        """True if object is within _SUCCESS_DIST of target."""
        is_open = float(self.data.ctrl[self._gripper_act_id]) <= _FINGER_THRESHOLD
        is_placed = float(np.linalg.norm(obj_pos - self._target_pos)) < _SUCCESS_DIST
        return is_open and is_placed

    # ===================================================================
    # Info helpers
    # ===================================================================

    def _get_reset_info(self) -> dict:
        return {
            "pinch_pos": self._get_pinch_pos().tolist(),
            "obj_position": self._get_obj_pos().tolist(),
            "target_pos": self._target_pos.tolist(),
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
        self._reset_gripper()
        self._reset_object()
        self._step_count = 0
        self._is_grasped = False
        self._stage = "REACH"
        self.reward_info = {}
        return self._get_obs()

    def _reset_simulation(self):
        qpos_noise = self.np_random.uniform(
            -self._reset_noise_scale,
            self._reset_noise_scale,
            size=self.model.nq,
        )
        qvel_noise = self.np_random.uniform(
            -self._reset_noise_scale,
            self._reset_noise_scale,
            size=self.model.nv,
        )
        self.set_state(self.init_qpos + qpos_noise, self.init_qvel + qvel_noise)

    def _reset_arm(self):
        self.data.ctrl[self._arm_act_ids] = np.clip(
            self.data.qpos[self._arm_qpos_addrs],
            self._arm_ctrl_lo,
            self._arm_ctrl_hi,
        )

    def _reset_gripper(self):
        self.data.ctrl[self._gripper_act_id] = 0.0

    def _reset_object(self):
        xy = self._sample_object_xy()
        self._place_object_at(xy)
        mujoco.mj_forward(self.model, self.data)

    def _sample_object_xy(self) -> np.ndarray:
        for _ in range(1000):  # rejection sampling
            xy = self.np_random.uniform(
                _XY_LOW + _OBJ_HALF, _XY_HIGH - _OBJ_HALF, size=2
            )
            if self._is_invalid_spawn(xy):
                continue
            return xy
        # Fallback
        return np.array([0.25, -0.20])

    def _is_invalid_spawn(self, xy: np.ndarray) -> bool:
        if np.linalg.norm(xy) < _ROBOT_EXCLUSION_R:
            return True
        if np.linalg.norm(xy - self._target_pos[:2]) < _OBJ_HALF * 3:
            return True
        return False

    def _place_object_at(self, xy: np.ndarray):
        addr = self._obj_qpos_addr
        self.data.qpos[addr : addr + 3] = [xy[0], xy[1], _Z_OBJ]
        self.data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._build_control(action)
        self.do_simulation(ctrl, self.frame_skip)
        self._step_count += 1
        return self._build_step_return()

    def _build_control(self, action: np.ndarray) -> np.ndarray:
        arm_delta = action[:_N_ARM_JOINTS] * 0.05  # rad
        gripper_cmd = (action[6] + 1.0) / 2.0 * 255.0  # [0, 255]

        current_targets = self.data.ctrl[self._arm_act_ids].copy()
        new_targets = np.clip(
            current_targets + arm_delta,
            self._arm_ctrl_lo,
            self._arm_ctrl_hi,
        )

        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_act_ids] = new_targets
        ctrl[self._gripper_act_id] = gripper_cmd
        return ctrl

    def _build_step_return(self):
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = reward >= 100.0
        truncated = False

        info = self._get_reset_info()
        info.update(self.reward_info)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
