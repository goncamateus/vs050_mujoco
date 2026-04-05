"""
Pick-and-Place Gymnasium Environment for DENSO VS050 + Robotiq 2F-85
"""
from __future__ import annotations

import os
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces


# Path to the MuJoCo XML scene
_SCENE_XML = os.path.join(os.path.dirname(__file__), "..", "models", "pick_and_place_scene.xml")

# Robot joints (6-DoF arm + 1 gripper tendon actuator = 7 total)
_N_ARM_JOINTS    = 6
_N_ACTUATORS     = 7          # 6 arm + 1 gripper
_N_OBJECTS       = 3

# Home qpos for arm joints (from keyframe)
_HOME_QPOS = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0], dtype=np.float64)

# Cage work-surface limits (XY, metres)
_XY_LOW  = -0.55
_XY_HIGH =  0.55
_Z_OBJ   =  0.03   # initial z for objects (on the wood floor)

# Object half-size (cube)
_OBJ_HALF = 0.025

# Minimum distance between objects at spawn
_MIN_OBJ_DIST = 0.12

# Target zone (should match target_site in XML)
_TARGET_POS = np.array([0.30, 0.30, _OBJ_HALF], dtype=np.float64)
_SUCCESS_DIST = 0.05   # 5 cm

# Exclude zone around robot base
_ROBOT_EXCLUSION_R = 0.18


class PickAndPlaceEnv(gym.Env):
    """
    Gymnasium environment for a pick-and-place task with the DENSO VS050
    robotic arm and Robotiq 2F-85 gripper.

    **Observation** (37-dim float32):
        - joint positions  (6)
        - joint velocities (6)
        - gripper opening  (1)  – normalised in [0, 1]
        - object positions (3 × 3 = 9)
        - object quats     (3 × 4 = 12)
        - target position  (3)
                        = 37-dim

    **Action** (7-dim float32, clipped to [-1, 1]):
        - delta joint targets (6)  → scaled by 0.05 rad/step
        - gripper command     (1)  → mapped to [0, 255] (0=open, 1=closed)

    **Reward** (dense):
        r = -d(pinch → nearest object)
          + 0.5 * grasping_bonus
          - d(nearest grasped object → target)
          + 10.0 * success

    **Termination**:
        - Success: any object within _SUCCESS_DIST of target
        - Truncation: step limit exceeded (max_episode_steps)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # -------------------------------------------------------------------
    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 500,
        target_pos: np.ndarray | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Custom target override
        self._target_pos = (
            np.array(target_pos, dtype=np.float64)
            if target_pos is not None
            else _TARGET_POS.copy()
        )

        # Load model
        xml_path = os.path.abspath(_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Cache body/site IDs
        self._pinch_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "pinch")
        self._target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,  "target_site")
        self._obj_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"object{i}")
            for i in range(_N_OBJECTS)
        ]
        # Joint qpos address for each free object body (7-DOF each: 3 pos + 4 quat)
        self._obj_qpos_addrs = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"object{i}_joint")
            ]
            for i in range(_N_OBJECTS)
        ]
        # Robot joint qpos addresses (hinge)
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
        # Gripper actuator index (the 2f85 "fingers_actuator")
        self._gripper_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator"
        )
        # Arm actuator indices
        self._arm_act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint_{i+1}")
            for i in range(_N_ARM_JOINTS)
        ]
        # Arm ctrl ranges for clipping
        self._arm_ctrl_lo = np.array([
            self.model.actuator_ctrlrange[idx, 0] for idx in self._arm_act_ids
        ])
        self._arm_ctrl_hi = np.array([
            self.model.actuator_ctrlrange[idx, 1] for idx in self._arm_act_ids
        ])

        # ---- Spaces ------------------------------------------------
        obs_dim = _N_ARM_JOINTS + _N_ARM_JOINTS + 1 + _N_OBJECTS * 3 + _N_OBJECTS * 4 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(_N_ACTUATORS,), dtype=np.float32
        )

        # MuJoCo viewer (lazy init)
        self._viewer = None
        self._renderer = None

    # -------------------------------------------------------------------
    # Core Gymnasium API
    # -------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)

        # Set arm to home position
        for i, addr in enumerate(self._arm_qpos_addrs):
            self.data.qpos[addr] = _HOME_QPOS[i]

        # Set arm actuators to home
        for i, idx in enumerate(self._arm_act_ids):
            self.data.ctrl[idx] = _HOME_QPOS[i]

        # Open gripper
        self.data.ctrl[self._gripper_act_id] = 0.0

        # Randomise object positions (no overlap, outside robot base radius)
        rng = self.np_random
        placed: list[np.ndarray] = []
        for obj_idx in range(_N_OBJECTS):
            for _ in range(1000):  # rejection sampling
                xy = rng.uniform(_XY_LOW + _OBJ_HALF, _XY_HIGH - _OBJ_HALF, size=2)
                # Exclude robot base footprint
                if np.linalg.norm(xy) < _ROBOT_EXCLUSION_R:
                    continue
                # Exclude target zone
                if np.linalg.norm(xy - self._target_pos[:2]) < _OBJ_HALF * 3:
                    continue
                # Exclude overlap with other objects
                too_close = any(
                    np.linalg.norm(xy - p) < _MIN_OBJ_DIST for p in placed
                )
                if too_close:
                    continue
                placed.append(xy)
                addr = self._obj_qpos_addrs[obj_idx]
                self.data.qpos[addr:addr+3]   = [xy[0], xy[1], _Z_OBJ]
                self.data.qpos[addr+3:addr+7] = [1.0, 0.0, 0.0, 0.0]  # unit quat
                break
            else:
                # Fallback: fixed scatter positions
                fallback = [
                    np.array([ 0.25, -0.20]),
                    np.array([-0.30,  0.15]),
                    np.array([ 0.10,  0.35]),
                ][obj_idx]
                placed.append(fallback)
                addr = self._obj_qpos_addrs[obj_idx]
                self.data.qpos[addr:addr+3]   = [fallback[0], fallback[1], _Z_OBJ]
                self.data.qpos[addr+3:addr+7] = [1.0, 0.0, 0.0, 0.0]

        # Forward to settle
        mujoco.mj_forward(self.model, self.data)

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        arm_delta = action[:_N_ARM_JOINTS] * 0.05       # rad
        gripper_cmd = (action[6] + 1.0) / 2.0 * 255.0  # [0, 255]

        # Current arm joint targets
        current_targets = np.array([
            self.data.ctrl[idx] for idx in self._arm_act_ids
        ])
        new_targets = np.clip(
            current_targets + arm_delta,
            self._arm_ctrl_lo,
            self._arm_ctrl_hi,
        )
        for i, idx in enumerate(self._arm_act_ids):
            self.data.ctrl[idx] = new_targets[i]
        self.data.ctrl[self._gripper_act_id] = gripper_cmd

        # Step physics (5 physics steps per env step → 0.01 s)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs     = self._get_obs()
        reward  = self._compute_reward()
        success = self._check_success()
        terminated = success
        truncated  = self._step_count >= self.max_episode_steps
        info = self._get_info()
        info["is_success"] = success

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # Configure a free camera to see the whole cage from the front
                cam = self._viewer.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = [0.0, 0.0, 0.5]
                cam.distance = 3.
                cam.azimuth = 125.0
                cam.elevation = -15.0
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

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Joint positions & velocities
        joint_pos = np.array([self.data.qpos[a] for a in self._arm_qpos_addrs])
        joint_vel = np.array([self.data.qvel[a] for a in self._arm_qvel_addrs])

        # Gripper opening normalised [0,1]
        gripper_open = np.array([self.data.ctrl[self._gripper_act_id] / 255.0])

        # Object poses
        obj_pos   = []
        obj_quat  = []
        for i in range(_N_OBJECTS):
            addr = self._obj_qpos_addrs[i]
            obj_pos.append(self.data.qpos[addr:addr+3])
            obj_quat.append(self.data.qpos[addr+3:addr+7])
        obj_pos  = np.concatenate(obj_pos)
        obj_quat = np.concatenate(obj_quat)

        obs = np.concatenate([
            joint_pos, joint_vel, gripper_open,
            obj_pos, obj_quat,
            self._target_pos,
        ]).astype(np.float32)
        return obs

    def _get_pinch_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._pinch_id].copy()

    def _get_obj_pos(self, idx: int) -> np.ndarray:
        addr = self._obj_qpos_addrs[idx]
        return self.data.qpos[addr:addr+3].copy()

    def _compute_reward(self) -> float:
        pinch = self._get_pinch_pos()

        # Distance from pinch to every object
        obj_positions = [self._get_obj_pos(i) for i in range(_N_OBJECTS)]
        dists_to_objs = [np.linalg.norm(pinch - p) for p in obj_positions]
        nearest_idx   = int(np.argmin(dists_to_objs))
        d_reach       = dists_to_objs[nearest_idx]

        # Grasping heuristic: object is above floor and close to pinch
        nearest_pos = obj_positions[nearest_idx]
        is_lifted  = nearest_pos[2] > _Z_OBJ + 0.01
        is_grasped = d_reach < 0.08 and is_lifted
        grasp_bonus = 0.5 if is_grasped else 0.0

        # Distance from nearest object to target
        d_place = np.linalg.norm(nearest_pos - self._target_pos)

        # Success
        success_bonus = 10.0 if self._check_success() else 0.0

        reward = (
            -d_reach          # pull gripper toward object
            + grasp_bonus     # encourage lifting
            - d_place         # pull object toward target
            + success_bonus
        )
        return float(reward)

    def _check_success(self) -> bool:
        """True if any object is within _SUCCESS_DIST of target."""
        for i in range(_N_OBJECTS):
            pos = self._get_obj_pos(i)
            if np.linalg.norm(pos - self._target_pos) < _SUCCESS_DIST:
                return True
        return False

    def _get_info(self) -> dict:
        pinch = self._get_pinch_pos()
        obj_positions = [self._get_obj_pos(i) for i in range(_N_OBJECTS)]
        return {
            "pinch_pos":  pinch.tolist(),
            "obj_positions": [p.tolist() for p in obj_positions],
            "target_pos": self._target_pos.tolist(),
            "step": self._step_count,
        }
