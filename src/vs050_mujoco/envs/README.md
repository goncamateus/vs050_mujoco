# VS050 Environments

This module contains the reinforcement learning environments built using Gymnasium for the DENSO VS050 robot arm.

## `VS050-ReachPose-v0`

![ReachPose Environment](../../../assets/reach_pose.gif)

ReachPose is a goal-oriented environment designed for Hindsight Experience Replay (HER). The 6-DoF VS050 arm must move its end-effector to a randomly sampled 3D target position inside its reachable workspace.

* **Simulation:** The bare VS050 arm on a floor. A translucent red sphere marks the goal position, sampled from a conservative workspace envelope.
* **Objective:** Drive the end-effector (the `attachment_site`) within `4 cm` of the desired target.
### Action Space

The action space is a `Box(-1.0, 1.0, (6,), float32)`.

| Index | Name | Control Type | Max delta per step |
|-------|------|--------------|--------------------|
| `0–5` | Base & Arm Joints | Delta Position Target | `[0.08, 0.08, 0.08, 0.10, 0.10, 0.15]` rad |

### Observation Space

The observation space is a flat `Box`:

| Indices | Description | Details |
|---------|-------------|---------|
| `0–5`  | Joint positions (`qpos`) | 6 arm joint positions (rad). |
| `6–11` | Joint velocities (`qvel`) | 6 arm joint velocities (rad/s). |
| `12–14`| End-effector XYZ | Position of `attachment_site` in world frame. |
| `15–17`| Goal position XYZ | Position of `target_marker` geom in world frame. |
| `18–20`| Relative position | `goal_pos - ee_pos`. |

### Reward

Potential-based shaping plus control cost and sparse success bonus:

- **Dense:** `-distance - ctrl_cost`
- **Success bonus:** `+100.0` when end-effector is within `1 cm` of goal.

### Termination / Truncation

- **Termination:** When end-effector is within `1 cm` of goal.
- **Truncation:** Standard maximum episode limit of `500` timesteps.

---

## `VS050-PickAndPlace-v0`

![Pick and Place Environment](../../../assets/pick_and_place.gif)

A dense-reward environment where the agent must control the VS050 robot arm and a Robotiq 2F-85 gripper to pick up cubic objects and place them on a target marker. Inherits from `gymnasium.envs.mujoco.MujocoEnv`.

* **Simulation:** The robot operates inside a transparent 1.2m³ glass cage with a wood floor. One red cube spawns at a random position within the cage area.
* **Objective:** Move the gripper, securely grab the object, and bring it within `1 mm` of the green target site.

### Action Space

The action space is a `Box(-1.0, 1.0, (7,), float32)`.

| Index | Name | Control Type | Action Range |
|-------|------|--------------|--------------|
| `0–5` | Base & Arm Joints | Delta Position Target | `[-0.05 rad, 0.05 rad]` |
| `6`   | Gripper Opening | Absolute Position | `[-1.0 (open), 1.0 (closed)]` |

### Observation Space

The observation space is a `Box(-inf, inf, (23,), float32)`.

| Indices | Description | Details |
|---------|-------------|---------|
| `0–5` | Joint positions (`qpos`) | The rotational position of the 6 robot joints (rad). |
| `6–11`| Joint velocities (`qvel`) | The rotational velocity of the 6 robot joints (rad/s). |
| `12`  | Gripper state | The normalized `[0, 1]` state of the gripper actuator. |
| `13–15`| Object position | XYZ Cartesian coordinates of the spawnable object. |
| `16–19`| Object orientation | XYZW Quaternion of the object. |
| `20–22`| Target position | XYZ Cartesian coordinates of the target drop zone. |

### Reward

The reward heavily penalizes distance while reinforcing successful manipulation heuristics:

```math
R = -D_{\text{reach}} + B_{\text{grasp}} - D_{\text{place}} + B_{\text{success}}
```

- **Reach Penalty ($D_{\text{reach}}$):** Negative Euclidean distance from the gripper pinch point to the object.
- **Grasp Bonus ($B_{\text{grasp}}$):** One-shot `+1.0` granted the first time the object is lifted `>1 cm` from the floor while the gripper is within `5 cm` of it.
- **Place Penalty ($D_{\text{place}}$):** Negative Euclidean distance between the object and the final destination.
- **Success Bonus ($B_{\text{success}}$):** A flat `+100.0` points is given if the object arrives within `1 mm` of the target area.

### Termination / Truncation

- **Termination:** Triggers exactly when the object is within `1 mm` of the `target_site`.
- **Truncation:** Standard maximum episode limits bound at `500` timesteps.

---
