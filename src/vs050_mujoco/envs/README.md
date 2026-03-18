# VS050 Environments

This module contains the reinforcement learning environments built using Gymnasium for the DENSO VS050 robot arm.

## `VS050-PickAndPlace-v0`

![Pick and Place Environment](../../../assets/pick_and_place.gif)

A dense-reward environment where the agent must control the VS050 robot arm and a Robotiq 2F-85 gripper to pick up cubic objects and place them on a target marker.

* **Simulation:** The robot operates inside a transparent 1.2m³ glass cage with a wood floor. Three colored cubes (red, blue, yellow) spawn at random intervals within the cage area.
* **Objective:** Move the gripper, securely grab an object, and bring it within `5 cm` of the green target site.

### Action Space

The action space is a `Box(-1.0, 1.0, (7,), float32)`.

| Index | Name | Control Type | Action Range |
|-------|------|--------------|--------------|
| `0–5` | Base & Arm Joints | Delta Position Target | `[-0.05 rad, 0.05 rad]` |
| `6`   | Gripper Opening | Absolute Position | `[-1.0 (open), 1.0 (closed)]` |

### Observation Space

The observation space is a `Box(-inf, inf, (37,), float32)`.

| Indices | Description | Details |
|---------|-------------|---------|
| `0–5` | Joint positions (`qpos`) | The rotational position of the 6 robot joints (rad). |
| `6–11`| Joint velocities (`qvel`) | The rotational velocity of the 6 robot joints (rad/s). |
| `12`  | Gripper state | The normalized `[0, 1]` state of the gripper actuator. |
| `13–21`| Object positions (x3) | XYZ Cartesian coordinates for each of the 3 spawnable objects. |
| `22–33`| Object orientations (x3)| XYZW Quaternions for each of the 3 objects. |
| `34–36`| Target position | XYZ Cartesian coordinates of the target drop zone. |

### Reward

The reward heavily penalizes distance while reinforcing successful manipulation heuristics:

```math
R = -D_{\text{reach}} + B_{\text{grasp}} - D_{\text{place}} + B_{\text{success}}
```

- **Reach Penalty ($D_{\text{reach}}$):** Negative Euclidean distance from the gripper pinch point to the nearest object.
- **Grasp Bonus ($B_{\text{grasp}}$):** Exact `+0.5` points granted continuously when the object is lifted `>1 cm` from the floor while gripped.
- **Place Penalty ($D_{\text{place}}$):** Negative Euclidean distance between the currently grasped object and the final destination.
- **Success Bonus ($B_{\text{success}}$):** A flat `+10.0` points is given if the object arrives inside the target area.

### Termination / Truncation

- **Termination:** Triggers exactly when any object reaches `< 5 cm` from the `target_site`.
- **Truncation:** Standard maximum episode limits bound at `500` timesteps.

---

**Running an Example Headless/Human Agent:**
```bash
uv run python examples/random_agent.py
```
