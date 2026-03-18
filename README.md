# DENSO VS050 – MuJoCo Model

A [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)-style model of the
**DENSO VS050** 6-DoF industrial robot arm, derived from the
[denso_robot_ros2](https://github.com/DENSORobot/denso_robot_ros2) ROS 2 package.

## Robot Specifications

| Parameter | Value |
|-----------|-------|
| Degrees of freedom | 6 revolute |
| Total reach | ~800 mm |
| Payload | 5 kg |
| Mounting | Floor (fixed base) |

### Kinematic Chain

```
world → base_link
  joint_1 (Z) → J1   @ z +181.5 mm
    joint_2 (Y) → J2   @ z +163.5 mm
      joint_3 (Y) → J3   @ z +250.0 mm
        joint_4 (Z) → J4   @ x -10 mm, z +119.5 mm
          joint_5 (Y) → J5   @ z +135.5 mm
            joint_6 (Z) → J6   @ z  +70.0 mm
              attachment_site    @ z  +50.0 mm
```

### Joint Limits

| Joint | Lower (rad) | Upper (rad) | Max vel (rad/s) |
|-------|------------|------------|----------------|
| joint_1 | -2.967 | 2.967 | 3.731 |
| joint_2 | -2.094 | 2.094 | 2.487 |
| joint_3 | -2.182 | 2.705 | 2.715 |
| joint_4 | -4.712 | 4.712 | 3.731 |
| joint_5 | -2.094 | 2.094 | 2.871 |
| joint_6 | -6.283 | 6.283 | 5.969 |

## File Structure

```
vs050_mujoco/
├── assets/
│   ├── base_link.dae   # Base link mesh (Collada)
│   ├── J1.dae          # Link 1 mesh
│   ├── J2.dae          # Link 2 mesh
│   ├── J3.dae          # Link 3 mesh
│   ├── J4.dae          # Link 4 mesh
│   ├── J5.dae          # Link 5 mesh
│   └── J6.dae          # Link 6 mesh (wrist)
├── vs050.xml           # Complete scene (floor + robot + actuators + keyframe)
└── README.md
```

## Usage

### Open in MuJoCo Viewer

```bash
# Python bindings (MuJoCo ≥ 2.3)
python3 -m mujoco.viewer --mjcf vs050_mujoco/vs050.xml
```

Or interactively:

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("vs050_mujoco/vs050.xml")
data  = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, 0)  # load "home" pose
    viewer.sync()
    input("Press Enter to exit...")
```

### Embed in Your Own Scene

Paste the `<worldbody>` contents of `vs050.xml` into your scene, or use
`<include file="path/to/vs050.xml"/>` (note: MuJoCo's `<include>` requires the
included file to contain a complete `<mujoco>` element).

### Programmatic Control (Python)

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("vs050_mujoco/vs050.xml")
data  = mujoco.MjData(model)

# Reset to home keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

# Move joint_1 to 0.5 rad
data.ctrl[0] = 0.5

# Step simulation
for _ in range(1000):
    mujoco.mj_step(model, data)

print("Joint positions:", data.qpos)
```

### Velocity-controlled actuators

The default actuators are **position-controlled** (`<position>`).  
To switch to torque control, replace the `<actuator>` block with:

```xml
<actuator>
  <motor name="act_joint_1" joint="joint_1" gear="1" ctrlrange="-200 200"/>
  <!-- … repeat for joints 2–6 -->
</actuator>
```

## Keyframes

| Name | Description |
|------|-------------|
| `home` | All joints zero except joint_3 = π/2 (≈ 90°), matching `initial_positions.yaml` |

## Meshes

Collada (`.dae`) meshes are used directly as supplied by DENSO WAVE INCORPORATED in the
original ROS 2 package. MuJoCo ≥ 2.3.x supports Collada natively.

If you need OBJ/STL meshes (e.g. for older MuJoCo versions or `obj2mjcf` post-processing),
convert with:

```bash
# requires Blender CLI or trimesh
python3 -c "
import trimesh, pathlib
for p in pathlib.Path('assets').glob('*.dae'):
    trimesh.load(p).export(p.with_suffix('.obj'))
"
```

## License

Meshes: © DENSO WAVE INCORPORATED (original ROS 2 package, see parent repo LICENSE).  
MJCF model: MIT (same as parent repo).
