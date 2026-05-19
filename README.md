# VS050 MuJoCo (Gymnasium)

![PyPI](https://img.shields.io/pypi/v/vs050-mujoco)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A complete Gym environment and simulation suite for the **DENSO VS050 6-DoF** robotic arm extended with a **Robotiq 2F-85** intelligent gripper. This repository offers a clean Python package and [Gymnasium](https://gymnasium.farama.org/) environments simulating precise manipulation physics via [DeepMind MuJoCo](https://mujoco.org/).

## Installation

This project is fully managed with `uv`. To install it natively or use it in other projects:

```bash
git clone https://github.com/goncamateus/vs050_mujoco.git
cd vs050_mujoco

# Install the package seamlessly into the workspace
uv sync
```

## Quick Start

```python
import gymnasium as gym
import vs050_mujoco  # Registers the environment automatically

env = gym.make("VS050-ReachPose-v0", render_mode="human")
# or
env = gym.make("VS050-PickAndPlace-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # Your agent code here
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Documentation

The project documentation is cleanly split into two modules covering the simulation physics and the reinforcement learning interface:

- 🤖 **[Models Documentation](src/vs050_mujoco/models/README.md)**: Kinematics, joint limits, meshes, and standalone MuJoCo models.
- 🎯 **[Environments Documentation](src/vs050_mujoco/envs/README.md)**: Details on observation spaces, action constraints, and reward functions (includes visual previews).

## Relationship with vs050-mjlab

`vs050-mjlab` ([github.com/goncamateus/vs050_mjlab](https://github.com/goncamateus/vs050_mjlab)) is the RL training companion repo. It depends on `vs050-mujoco` for robot models and provides:

- **ManagerBasedRlEnv** wrappers for batched (100s-1000s env) training with RSL-RL
- **Smooth gaussian reward functions** for stable policy learning
- **PPO training scripts** with W&B logging
- **Entity-based scene composition** via mjlab's API

Use `vs050-mujoco` for: testing, visualization, SB3 training, debugging
Use `vs050-mjlab` for: production RL training with RSL-RL, large-scale parallel simulation

The two repos share the same MuJoCo XML models and robot kinematics, but use different abstraction layers:
- `vs050-mujoco`: `gymnasium.envs.mujoco.MujocoEnv` → flat numpy arrays, single env
- `vs050-mjlab`: `mjlab.envs.ManagerBasedRlEnv` → dict of torch tensors, batched

## License

This project integrates assets from multiple open-source sources, distributed under their respective licenses:

- **DENSO VS050 Robot Assets (Meshes & XML):** Sourced from [DENSORobot/denso_robot_ros2](https://github.com/DENSORobot/denso_robot_ros2) under the **MIT and BSD 3-Clause Licenses**.
- **Robotiq 2F-85 Gripper Assets:** Sourced from Google DeepMind's [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) under the **Apache License 2.0**.
- **Environment & Implementation Code:** Provided under the **MIT License**.

See the `LICENSE` file for the exact terms and complete license texts.
