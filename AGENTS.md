# Agent Notes — vs050-mujoco

High-signal facts for working in this repo.

## Toolchain

- **Package manager:** `uv`. Run all Python commands via `uv run ...`. Do not use raw `pip`.
- **Sync deps:** `uv sync` (base), `uv sync --group dev`, `uv sync --group rl`, `uv sync --group wandb`.
- **Build backend:** `hatchling`. Source package at `src/vs050_mujoco`.
- **Formatter / linter:** `ruff`. Format with `uv run ruff format .`. Check/fix with `uv run ruff check --fix .`.

## Tests

- **Pytest suite:** `uv run pytest tests/test_env.py`
- **Standalone smoke test:** `uv run python test_env.py` (root file, tests ReachPose)

## Architecture

- Importing `vs050_mujoco` auto-registers two Gymnasium environments in `src/vs050_mujoco/__init__.py`.
- `ReachPoseEnv` inherits from `gymnasium.envs.mujoco.MujocoEnv`.
- `PickAndPlaceEnv` inherits from `gymnasium.envs.mujoco.MujocoEnv`.

## Environments

### VS050-ReachPose-v0

- **Observation:** flat `Box` `(21,)` — `[qpos(6), qvel(6), ee_pos(3), goal_pos(3), ee_to_goal(3)]`. **Not** a `GoalEnv` / `Dict`.
- **Action:** `Box(-1, 1, (6,))` delta position targets per arm joint.
- **Reward:** `-distance - ctrl_cost + success_bonus` where success is `< 1 cm`.
- **Termination:** on success. Truncation at `max_episode_steps=500`.
- **Render modes:** `human`, `rgb_array`, `depth_array`.

### VS050-PickAndPlace-v0

- **Observation:** flat `Box` `(23,)` — `[qpos(6), qvel(6), gripper(1), obj_pos(3), obj_quat(4), target(3)]`.
- **Action:** `Box(-1, 1, (7,))` — 6 arm joint deltas (`±0.05` rad) + gripper (`[-1, 1]` → `[0, 255]`).
- **Reward:** dense: `-d_reach + grasp_bonus(1.0) - d_place + success_bonus(100.0)`.
- **Success:** object within `1 mm` of target. Truncation at `500` steps.
- **Render modes:** `human`, `rgb_array`, `depth_array`.

## Known Stale References

- `examples/` directory was deleted in commit `6e1c8c2`. READMEs previously referenced `examples/random_agent.py`, `examples/sb3_sac.py`, and `examples/sb3_her.py` — these no longer exist.

## CI / Release

- `.github/workflows/python-publish.yml` triggers on GitHub Release `published`.
- Builds with `python -m build`, publishes to PyPI via trusted publishing using environment `pypi`.

## Coding Style

- **Modular over monolithic.** Prefer many small, single-purpose methods over long blocks. Example: `__init__` should delegate to `_init_cache_ids()`, `_init_spaces()`, etc. Observation building → `_get_joint_obs()`, `_get_gripper_obs()`, etc. Keep each method under ~15 lines when possible.

## Runtime Artifacts

- MuJoCo writes `MUJOCO_LOG.TXT` to cwd.
- Training runs output to `runs/` and `wandb/`.
- All three are gitignored.
