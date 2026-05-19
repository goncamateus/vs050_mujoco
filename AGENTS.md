# Agent Notes — vs050-mujoco

High-signal facts for working in this repo.

## Toolchain

- **Package manager:** `uv`. Run all Python commands via `uv run ...`. Do not use raw `pip`.
- **Sync deps:** `uv sync` (base), `uv sync --group dev`, `uv sync --group rl`, `uv sync --group wandb`.
- **Build backend:** `hatchling`. Source package at `src/vs050_mujoco`.
- **Formatter / linter:** `ruff`. Format with `uv run ruff format .`. Check/fix with `uv run ruff check --fix .`.

## Tests

- **Pytest suite:** `uv run pytest tests/test_env.py`
- **Standalone smoke test:** `uv run python test_env.py` (tests ReachPose)
- **`tests/test_env.py`:** tests PickAndPlace (imports `vs050_mujoco` to register envs)

## Training Examples

CleanRL-style single-file training scripts in `rlagents/` (8 scripts):

```
rlagents/
  common.py                  # Shared utilities (networks, buffers, save/load)
  train_reach_ddpg.py        # DDPG for VS050-ReachPose-v0
  train_reach_sac.py         # SAC for VS050-ReachPose-v0
  train_reach_ppo.py         # PPO for VS050-ReachPose-v0
  train_reach_rpo.py         # RPO (PGD adversarial) for VS050-ReachPose-v0
  train_pick_ddpg.py         # DDPG for VS050-PickAndPlace-v0
  train_pick_sac.py          # SAC for VS050-PickAndPlace-v0
  train_pick_ppo.py          # PPO for VS050-PickAndPlace-v0
  train_pick_rpo.py          # RPO (PGD adversarial) for VS050-PickAndPlace-v0
```

Install RL dependencies: `uv sync --group cleanrl`

Run: `uv run python -m rlagents.train_reach_sac --env-id VS050-ReachPose-v0`

Sim-to-real strategy: train RPO in mujoco, load checkpoint with `--load-path` in mjlab.

## Architecture

- Importing `vs050_mujoco` auto-registers two Gymnasium environments in `src/vs050_mujoco/__init__.py`.
- `ReachPoseEnv` inherits from `gymnasium.envs.mujoco.MujocoEnv`.
- `PickAndPlaceEnv` inherits from `gymnasium.envs.mujoco.MujocoEnv`.

## Environments

### VS050-ReachPose-v0

- **Observation:** flat `Box` `(21,)` — `[qpos(6), qvel(6), ee_pos(3), goal_pos(3), ee_to_goal(3)]`. **Not** a `GoalEnv` / `Dict`.
- **Action:** `Box(-1, 1, (6,))` delta position targets per arm joint.
- **Reward:** `exp(-d^2/0.02) + exp(-d^2/0.00125) - 0.01*||a-a_prev||^2` (smooth gaussian kernels, matches vs050-mjlab).
- **Termination:** time_out only (no success termination; matches vs050-mjlab). Truncation at `max_episode_steps=500`.
- **Render modes:** `human`, `rgb_array`, `depth_array`.

### VS050-PickAndPlace-v0

- **Observation:** flat `Box` `(38,)` — `[qpos(6), qvel(6), gripper(1), ee_pos(3), ee-to-obj(3), obj-to-target(3), obj_pos(3), obj_quat(4), obj_vel(6), target(3)]`.
- **Action:** `Box(-1, 1, (7,))` — 6 arm joint deltas (`±0.05` rad) + gripper (`[-1, 1]` → `[0, 255]`).
- **Reward:** `reach*(1 + lift*(1 + place)) - 0.01*||a||^2` (multiplicative gaussian shaping, matches vs050-mjlab).
- **Termination:** success (object near target, gripper open) OR out_of_bounds (obj > 0.5m from base). Truncation at `500` steps.
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

## Cross-Reference: vs050-mjlab

`vs050-mjlab` ([github.com/goncamateus/vs050_mjlab](https://github.com/goncamateus/vs050_mjlab)) depends on this package for XML models. See [SYNC_SUMMARY.md](../vs050_mjlab/SYNC_SUMMARY.md) for the full sync status between the two repos.

Key alignment points:
- Both use the same J6 joint range (`[-2pi, 2pi]`)
- Both use smooth gaussian reward functions
- Both use time_out-only termination for reach (mjlab) / success+out_of_bounds for pick_and_place
