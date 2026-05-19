# VS050 Training Examples

Single-file CleanRL-style training scripts for all 4 major algorithms across both environments.

## Environments

| ID | Obs dim | Action dim | Description |
|---|---|---|---|
| `VS050-ReachPose-v0` | 21 | 6 | Move end-effector to random goal |
| `VS050-PickAndPlace-v0` | 38 | 7 | Pick object, lift, place at target (with gripper) |

## Algorithms

| Script | Algorithm | Key features |
|---|---|---|
| `train_reach_ddpg.py` | DDPG | Deterministic policy, replay buffer, target networks, Ornstein-Uhlenbeck exploration |
| `train_reach_sac.py` | SAC | Stochastic policy, dual Q-networks, entropy regularization, auto-Tuning |
| `train_reach_ppo.py` | PPO | GAE advantage estimation, clipped surrogate objective, shared actor-critic |
| `train_reach_rpo.py` | RPO | PPO + adversarial action perturbation (PGD) for sim-to-real robustness |

## Quick Start

Install dependencies:
```bash
uv sync --group cleanrl
```

Run a training job:
```bash
# DDPG on Reach
uv run python -m examples.train_reach_ddpg --env-id VS050-ReachPose-v0 --total-timesteps 1000000

# SAC on PickAndPlace with auto entropy
uv run python -m examples.train_pick_sac --env-id VS050-PickAndPlace-v0 --auto-alpha --total-timesteps 3000000

# PPO with video capture (first 3 episodes)
uv run python -m examples.train_reach_ppo --capture-video --total-timesteps 500000

# RPO on PickAndPlace (robust to sim2real)
uv run python -m examples.train_pick_rpo --epsilon 0.08 --rpo-steps 3
```

## Sim-to-Real Transfer Strategy

The RPO scripts are specifically designed for sim-to-real transfer. The workflow:

1. **Train in mujoco** (this repo): Use RPO or SAC with entropy regularization to learn robust policies
2. **Export checkpoint**: `checkpoints/<run_name>_step*.pt`
3. **Load in mjlab**: Use `--load-path checkpoint.pt` to continue training in the parallel environment
4. **Transfer**: The robust policies learned via RPO handle sim-to-real gaps better because:
   - Adversarial perturbations during training act as data augmentation
   - The policy learns to maintain performance under action disturbances
   - Similar to adding motor noise / latency on the real robot

## Checkpoint Loading

All scripts support loading from checkpoints via `--load-path`:
```bash
uv run python -m examples.train_reach_sac --load-path checkpoints/my_run_step500000.pt
```

This loads the actor network and optimizer state, allowing training resumption or fine-tuning on a different environment.

## wandb Logging

All scripts log to wandb by default. Configure with:
```bash
uv run python -m examples.train_reach_ppo \
    --wandb-project my-project \
    --wandb-entity my-team
```

## Hyperparameters

Each algorithm has sensible defaults. Key tunable args:

- `--epsilon` (RPO only): Adversarial perturbation budget (higher = more robust but harder to train)
- `--rpo-steps` (RPO only): Number of PGD steps (3-5 recommended)
- `--noise-std` (DDPG): Exploration noise level (0.1-0.2 reach, 0.2-0.3 pick)
- `--alpha` (SAC): Entropy coefficient (use --auto-alpha to let it learn)
- `--total-timesteps`: Total training steps (2M reach, 3M pick recommended)

## File Structure

```
examples/
  __init__.py
  common.py              # Shared utilities: networks, buffer, save/load
  train_reach_ddpg.py    # DDPG for ReachPose
  train_reach_sac.py     # SAC for ReachPose
  train_reach_ppo.py     # PPO for ReachPose
  train_reach_rpo.py     # RPO for ReachPose
  train_pick_ddpg.py     # DDPG for PickAndPlace
  train_pick_sac.py      # SAC for PickAndPlace
  train_pick_ppo.py      # PPO for PickAndPlace
  train_pick_rpo.py      # RPO for PickAndPlace
```
