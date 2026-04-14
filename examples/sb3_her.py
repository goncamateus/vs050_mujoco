"""
Train and run a Hindsight Experience Replay (HER) agent on VS050 reach-pose.

Usage:
    uv run python examples/sb3_her.py --train --total-timesteps 200000
    uv run python examples/sb3_her.py --play --model-path runs/her_vs050/final_model

Notes:
    - Requires stable-baselines3 and torch installed in the environment.
    - HER works best with sparse rewards and goal-based environments like VS050-ReachPose-v0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import vs050_mujoco  # noqa: F401  # Registers VS050-ReachPose-v0
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
import gymnasium as gym


ENV_ID = "VS050-ReachPose-v0"


def make_env(render_mode: str | None = None) -> gym.Env:
    """Create a monitored environment instance."""
    env = gym.make(ENV_ID, render_mode=render_mode)
    return Monitor(env)


def make_vec_env(
    n_envs: int,
    vec_env_cls,
    render_mode: str | None = None,
    seed: int | None = None,
    start_index: int = 0,
    video_folder: str | None = None,
    video_episodes: int = 100,
):
    """Create a vectorized environment with video recording for the first env."""

    def make_env_fn(rank):
        def _init():
            # Determine the render mode for this environment
            # All environments must use the same render_mode to avoid AssertionError in SubprocVecEnv
            env_render_mode = render_mode
            if video_folder is not None:
                # When recording video, we need rgb_array mode for the RecordVideo wrapper
                env_render_mode = "rgb_array"

            env = gym.make(ENV_ID, render_mode=env_render_mode)

            # Set seed for reproducibility
            if seed is not None:
                env.reset(seed=seed + rank)

            # Wrap first environment with video recording
            if rank == 0 and video_folder is not None:
                env = gym.wrappers.RecordVideo(
                    env,
                    video_folder,
                    episode_trigger=lambda x: x % video_episodes == 0,
                    name_prefix=f"vs050-her-rank{rank}",
                )

            return Monitor(env)

        return _init

    vec_env = vec_env_cls([make_env_fn(i + start_index) for i in range(n_envs)])
    # Apply VecMonitor only once at the vectorized level
    return VecMonitor(vec_env)


def _maybe_setup_wandb(args: argparse.Namespace, out_dir: Path):
    if not args.wandb:
        return None, None

    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases is not installed. Install with: uv sync --group wandb"
        ) from exc

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "algorithm": "SAC+HER",
            "env_id": ENV_ID,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
        },
        tags=tags,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    wandb_cb = WandbCallback(
        model_save_path=str(out_dir / "wandb_models"),
        model_save_freq=20_000,
        gradient_save_freq=0,
        verbose=2,
    )
    return run, wandb_cb


def train(args: argparse.Namespace) -> Path:
    total_timesteps = args.total_timesteps
    out_dir = args.out_dir
    seed = args.seed

    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose vectorized environment class
    vec_env_cls = DummyVecEnv if args.vec_env_type == "dummy" else SubprocVecEnv

    # Create vectorized environments
    train_env = make_vec_env(
        n_envs=args.num_envs,
        vec_env_cls=vec_env_cls,
        render_mode=None,
        seed=seed,
        video_folder=str(out_dir / "videos") if args.train else None,
        video_episodes=50,
    )
    eval_env = make_vec_env(
        n_envs=args.num_eval_envs, vec_env_cls=vec_env_cls, render_mode=None, seed=seed
    )
    run, wandb_cb = _maybe_setup_wandb(args, out_dir)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="her_vs050",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # HER-specific parameters
    n_sampled_goal = 4  # Reduced for better performance with vec_env
    goal_selection_strategy = "future"  # 'future' often works better than 'episode'

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        verbose=1,
        seed=seed,
        tensorboard_log=str(out_dir / "tb") if args.tensorboard else None,
        learning_rate=3e-4,
        buffer_size=1_000_000,  # HER typically needs larger buffer
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        # Configure HER replay buffer
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=goal_selection_strategy,
            # For HER with GoalEnv, we need to specify the observation keys
            # The environment uses Dict observation space with 'observation', 'achieved_goal', 'desired_goal'
            # HER will automatically handle the goal relabeling
        ),
    )

    callbacks = [checkpoint_cb, eval_cb]
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    model_path = out_dir / "final_model"
    model.save(str(model_path))

    train_env.close()
    eval_env.close()
    if run is not None:
        run.summary["saved_model"] = str(model_path.with_suffix(".zip"))
        run.summary["train_args"] = json.dumps(vars(args), default=str)
        run.finish()
    return model_path


def play(model_path: Path, episodes: int, max_steps: int, deterministic: bool) -> None:
    env = make_env(render_mode="human")
    model = SAC.load(str(model_path), env=env)

    success_count = 0
    for ep in range(episodes):
        obs, _info = env.reset()
        ep_reward = 0.0
        success = False

        for step in range(max_steps):
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)

            # Check if we reached the goal (success)
            # For vectorized envs, info might be a list, but we're using single env here
            if isinstance(info, dict) and info.get("is_success", False):
                success = True
            elif (
                isinstance(info, list)
                and len(info) > 0
                and info[0].get("is_success", False)
            ):
                success = True

            if terminated or truncated:
                break

        if success:
            success_count += 1

        print(
            f"Episode {ep + 1}/{episodes} reward: {ep_reward:.3f} "
            f"{'SUCCESS' if success else 'failed'} "
            f"(steps: {step + 1})"
        )

    success_rate = success_count / episodes * 100
    print(f"\nSuccess rate: {success_rate:.1f}% ({success_count}/{episodes})")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HER example for VS050 reach-pose")
    parser.add_argument("--train", action="store_true", help="Train a new HER policy")
    parser.add_argument(
        "--play", action="store_true", help="Run a trained policy with rendering"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/her_vs050"),
        help="Output directory for checkpoints, logs, and model",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("runs/her_vs050/final_model"),
        help="Path to a saved SAC model zip path stem",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--episodes", type=int, default=3, help="Play episodes")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max env steps per play episode",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy at inference time",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging during training",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vs050-her",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user). Omit to use your default account",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="vs050,her,sb3",
        help="Comma-separated W&B tags",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable tensorboard logging",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments for training (default: 1)",
    )
    parser.add_argument(
        "--num-eval-envs",
        type=int,
        default=1,
        help="Number of parallel environments for evaluation (default: 1)",
    )
    parser.add_argument(
        "--vec-env-type",
        type=str,
        choices=["dummy", "subproc"],
        default="dummy",
        help="Type of vectorized environment (dummy or subproc)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train and not args.play:
        raise SystemExit("Select at least one mode: --train and/or --play")

    model_path = args.model_path
    if args.train:
        model_path = train(args)
        print(f"Saved final model to: {model_path}.zip")

    if args.play:
        if not model_path.with_suffix(".zip").exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}.zip. Run with --train first or pass --model-path."
            )
        play(
            model_path=model_path,
            episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
        )


if __name__ == "__main__":
    main()
