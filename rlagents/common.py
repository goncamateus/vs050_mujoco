"""Shared utilities for all training scripts.

Provides:
  - seed_everything(): deterministic RNG seeding
  - make_env(): environment factory with AutoReset + RecordVideo wrappers
  - save_model() / load_model(): checkpoint I/O
  - VideoRecorder: renders episodes and saves MP4 (via gymnasium RecordVideo)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo, RescaleAction, TransformObservation


def seed_everything(seed: int):
    """Set RNG seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Environment factory
# ------------------------------------------------------------------

def make_env(
    env_id: str,
    seed: int,
    video_dir: str | None = None,
    capture_video: bool = False,
    record_episodes: int = 3,
) -> gym.Wrapper:
    """Create an environment with standard wrappers.

    Wrappers applied (in order):
      1. RecordVideo  (optional, first N episodes)
      2. AutoReset
      3. RescaleAction (default ±1 → ±1, identity for our envs)
      4. TransformObservation (identity, placeholder for normalisation)
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if video_dir and capture_video:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda e: e < record_episodes,
        )

    # AutoReset is important for vectorised replay buffers
    env = gym.wrappers.AutoReset(env)

    # Rescale action to [-1, 1] (our envs already use this range)
    env = RescaleAction(env, -1.0, 1.0)

    return env


# ------------------------------------------------------------------
# Model checkpoint I/O
# ------------------------------------------------------------------

def save_model(
    actor: torch.nn.Module,
    critic: torch.nn.Module | None,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer | None,
    save_path: str,
    extras: dict[str, Any] | None = None,
):
    """Save actor + critic networks and optimiser states."""
    checkpoint = {
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict() if critic is not None else None,
        "actor_optim": actor_optim.state_dict(),
        "critic_optim": critic_optim.state_dict() if critic_optim is not None else None,
    }
    if extras:
        checkpoint.update(extras)
    torch.save(checkpoint, save_path)
    print(f"  -> saved checkpoint to {save_path}")


def load_model(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module | None,
    actor_optim: torch.optim.Optimizer | None = None,
    critic_optim: torch.optim.Optimizer | None = None,
    device: str = "cpu",
):
    """Load actor + critic from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    if critic is not None and ckpt.get("critic_state") is not None:
        critic.load_state_dict(ckpt["critic_state"])
        critic.eval()
    if actor_optim is not None and ckpt.get("actor_optim") is not None:
        actor_optim.load_state_dict(ckpt["actor_optim"])
    if critic_optim is not None and ckpt.get("critic_optim") is not None:
        critic_optim.load_state_dict(ckpt["critic_optim"])
    print(f"  -> loaded checkpoint from {path}")
    return actor, critic


# ------------------------------------------------------------------
# Network utilities
# ------------------------------------------------------------------

class ActorCritic(torch.nn.Module):
    """Shared-base actor-critic network (PPO / RPO)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        self.actor_mean = torch.nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = torch.nn.Parameter(torch.zeros(1, act_dim))
        self.critic = torch.nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        shared = self.shared(obs)
        return self.actor_mean(shared), self.actor_log_std, self.critic(shared)


class QNetwork(torch.nn.Module):
    """Q-network for DDPG / SAC (actor-input concatenation)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ActorNetwork(torch.nn.Module):
    """Deterministic actor for DDPG (tanh squashed)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, act_dim),
            torch.nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
