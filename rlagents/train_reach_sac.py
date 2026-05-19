"""SAC training for VS050 ReachPose environment.

CleanRL-style single-file implementation.
Trains a stochastic actor with dual Q-networks and entropy scheduling.

Usage:
    uv run --group cleanrl python -m examples.train_reach_sac --env-id VS050-ReachPose-v0
"""

from __future__ import annotations

import argparse
import uuid
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from rlagents.common import (
    QNetwork,
    ActorNetwork,
    load_model,
    make_env,
    save_model,
    seed_everything,
)


class StochasticActor(nn.Module):
    """Gaussian policy with learnable log-std (tanh squashed)."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_sample = dist.rsample()
        action = torch.tanh(raw_sample)
        # Adjust log_prob for tanh squashing
        log_prob = dist.log_prob(raw_sample)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        return action, log_prob


class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.buffer = np.zeros((capacity, 5), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.capacity = capacity

    def push(self, obs, act, rew, next_obs, done):
        self.buffer[self.pos] = (obs, act, rew, next_obs, done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = self.buffer[idx]
        return (
            torch.as_tensor(batch[:, 0], dtype=torch.float32),
            torch.as_tensor(batch[:, 1], dtype=torch.float32),
            torch.as_tensor(batch[:, 2], dtype=torch.float32),
            torch.as_tensor(batch[:, 3], dtype=torch.float32),
            torch.as_tensor(batch[:, 4], dtype=torch.float32),
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="VS050-ReachPose-v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha", type=float, default=0.2)  # entropy coefficient
    p.add_argument("--auto-alpha", action="store_true")
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--capture-video", action="store_true")
    p.add_argument("--wandb-project", default="vs050-mujoco-sac")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--load-path", default="")
    p.add_argument("--steps-per-episode", type=int, default=500)
    return p.parse_args()


def evaluate(env, actor, device, n_episodes):
    actor.eval()
    returns = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset()
            ep_return = 0.0
            truncated = False
            while not truncated:
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
                mean, _ = actor(obs_tensor)
                act = torch.tanh(mean).squeeze(0).cpu().numpy()
                act = np.clip(act, -1.0, 1.0)
                obs, rew, terminated, truncated, info = env.step(act)
                ep_return += rew
            returns.append(ep_return)
    actor.train()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env_id, args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    run_name = (
        f"{args.env_id.split('-')[-1]}_sac_seed{args.seed}_{uuid.uuid4().hex[:6]}"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        entity=args.wandb_entity or None,
        monitor_gym=False,
    )

    # Networks
    actor = StochasticActor(obs_dim, act_dim, args.hidden_dim).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    q1 = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q1_optim = torch.optim.Adam(q1.parameters(), lr=args.critic_lr)
    q2 = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q2_optim = torch.optim.Adam(q2.parameters(), lr=args.critic_lr)

    q1_target = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q2_target.load_state_dict(q2.state_dict())
    for net in (q1_target, q2_target):
        net.eval()

    buffer = ReplayBuffer(args.buffer_size)

    # Target entropy for automatic alpha
    if args.auto_alpha:
        target_entropy = -0.5 * np.log(1 / act_dim)
        log_alpha = torch.tensor(np.log(args.alpha), requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.critic_lr)
        alpha = args.alpha
    else:
        alpha = args.alpha
        target_entropy = 0.0
        log_alpha = None
        alpha_optim = None

    # Load checkpoint
    if args.load_path:
        actor, _ = load_model(args.load_path, actor, None, actor_optim, device=device)

    obs, _ = env.reset()
    last_q1_loss = 0.0
    last_q2_loss = 0.0
    last_actor_loss = 0.0

    for global_step in range(args.total_timesteps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
            action, _ = actor.get_action(obs_tensor)
            action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs if not done else next_obs

        # Learning step
        if global_step >= args.learning_starts:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(args.batch_size)

            with torch.no_grad():
                next_action, next_log_prob = actor.get_action(next_obs_b)
                q1_next = q1_target(next_obs_b, next_action)
                q2_next = q2_target(next_obs_b, next_action)
                min_q = torch.min(q1_next, q2_next)
                target_q = rew_b + args.gamma * (1.0 - done_b) * (
                    min_q - args.alpha * next_log_prob
                )

            q1_pred = q1(obs_b, act_b)
            q2_pred = q2(obs_b, act_b)
            q1_loss = F.mse_loss(q1_pred, target_q).item()
            q2_loss = F.mse_loss(q2_pred, target_q).item()
            F.mse_loss(q1_pred, target_q).backward()
            q1_optim.step()
            q1_optim.zero_grad()
            F.mse_loss(q2_pred, target_q).backward()
            q2_optim.step()
            q2_optim.zero_grad()

            # Actor update
            act_out, actor_log_prob = actor.get_action(obs_b)
            q1_out = q1(obs_b, act_out)
            q2_out = q2(obs_b, act_out)
            min_q_out = torch.min(q1_out, q2_out)
            actor_loss = (args.alpha * actor_log_prob - min_q_out).mean().item()
            (-min_q_out).backward()
            actor_optim.step()
            actor_optim.zero_grad()

            # Target soft update
            for tgt, src in zip(q1_target.parameters(), q1.parameters()):
                tgt.data.copy_(args.tau * src.data + (1 - args.tau) * tgt.data)
            for tgt, src in zip(q2_target.parameters(), q2.parameters()):
                tgt.data.copy_(args.tau * src.data + (1 - args.tau) * tgt.data)

            # Update alpha
            if args.auto_alpha and log_alpha is not None:
                with torch.no_grad():
                    _, current_log_prob = actor.get_action(obs_b)
                    alpha_loss = -(
                        log_alpha * (current_log_prob + target_entropy)
                    ).mean()
                alpha_loss.backward()
                alpha_optim.step()
                alpha = log_alpha.exp().item()

            last_q1_loss = q1_loss
            last_actor_loss = actor_loss

        # Logging
        if global_step % 1000 == 0:
            log_dict = {
                "rewards/mean_reward": np.mean(
                    [
                        r
                        for _, _, r, _, _ in list(buffer.buffer)[
                            : min(1000, buffer.size)
                        ]
                    ]
                ),
                "losses/q1": last_q1_loss,
                "losses/q2": last_q2_loss,
                "losses/actor": last_actor_loss,
                "global_step": global_step,
            }
            if args.auto_alpha and log_alpha is not None:
                log_dict["alpha"] = alpha
            wandb.log(log_dict, step=global_step)

        # Evaluation
        if global_step % args.eval_every == 0 and global_step > 0:
            eval_ret, eval_std = evaluate(env, actor, device, args.eval_episodes)
            wandb.log(
                {"eval/mean_return": eval_ret, "eval/std_return": eval_std},
                step=global_step,
            )
            print(f"Step {global_step}: eval_return={eval_ret:.2f} ± {eval_std:.2f}")

        # Save
        if global_step % args.save_every == 0 and global_step > 0:
            save_model(
                actor,
                None,
                actor_optim,
                None,
                f"checkpoints/{run_name}_step{global_step}.pt",
            )

    env.close()
    wandb.finish()
    print(f"Training complete. Checkpoints in checkpoints/{run_name}_*.pt")


if __name__ == "__main__":
    main()
