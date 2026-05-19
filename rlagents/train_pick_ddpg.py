"""DDPG training for VS050 PickAndPlace environment.

CleanRL-style single-file implementation.
Trains a deterministic actor-critic with target networks and replay buffer.

Usage:
    uv run --group cleanrl python -m examples.train_pick_ddpg --env-id VS050-PickAndPlace-v0
"""

from __future__ import annotations

import argparse
import uuid

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from rlagents.common import (
    ActorNetwork,
    QNetwork,
    load_model,
    make_env,
    save_model,
    seed_everything,
)


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
    p.add_argument("--env-id", default="VS050-PickAndPlace-v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=3_000_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learn-every", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--noise-std", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--capture-video", action="store_true")
    p.add_argument("--wandb-project", default="vs050-mujoco-ddpg")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--load-path", default="")
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
                act = actor(obs_tensor).squeeze(0).cpu().numpy()
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
        f"{args.env_id.split('-')[-1]}_ddpg_seed{args.seed}_{uuid.uuid4().hex[:6]}"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        entity=args.wandb_entity or None,
        monitor_gym=False,
    )

    actor = ActorNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    q1 = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q1_optim = torch.optim.Adam(q1.parameters(), lr=args.critic_lr)
    q2 = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q2_optim = torch.optim.Adam(q2.parameters(), lr=args.critic_lr)

    q1_target = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target = QNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    q2_target.load_state_dict(q2.state_dict())
    actor_target = ActorNetwork(obs_dim, act_dim, args.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())

    for net in (q1_target, q2_target, actor_target):
        net.eval()

    buffer = ReplayBuffer(args.buffer_size)

    if args.load_path:
        actor, _ = load_model(args.load_path, actor, None, actor_optim, device=device)

    obs, _ = env.reset()
    total_updates = 0
    last_q1_loss = 0.0
    last_q2_loss = 0.0
    last_actor_loss = 0.0

    for global_step in range(args.total_timesteps):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
            action = actor(obs_tensor).squeeze(0).cpu().numpy()
            action = action + np.random.normal(0, args.noise_std, size=act_dim)
            action = np.clip(action, -1.0, 1.0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs if not done else next_obs

        # Learning step
        if global_step >= args.learning_starts and global_step % args.learn_every == 0:
            obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(args.batch_size)

            with torch.no_grad():
                next_action = actor_target(next_obs_b).clip(-1.0, 1.0)
                q1_next = q1_target(next_obs_b, next_action)
                q2_next = q2_target(next_obs_b, next_action)
                target_q = rew_b + args.gamma * (1.0 - done_b) * torch.min(
                    q1_next, q2_next
                )

            q1_pred = q1(obs_b, act_b)
            q2_pred = q2(obs_b, act_b)
            last_q1_loss = F.mse_loss(q1_pred, target_q).item()
            last_q2_loss = F.mse_loss(q2_pred, target_q).item()
            F.mse_loss(q1_pred, target_q).backward()
            q1_optim.step()
            q1_optim.zero_grad()
            F.mse_loss(q2_pred, target_q).backward()
            q2_optim.step()
            q2_optim.zero_grad()

            act_output = actor(obs_b)
            q1_out = q1(obs_b, act_output)
            last_actor_loss = -q1_out.mean().item()
            (-q1_out.mean()).backward()
            actor_optim.step()
            actor_optim.zero_grad()

            for tgt, src in zip(q1_target.parameters(), q1.parameters()):
                tgt.data.copy_(args.tau * src.data + (1 - args.tau) * tgt.data)
            for tgt, src in zip(q2_target.parameters(), q2.parameters()):
                tgt.data.copy_(args.tau * src.data + (1 - args.tau) * tgt.data)
            for tgt, src in zip(actor_target.parameters(), actor.parameters()):
                tgt.data.copy_(args.tau * src.data + (1 - args.tau) * tgt.data)

            total_updates += 1

        if global_step % 1000 == 0:
            recent_rewards = [
                r for _, _, r, _, _ in list(buffer.buffer)[: min(1000, buffer.size)]
            ]
            wandb.log(
                {
                    "rewards/mean_reward": float(np.mean(recent_rewards))
                    if recent_rewards
                    else 0.0,
                    "losses/q1": last_q1_loss,
                    "losses/q2": last_q2_loss,
                    "losses/actor": last_actor_loss,
                    "global_step": global_step,
                },
                step=global_step,
            )

        if global_step % args.eval_every == 0 and global_step > 0:
            eval_ret, eval_std = evaluate(env, actor, device, args.eval_episodes)
            wandb.log(
                {"eval/mean_return": eval_ret, "eval/std_return": eval_std},
                step=global_step,
            )
            print(f"Step {global_step}: eval_return={eval_ret:.2f} ± {eval_std:.2f}")

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
