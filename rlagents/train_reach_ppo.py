"""PPO training for VS050 ReachPose environment.

CleanRL-style single-file implementation with GAE advantage estimation.

Usage:
    uv run --group cleanrl python -m examples.train_reach_ppo --env-id VS050-ReachPose-v0
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
    ActorCritic,
    load_model,
    make_env,
    save_model,
    seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="VS050-ReachPose-v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mini-batch-size", type=int, default=32)
    p.add_argument("--updates-per-step", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--capture-video", action="store_true")
    p.add_argument("--wandb-project", default="vs050-mujoco-ppo")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--load-path", default="")
    return p.parse_args()


def compute_gae(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    """Compute GAE advantage estimates."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae
        )
    returns = advantages + values
    return returns, advantages


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
                mean, _, _ = actor(obs_tensor)
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
        f"{args.env_id.split('-')[-1]}_ppo_seed{args.seed}_{uuid.uuid4().hex[:6]}"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        entity=args.wandb_entity or None,
        monitor_gym=False,
    )

    # Network
    actor_critic = ActorCritic(obs_dim, act_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # Load checkpoint
    if args.load_path:
        actor_critic, _ = load_model(
            args.load_path, actor_critic, None, optimizer, device=device
        )

    # Storage
    storage_size = args.batch_size * args.updates_per_step
    obs_storage = np.zeros((storage_size, obs_dim), dtype=np.float32)
    act_storage = np.zeros((storage_size, act_dim), dtype=np.float32)
    rew_storage = np.zeros((storage_size,), dtype=np.float32)
    dones_storage = np.zeros((storage_size,), dtype=np.float32)
    logprobs_storage = np.zeros((storage_size,), dtype=np.float32)
    values_storage = np.zeros((storage_size,), dtype=np.float32)

    # Training
    obs, _ = env.reset()
    global_step = 0
    episode_count = 0

    for epoch in range(args.total_timesteps // storage_size):
        # Rollout
        for step in range(storage_size):
            global_step += 1
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
                mean, log_std, value = actor_critic(obs_tensor)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(dim=-1)
                value = value.squeeze()

            action_np = torch.tanh(action).squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            obs_storage[step] = obs
            act_storage[step] = action_np
            rew_storage[step] = reward
            dones_storage[step] = float(done)
            logprobs_storage[step] = logprob.cpu().numpy()
            values_storage[step] = value.cpu().numpy()

            if done:
                obs, _ = env.reset()
                episode_count += 1
            else:
                obs = next_obs

        # Compute returns and advantages
        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
            _, _, next_value = actor_critic(next_obs_tensor)
            next_value = next_value.squeeze()

        returns, advantages = compute_gae(
            values_storage,
            rew_storage,
            dones_storage,
            args.gamma,
            args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates
        indices = np.random.permutation(storage_size)
        for mini_epoch in range(args.updates_per_step):
            for start in range(0, storage_size, args.mini_batch_size):
                end = start + args.mini_batch_size
                batch_idx = indices[start:end]

                batch_obs = torch.as_tensor(obs_storage[batch_idx]).to(device)
                batch_act = torch.as_tensor(act_storage[batch_idx]).to(device)
                batch_ret = torch.as_tensor(returns[batch_idx]).to(device)
                batch_adv = torch.as_tensor(advantages[batch_idx]).to(device)
                batch_logprob = torch.as_tensor(logprobs_storage[batch_idx]).to(device)

                mean, log_std, value = actor_critic(batch_obs)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                new_logprob = dist.log_prob(batch_act).sum(dim=-1)

                ratio = torch.exp(new_logprob - batch_logprob)
                clipped_ratio = torch.clamp(
                    ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
                )
                actor_loss = -torch.min(
                    ratio * batch_adv, clipped_ratio * batch_adv
                ).mean()

                vpred_new = value.squeeze()
                value_loss = F.mse_loss(vpred_new, batch_ret)
                total_loss = actor_loss + 0.5 * value_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        wandb.log(
            {
                "rewards/mean_reward": float(np.mean(rew_storage)),
                "losses/actor": float(actor_loss),
                "losses/value": float(value_loss),
                "eval/mean_return": float(np.mean(rew_storage)),
                "global_step": global_step,
                "episodes": episode_count,
            },
            step=global_step,
        )

        # Evaluation
        if global_step % args.eval_every == 0 and global_step > 0:
            eval_ret, eval_std = evaluate(env, actor_critic, device, args.eval_episodes)
            wandb.log(
                {"eval/mean_return": eval_ret, "eval/std_return": eval_std},
                step=global_step,
            )
            print(f"Step {global_step}: eval_return={eval_ret:.2f} ± {eval_std:.2f}")

        # Save
        if global_step % args.save_every == 0 and global_step > 0:
            save_model(
                actor_critic,
                actor_critic,
                optimizer,
                optimizer,
                f"checkpoints/{run_name}_step{global_step}.pt",
            )

    env.close()
    wandb.finish()
    print(f"Training complete. Checkpoints in checkpoints/{run_name}_*.pt")


if __name__ == "__main__":
    main()
