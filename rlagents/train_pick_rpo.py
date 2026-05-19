"""RPO training for VS050 PickAndPlace environment.

Robust Policy Optimization (Hwangbo et al., 2019): adds adversarial action
perturbations during training to improve sim-to-real transfer.

Uses PPO's GAE advantage estimation + PGD-style adversarial perturbations.

Usage:
    uv run --group cleanrl python -m examples.train_pick_rpo --env-id VS050-PickAndPlace-v0
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
    p.add_argument("--env-id", default="VS050-PickAndPlace-v0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=3_000_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--mini-batch-size", type=int, default=32)
    p.add_argument("--updates-per-step", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.08,
        help="Adversarial perturbation budget per action dimension",
    )
    p.add_argument(
        "--rpo-steps",
        type=int,
        default=3,
        help="Number of PGD steps for adversarial perturbation",
    )
    p.add_argument(
        "--rpo-step-size",
        type=float,
        default=0.02,
        help="Step size for PGD adversarial perturbation",
    )
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--capture-video", action="store_true")
    p.add_argument("--wandb-project", default="vs050-mujoco-rpo")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--load-path", default="")
    return p.parse_args()


def compute_gae(values, rewards, dones, gamma, gae_lambda):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0
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


def evaluate(env, actor_critic, device, n_episodes):
    actor_critic.eval()
    returns = []
    with torch.no_grad():
        for _ in range(n_episodes):
            obs, info = env.reset()
            ep_return = 0.0
            truncated = False
            while not truncated:
                obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(device)
                mean, _, _ = actor_critic(obs_tensor)
                act = torch.tanh(mean).squeeze(0).cpu().numpy()
                act = np.clip(act, -1.0, 1.0)
                obs, rew, terminated, truncated, info = env.step(act)
                ep_return += rew
            returns.append(ep_return)
    actor_critic.train()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env_id, args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    run_name = (
        f"{args.env_id.split('-')[-1]}_rpo_seed{args.seed}_{uuid.uuid4().hex[:6]}"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        entity=args.wandb_entity or None,
        monitor_gym=False,
    )

    actor_critic = ActorCritic(obs_dim, act_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    if args.load_path:
        actor_critic, _ = load_model(
            args.load_path, actor_critic, None, optimizer, device=device
        )

    storage_size = args.batch_size * args.updates_per_step
    obs_s = np.zeros((storage_size, obs_dim), dtype=np.float32)
    act_s = np.zeros((storage_size, act_dim), dtype=np.float32)
    rew_s = np.zeros((storage_size,), dtype=np.float32)
    dones_s = np.zeros((storage_size,), dtype=np.float32)
    logprobs_s = np.zeros((storage_size,), dtype=np.float32)
    values_s = np.zeros((storage_size,), dtype=np.float32)

    last_actor_loss = 0.0
    last_value_loss = 0.0

    obs, _ = env.reset()
    global_step = 0
    episode_count = 0

    for epoch in range(args.total_timesteps // storage_size):
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

            obs_s[step] = obs
            act_s[step] = action_np
            rew_s[step] = reward
            dones_s[step] = float(done)
            logprobs_s[step] = logprob.cpu().numpy()
            values_s[step] = value.cpu().numpy()

            if done:
                obs, _ = env.reset()
                episode_count += 1
            else:
                obs = next_obs

        with torch.no_grad():
            next_obs_t = torch.as_tensor(obs).unsqueeze(0).to(device)
            _, _, next_val = actor_critic(next_obs_t)
            next_val = next_val.squeeze()

        returns, advantages = compute_gae(
            values_s,
            rew_s,
            dones_s,
            args.gamma,
            args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.random.permutation(storage_size)
        for mini_epoch in range(args.updates_per_step):
            for start in range(0, storage_size, args.mini_batch_size):
                end = start + args.mini_batch_size
                batch_idx = indices[start:end]

                batch_obs = torch.as_tensor(obs_s[batch_idx]).to(device)
                batch_act = torch.as_tensor(act_s[batch_idx]).to(device)
                batch_adv = torch.as_tensor(advantages[batch_idx]).to(device)
                batch_logprob = torch.as_tensor(logprobs_s[batch_idx]).to(device)

                mean, log_std, _ = actor_critic(batch_obs)
                std = log_std.exp()

                # Base logprob (for clipping ratio)
                base_dist = torch.distributions.Normal(mean, std)
                base_logprob = base_dist.log_prob(batch_act).sum(dim=-1)

                # Adversarial perturbation (PGD)
                adv_noise = torch.zeros_like(batch_act, requires_grad=True)
                for _ in range(args.rpo_steps):
                    perturbed_act = torch.clamp(batch_act + adv_noise, -1.0, 1.0)
                    perturbed_dist = torch.distributions.Normal(mean, std)
                    perturbed_logprob = perturbed_dist.log_prob(perturbed_act).sum(
                        dim=-1
                    )
                    perturbed_ratio = torch.exp(perturbed_logprob - batch_logprob)
                    perturbed_clip = torch.clamp(
                        perturbed_ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
                    )
                    perturbed_actor_loss = -torch.min(
                        perturbed_ratio * batch_adv, perturbed_clip * batch_adv
                    ).mean()
                    perturbed_actor_loss.backward(retain_graph=True)

                    with torch.no_grad():
                        sign = adv_noise.grad.sign()
                        adv_noise.grad.zero_()
                        adv_noise.data = torch.clamp(
                            adv_noise.data + args.rpo_step_size * sign,
                            -args.epsilon,
                            args.epsilon,
                        )

                perturbed_act = torch.clamp(batch_act + adv_noise.data, -1.0, 1.0)
                perturbed_dist = torch.distributions.Normal(mean, std)
                final_logprob = perturbed_dist.log_prob(perturbed_act).sum(dim=-1)
                ratio = torch.exp(final_logprob - batch_logprob)
                clip_ratio = torch.clamp(
                    ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon
                )
                actor_loss = -torch.min(
                    ratio * batch_adv, clip_ratio * batch_adv
                ).mean()

                vpred = actor_critic(batch_obs)[2].squeeze()
                value_loss = F.mse_loss(
                    vpred, torch.as_tensor(returns[batch_idx]).to(device)
                )
                total_loss = actor_loss + 0.5 * value_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
                optimizer.step()
                last_actor_loss = actor_loss.item()
                last_value_loss = value_loss.item()

        wandb.log(
            {
                "rewards/mean_reward": float(np.mean(rew_s)),
                "losses/actor": last_actor_loss,
                "losses/value": last_value_loss,
                "global_step": global_step,
                "episodes": episode_count,
            },
            step=global_step,
        )

        if global_step % args.eval_every == 0 and global_step > 0:
            eval_ret, eval_std = evaluate(env, actor_critic, device, args.eval_episodes)
            wandb.log(
                {"eval/mean_return": eval_ret, "eval/std_return": eval_std},
                step=global_step,
            )
            print(f"Step {global_step}: eval_return={eval_ret:.2f} ± {eval_std:.2f}")

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
