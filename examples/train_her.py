"""
Goal-conditioned RL training on VS050-ReachPose with custom Hindsight Experience Replay (HER) + SAC.

This script implements HER manually so it works regardless of SB3's HER wrapper availability.
The environment (ReachPoseEnv) follows gymnasium.GoalEnv with Dict observations.
We use a custom HER replay buffer that relabels achieved goals as desired goals for past transitions.

Usage:
    # Train from scratch
    uv run python examples/train_her.py --train --total-timesteps 100000

    # Evaluate a trained model with rendering
    uv run python examples/train_her.py --play --model-path runs/her_vs050/final_model
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import vs050_mujoco  # noqa: F401  # Registers VS050-ReachPose-v0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  Networks
# ============================================================

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        self.log_std_min, self.log_std_max = -5, 2
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.backbone.apply(init_weights)

    def forward(self, obs, return_dist=False):
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = torch.tanh(self.log_std(h))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z).sum(-1)
        log_prob = log_prob - torch.log1p(action.pow(2) + 1e-6).sum(-1)
        mean = torch.tanh(mu)
        return (mean, log_prob) if return_dist else (action, log_prob)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + goal_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q1.apply(init_weights)
        self.q2.apply(init_weights)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


# ============================================================
#  HER Replay Buffer
# ============================================================

class HERReplayBuffer:
    """
    Standard circular replay buffer with HER relabeling at sample time.
    Each stored transition: (observation, achieved_goal, desired_goal, action, reward, next_obs, next_achieved_goal, done)
    When sampling, future goals from the same episode are used to relabel desired_goal.
    """

    def __init__(
        self,
        capacity: int,
        n_sampled_goal: int = 4,
    ):
        self.capacity = capacity
        self.n_sampled_goal = n_sampled_goal
        self.buf: deque[dict] = deque(maxlen=capacity)
        self._episode_starts: list[int] = []
        self._episode_ends: list[int] = []  # exclusive end indices in buf

    def push(self, obs: dict, achieved: np.ndarray, desired: np.ndarray,
             action: np.ndarray, reward: float, next_obs: dict,
             next_achieved: np.ndarray, done: bool):
        self.buf.append({
            "obs": obs.copy(),
            "achieved": achieved.copy(),
            "desired": desired.copy(),
            "action": action.copy(),
            "reward": reward,
            "next_obs": next_obs.copy(),
            "next_achieved": next_achieved.copy(),
            "done": done,
        })
        if done:
            if len(self._episode_starts) == 0:
                self._episode_starts.append(0)
            else:
                self._episode_starts.append(len(self.buf) - 1)
            self._episode_ends.append(len(self.buf))

    def _future_goal(self, idx: int) -> np.ndarray:
        """Sample a future achieved goal from the same episode as achieved_goal."""
        # Find which episode this transition belongs to
        ep_start = 0
        ep_end = len(self.buf)
        for e, end in enumerate(self._episode_ends):
            if e < len(self._episode_starts):
                start = self._episode_starts[e]
                if start <= idx < end:
                    ep_start = start
                    ep_end = end
                    break

        goals = []
        for _ in range(self.n_sampled_goal):
            low = idx - ep_start + 1
            high = ep_end - ep_start
            if low < high:
                t = np.random.randint(low, high) + ep_start
                goals.append(self.buf[t]["next_achieved"])
            else:
                # No future steps — use this step's achieved goal
                goals.append(self.buf[idx]["next_achieved"])
        return goals

    def sample(self, batch_size: int, compute_reward_fn):
        """Return batch of (obs, action, reward, next_obs, done) with possible HER relabeling."""
        indices = np.random.randint(0, len(self.buf), size=batch_size)

        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []

        for idx in indices:
            t = self.buf[idx]
            if np.random.random() < 0.5 and len(self._episode_ends) > 0:
                # HER relabeling
                future_goals = self._future_goal(idx)
                desired = future_goals[np.random.randint(len(future_goals))]
            else:
                desired = t["desired"]

            achieved = t["next_achieved"]
            reward = compute_reward_fn(achieved.copy(), desired.copy(), {})
            done = t["done"]

            obs = np.concatenate([t["obs"]["observation"], t["desired"]])
            next_obs = np.concatenate([t["next_obs"]["observation"], desired])

            obs_list.append(obs)
            action_list.append(t["action"])
            reward_list.append(float(reward))
            next_obs_list.append(next_obs)
            done_list.append(float(done))

        return (
            torch.FloatTensor(np.array(obs_list)).to(DEVICE),
            torch.FloatTensor(np.array(action_list)).to(DEVICE),
            torch.FloatTensor(np.array(reward_list)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_obs_list)).to(DEVICE),
            torch.FloatTensor(np.array(done_list)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buf)

    def clear(self):
        self.buf.clear()
        self._episode_starts.clear()
        self._episode_ends.clear()


# ============================================================
#  SAC Agent with HER
# ============================================================

class HER_SAC:
    def __init__(self, obs_dim: int, goal_dim: int, action_dim: int,
                 action_low: float = -1.0, action_high: float = 1.0,
                 lr: float = 1e-3, gamma: float = 0.99, tau: float = 0.05,
                 ent_coef: str = "auto", target_entropy: float = -3.0,
                 buffer_size: int = 1_000_000, n_sampled_goal: int = 4,
                 learning_starts: int = 1_000, batch_size: int = 256):

        self.gamma = gamma
        self.tau = tau
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.action_low = action_low
        self.action_high = action_high

        self.actor = Actor(obs_dim, goal_dim, action_dim).to(DEVICE)
        self.critic = Critic(obs_dim, goal_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(obs_dim, goal_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # Temperature (auto entropy)
        if ent_coef == "auto":
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.target_entropy = 0
            self.log_alpha = torch.log(torch.tensor(ent_coef, device=DEVICE))

        self.replay_buffer = HERReplayBuffer(
            capacity=buffer_size,
            n_sampled_goal=n_sampled_goal,
        )

    def select_action(self, obs_array: dict, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(
                np.concatenate([obs_array["observation"], obs_array["desired_goal"]])
            ).unsqueeze(0).to(DEVICE)
            action, _ = self.actor(obs_tensor)
            if deterministic:
                pass  # already using mean through tanh
            else:
                action, _ = self.actor(obs_tensor)
            return action.cpu().numpy()[0].clip(self.action_low, self.action_high)

    def train_step(self, compute_reward_fn):
        if len(self.replay_buffer) < self.learning_starts:
            return

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            self.batch_size, compute_reward_fn
        )

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs, return_dist=True)
            q1_t, q2_t = self.critic_target(next_obs, next_action)
            min_q = torch.min(q1_t, q2_t)
            alpha = self.log_alpha.exp()
            target_q = rewards + (1 - dones) * self.gamma * (min_q - alpha * next_log_prob.unsqueeze(1))

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update ---
        pi, log_prob = self.actor(obs, return_dist=True)
        q1_pi, q2_pi = self.critic(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_prob - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -- Alpha update --
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: Path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha,
            },
            str(path) + ".pt",
        )

    def load(self, path: Path):
        ckpt = torch.load(str(path) + ".pt", map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.log_alpha = ckpt["log_alpha"]


# ============================================================
#  Train / Play
# ============================================================

def train(args: argparse.Namespace) -> Path:
    env = gym.make(ENV_ID, render_mode=None)
    eval_env = gym.make(ENV_ID, render_mode=None)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Infer dimensions
    sample_obs = env.observation_space.sample()
    obs_dim = sample_obs["observation"].shape[0]
    goal_dim = sample_obs["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]

    agent = HER_SAC(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        tau=0.05,
        buffer_size=500_000,
        batch_size=256,
        n_sampled_goal=4,
        target_entropy=-float(action_dim),
        learning_starts=1_000,
    )

    total_steps = args.total_timesteps
    out_dir = args.out_dir
    eval_freq = max(5_000, total_steps // 20)
    log_ep = 50

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0
    ep_success = 0
    success_count = 0
    ep_count = 0
    eval_returns = []

    print(f"\nTraining HER+SAC for {total_steps} steps")
    print(f"obs_dim={obs_dim}, goal_dim={goal_dim}, action_dim={action_dim}\n")

    step = 0
    while step < total_steps:
        action = agent.select_action(obs)
        action_clipped = action.copy()  # env does internal clipping

        next_obs, reward, terminated, truncated, info = env.step(action_clipped)

        agent.replay_buffer.push(
            obs=obs,
            achieved=obs["achieved_goal"],
            desired=obs["desired_goal"],
            action=action,
            reward=reward,
            next_obs=next_obs,
            next_achieved=next_obs["achieved_goal"],
            done=terminated or truncated,
        )

        ep_reward += reward
        if info.get("is_success", False):
            ep_success = 1
            success_count += 1

        if terminated or truncated:
            ep_count += 1
            eval_returns.append(ep_reward)

            if ep_count % log_ep == 0 and eval_returns:
                mean_ret = np.mean(eval_returns[-log_ep:])
                print(
                    f"  Step {step:>8} | Ep {ep_count:>4} | "
                    f"Return={mean_ret:.3f} | Success={success_count/max(ep_count,1):.3f}"
                )

            ep_reward = 0.0
            ep_success = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        agent.train_step(compute_reward_fn=env.unwrapped.compute_reward)
        step += 1

        if step % eval_freq == 0:
            eval_r = eval_agent(agent, eval_env, episodes=5)
            print(f"  [EVAL] Step {step:>8} | Mean Return={eval_r:.3f} | Saving checkpoint...")
            checkpoint = args.out_dir / f"checkpoint_step_{step}"
            agent.save(checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "final_model"
    agent.save(model_path)

    meta = {
        "env_id": ENV_ID,
        "total_timesteps": total_steps,
        "seed": seed,
        "eval_returns": eval_returns[-200:] if eval_returns else [],
    }
    (out_dir / "training_metadata.json").write_text(json.dumps(meta, indent=2))
    env.close()
    eval_env.close()
    return model_path


def eval_agent(agent: HER_SAC, env: gym.Env, episodes: int = 5) -> float:
    total_returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
        total_returns.append(ep_reward)
    return float(np.mean(total_returns))


def play(model_path: Path, episodes: int, max_steps: int) -> None:
    env = gym.make(ENV_ID, render_mode="human")

    sample_obs = env.observation_space.sample()
    obs_dim = sample_obs["observation"].shape[0]
    goal_dim = sample_obs["desired_goal"].shape[0]
    action_dim = env.action_space.shape[0]

    agent = HER_SAC(obs_dim=obs_dim, goal_dim=goal_dim, action_dim=action_dim)
    agent.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        successes = 0

        for _ in range(max_steps):
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            if info.get("is_success", False):
                successes += 1
            if terminated or truncated:
                break

        print(f"Episode {ep + 1}/{episodes}  reward: {ep_reward:+.3f}  successes: {successes}")

    env.close()


# ============================================================
#  CLI
# ============================================================

ENV_ID = "VS050-ReachPose-v0"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HER+SAC on VS050-ReachPose-v0")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/her_vs050"))
    parser.add_argument("--model-path", type=Path, default=Path("runs/her_vs050/final_model"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    return parser.parse_args()


out_dir = None  # global so train() can use it


def main() -> None:
    global out_dir
    args = parse_args()
    if not args.train and not args.play:
        raise SystemExit("Select at least one mode: --train and/or --play")

    out_dir = args.out_dir

    if args.train:
        model_path = train(args)
        print(f"Saved final model to: {model_path}.pt")

    if args.play:
        if not args.model_path.with_suffix(".pt").exists():
            raise FileNotFoundError(f"Model not found at {args.model_path}.pt")
        play(args.model_path, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
