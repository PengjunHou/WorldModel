"""CartPole PPO smoke test for ``PPODiffusion``.

This script runs a very small PPO training loop that exercises the diffusion
policy’s RL loss instead of the earlier behaviour-cloning shortcut.  It is not
intended to reach optimal performance, but to verify that the full diffusion
PPO stack (data collection, advantage estimation, PPO loss) executes without
errors on a classic control environment.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn

from controller.model.diffusion_ppo import PPODiffusion


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class CartPoleActor(nn.Module):
    """Noise predictor used by the diffusion policy for low-dimensional obs."""

    def __init__(
        self,
        obs_dim: int,
        cond_steps: int,
        action_dim: int,
        horizon_steps: int,
        denoising_steps: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.cond_steps = cond_steps
        self.time_embed = nn.Embedding(denoising_steps, hidden_dim)
        input_dim = horizon_steps * action_dim + cond_steps * obs_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, horizon_steps * action_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: dict) -> torch.Tensor:
        batch = x.size(0)
        features = torch.cat(
            (
                x.view(batch, -1),
                cond["state"].view(batch, -1),
                self.time_embed(t),
            ),
            dim=1,
        )
        noise = self.net(features)
        return noise.view(batch, self.horizon_steps, self.action_dim)


class CartPoleCritic(nn.Module):
    """State-value head for PPO updates."""

    def __init__(self, obs_dim: int, cond_steps: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * cond_steps, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cond: dict | torch.Tensor) -> torch.Tensor:
        if isinstance(cond, dict):
            state = cond["state"].view(cond["state"].size(0), -1)
        else:
            state = cond
        return self.net(state)


# ---------------------------------------------------------------------------
# Rollout collection / BC utilities
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    state: torch.Tensor         # [T, cond_steps, obs_dim]
    chains: torch.Tensor        # [T, K+1, horizon, action]
    logprobs: torch.Tensor      # [T, K, horizon, action]
    values: torch.Tensor        # [T]
    returns: torch.Tensor       # [T]
    advantages: torch.Tensor    # [T]


@dataclass
class BCDataset:
    states: torch.Tensor  # [N, obs_dim]
    actions: torch.Tensor  # [N, 1] in [-1, 1]
    masks: torch.Tensor   # [N]


def heuristic_policy(obs: np.ndarray) -> int:
    position, velocity, angle, angular_vel = obs
    score = angle + 0.5 * angular_vel + 0.05 * velocity + 0.01 * position
    return 1 if score > 0 else 0


def collect_bc_dataset(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], int],
    num_samples: int,
) -> BCDataset:
    samples = []
    actions = []
    masks = []
    obs, _ = env.reset()
    steps = 0
    total_reward = 0.0
    while len(samples) < num_samples:
        act = policy_fn(obs)
        next_obs, reward, terminated, truncated, _ = env.step(act)
        samples.append(obs.astype(np.float32))
        actions.append(act)
        keep = float(steps < 150 and total_reward >= -10.0)
        masks.append(keep)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            #print(f"[bc] episode reward: {total_reward:.1f}, steps: {steps}")
            steps = 0
            total_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs
    states = torch.tensor(np.stack(samples), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
    masks = torch.tensor(masks, dtype=torch.float32)
    actions = 2.0 * actions - 1.0
    return BCDataset(states=states, actions=actions, masks=masks)


def diffusion_supervised_loss(
    model: PPODiffusion,
    actions: torch.Tensor,
    cond: dict,
    timesteps: torch.Tensor,
    predict_epsilon: bool,
) -> torch.Tensor:
    noise = torch.randn_like(actions, device=actions.device)
    noisy = model.q_sample(x_start=actions, t=timesteps, noise=noise)
    if predict_epsilon:
        pred = model.actor_ft(noisy, timesteps, cond)
        target = noise
    else:
        pred = model.actor_ft(noisy, timesteps, cond)
        target = actions
    return torch.nn.functional.mse_loss(pred, target)


def collect_rollout(
    model: PPODiffusion,
    env: gym.Env,
    obs: np.ndarray,
    steps: int,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> Tuple[RolloutBatch, np.ndarray]:
    states = []
    chains = []
    logprobs = []
    rewards = []
    dones = []
    values = []

    for _ in range(steps):
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        cond = {"state": state_tensor.unsqueeze(1)}
        with torch.no_grad():
            samples = model(cond=cond, deterministic=False, return_chain=True)
            value = model.critic(cond).squeeze().item()
            chain_tensor = samples.chains.squeeze(0).cpu()
            logprob = model.get_logprobs(cond, samples.chains, get_ent=False)
            logprob = logprob.view(model.ft_denoising_steps, model.horizon_steps, model.action_dim)
            action_value = samples.trajectories[0, 0, 0].item()

        action = 1 if action_value > 0.0 else 0
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state_tensor.squeeze(0).cpu())
        chains.append(chain_tensor)
        logprobs.append(logprob.cpu())
        rewards.append(reward)
        dones.append(done)
        values.append(value)

        obs = next_obs if not done else env.reset()[0]

    with torch.no_grad():
        final_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        cond_final = {"state": final_state.unsqueeze(1)}
        next_value = model.critic(cond_final).squeeze().item()
    if dones[-1]:
        next_value = 0.0

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    advantages = torch.zeros_like(rewards_t)
    returns = torch.zeros_like(rewards_t)

    gae = 0.0
    next_val = next_value
    for step in reversed(range(steps)):
        mask = 1.0 - dones_t[step]
        delta = rewards_t[step] + gamma * next_val * mask - values_t[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
        returns[step] = gae + values_t[step]
        next_val = values_t[step]

    batch = RolloutBatch(
        state=torch.stack(states),
        chains=torch.stack(chains),
        logprobs=torch.stack(logprobs),
        values=values_t,
        returns=returns,
        advantages=advantages,
    )
    return batch, obs


def ppo_update(
    model: PPODiffusion,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    epochs: int,
    minibatch_size: int,
    ent_coef: float,
    vf_coef: float,
    update_actor: bool,
) -> None:
    model.train()
    ft_steps = model.ft_denoising_steps
    total_steps = batch.state.size(0)
    batch.state = batch.state.to(model.device)
    batch.chains = batch.chains.to(model.device)
    batch.logprobs = batch.logprobs.to(model.device)
    batch.values = batch.values.to(model.device)
    batch.returns = batch.returns.to(model.device)
    batch.advantages = batch.advantages.to(model.device)

    advantages = batch.advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        perm = torch.randperm(total_steps, device=model.device)
        for start in range(0, total_steps, minibatch_size):
            idx = perm[start : start + minibatch_size]
            obs_b = {"state": batch.state[idx]}
            denoising_inds = torch.randint(0, ft_steps, (idx.size(0),), device=model.device)
            chains_prev = batch.chains[idx, denoising_inds]
            chains_next = batch.chains[idx, denoising_inds + 1]
            returns_b = batch.returns[idx]
            values_b = batch.values[idx]
            adv_b = advantages[idx]
            oldlogprobs_b = batch.logprobs[idx, denoising_inds]

            (
                pg_loss,
                entropy_loss,
                v_loss,
                _,
                approx_kl,
                _,
                bc_loss,
                _,
            ) = model.loss(
                obs_b,
                chains_prev,
                chains_next,
                denoising_inds,
                returns_b,
                values_b,
                adv_b,
                oldlogprobs_b,
                use_bc_loss=True,
                reward_horizon=1,
            )
            loss = pg_loss + entropy_loss * ent_coef + v_loss * vf_coef + bc_loss

            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor_ft.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            if update_actor:
                actor_optimizer.step()
            critic_optimizer.step()

            if approx_kl > 0.2:  # crude early-stop safeguard
                return


# ---------------------------------------------------------------------------
# Training and evaluation entry points
# ---------------------------------------------------------------------------


def build_model(device: torch.device, denoising_steps: int = 16) -> PPODiffusion:
    obs_dim = 4
    action_dim = 1
    cond_steps = 1
    horizon_steps = 1

    actor = CartPoleActor(
        obs_dim=obs_dim,
        cond_steps=cond_steps,
        action_dim=action_dim,
        horizon_steps=horizon_steps,
        denoising_steps=denoising_steps,
    )
    critic = CartPoleCritic(obs_dim=obs_dim, cond_steps=cond_steps)

    model = PPODiffusion(
        gamma_denoising=0.99,
        clip_ploss_coef=0.2,
        actor=actor,
        critic=critic,
        ft_denoising_steps=denoising_steps,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        denoising_steps=denoising_steps,
        device=device,
        use_ddim=False,
        learn_eta=False,
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        randn_clip_value=3.0,
        final_action_clip_value=1.0,
    ).to(device)
    return model


def train_diffusion_agent(
    device: torch.device,
    total_updates: int,
    steps_per_update: int,
    ppo_epochs: int,
    minibatch_size: int,
    critic_warmup: int,
    bc_samples: int,
    bc_steps: int,
    predict_epsilon: bool,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> PPODiffusion:
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()

    model = build_model(device)
    model.predict_epsilon = predict_epsilon
    #print(f"predict_epsilon: {model.predict_epsilon}")
    actor_optimizer = torch.optim.Adam(model.actor_ft.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=3e-4)

    if bc_steps > 0 and bc_samples > 0:
        bc_env = gym.make("CartPole-v1")
        dataset = collect_bc_dataset(bc_env, heuristic_policy, bc_samples)
        bc_env.close()
        states = dataset.states.to(device)
        actions = dataset.actions.to(device).view(-1, 1, 1)
        model.train()
    for step in range(bc_steps):
        idx = torch.randint(0, states.size(0), (minibatch_size,), device=device)
        cond = {"state": states[idx].unsqueeze(1)}
        timesteps = torch.randint(0, model.denoising_steps, (minibatch_size,), device=device)
        filtered_actions = actions[idx]
        loss = diffusion_supervised_loss(
            model, filtered_actions, cond, timesteps, predict_epsilon
        )
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor_ft.parameters(), 1.0)
        actor_optimizer.step()
        if (step + 1) % 100 == 0:
            #print(f"[bc] step={step + 1:04d} loss={loss.item():.4f}")
    
    # 测试学习到的策略如何
    evaluate(
        model,
        episodes=5,
        render_human=True,
        save_gif=None,
    )

    for update in range(total_updates):
        model.eval()
        rollout, obs = collect_rollout(
            model,
            env,
            obs,
            steps_per_update,
            device,
            gamma,
            gae_lambda,
        )
        #print(
            f"[collect] update={update + 1:03d} mean_reward={rollout.returns.mean().item():.2f}"
        )
        ppo_update(
            model,
            actor_optimizer,
            critic_optimizer,
            rollout,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            ent_coef=0.01,
            vf_coef=0.5,
            update_actor=update >= critic_warmup,
        )

    env.close()
    return model


def evaluate(
    model: PPODiffusion,
    episodes: int,
    render_human: bool,
    save_gif: Optional[Path],
) -> None:
    render_mode: Optional[str]
    frames = []
    if save_gif is not None:
        render_mode = "rgb_array"
    elif render_human:
        render_mode = "human"
    else:
        render_mode = None

    env = gym.make("CartPole-v1", render_mode=render_mode)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            state_tensor = (
                torch.tensor(obs, dtype=torch.float32, device=model.device)
                .unsqueeze(0)
                .unsqueeze(1)
            )
            cond = {"state": state_tensor}
            with torch.no_grad():
                sample = model(cond, deterministic=True, return_chain=True)
            action_value = sample.trajectories[0, 0, 0].item()
            action = 1 if action_value > 0.0 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            if save_gif is not None and ep == 0:
                frame = env.render()
                frames.append(frame)
            elif render_human:
                env.render()
            total += reward
            done = terminated or truncated
        returns.append(total)
        #print(f"[eval] episode={ep + 1} return={total:.1f}")

    mean_return = sum(returns) / len(returns)
    #print(f"Mean return: {mean_return:.1f}")

    if save_gif is not None and frames:
        save_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(save_gif, frames, fps=30)
        #print(f"Saved animation to {save_gif}")

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CartPole PPODiffusion PPO test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--steps-per-update", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--critic-warmup", type=int, default=10, help="Number of PPO updates to train critic before actor updates")
    parser.add_argument("--bc-samples", type=int, default=2048, help="Number of heuristic samples for BC warm-up")
    parser.add_argument("--bc-steps", type=int, default=500, help="Number of BC gradient steps before PPO")
    parser.add_argument("--predict-epsilon", action="store_true", help="Train diffusion to predict epsilon (noise); otherwise predict x0")
    parser.add_argument("--render-human", action="store_true")
    parser.add_argument("--save-gif", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    #print(f"Using render_human: {args.render_human}")

    device = torch.device(args.device)
    model = train_diffusion_agent(
        device=device,
        total_updates=args.updates,
        steps_per_update=args.steps_per_update,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch,
        critic_warmup=args.critic_warmup,
        bc_samples=args.bc_samples,
        bc_steps=args.bc_steps,
        predict_epsilon=args.predict_epsilon,
    )

    gif_path = Path(args.save_gif) if args.save_gif else None
    evaluate(
        model,
        episodes=args.episodes,
        render_human=args.render_human,
        save_gif=gif_path,
    )


if __name__ == "__main__":
    main()
