"""Lightweight smoke test for the action-diffusion model.

This script instantiates a minimal diffusion model using the existing
``controller.model.diffusion.DiffusionModel`` implementation together with a
dummy MLP-based noise predictor.  It runs a single optimization step on random
data and prints the resulting loss along with a sampled trajectory.  The goal
is to verify that the diffusion components can execute end-to-end; it is *not*
an image generator and therefore cannot create cat photos or other pictures.

Usage:
    python -m controller.tests.diffusion_smoke_test
"""

from __future__ import annotations

import argparse
import torch
from torch import nn

from controller.model.diffusion import DiffusionModel


class DummyNoisePredictor(nn.Module):
    """Small MLP that mimics the behaviour expected by ``DiffusionModel``."""

    def __init__(
        self,
        horizon_steps: int,
        action_dim: int,
        obs_dim: int,
        cond_steps: int,
        denoising_steps: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.cond_steps = cond_steps

        self.time_embed = nn.Embedding(denoising_steps, hidden_dim)
        input_dim = horizon_steps * action_dim + cond_steps * obs_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon_steps * action_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: dict) -> torch.Tensor:
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        time_feat = self.time_embed(t)
        state = cond["state"].view(batch_size, -1)
        features = torch.cat((x_flat, state, time_feat), dim=1)
        prediction = self.net(features)
        return prediction.view(batch_size, self.horizon_steps, self.action_dim)


@torch.no_grad()
def _sample_actions(model: DiffusionModel, batch_size: int, cond_steps: int, obs_dim: int) -> torch.Tensor:
    cond = {"state": torch.randn(batch_size, cond_steps, obs_dim, device=model.device)}
    samples = model.forward(cond, deterministic=False, return_chain=False)
    return samples.trajectories


def run_smoke_test(device: str = "cpu") -> None:
    device = torch.device(device)

    horizon_steps = 4
    action_dim = 7
    obs_dim = 9
    cond_steps = 1
    denoising_steps = 100

    network = DummyNoisePredictor(
        horizon_steps=horizon_steps,
        action_dim=action_dim,
        obs_dim=obs_dim,
        cond_steps=cond_steps,
        denoising_steps=denoising_steps,
    )

    model = DiffusionModel(
        network=network,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        denoising_steps=denoising_steps,
        predict_epsilon=True,
        use_ddim=False,
    ).to(device)

    batch_size = 8
    cond = {
        "state": torch.randn(batch_size, cond_steps, obs_dim, device=device),
    }
    target_actions = torch.randn(batch_size, horizon_steps, action_dim, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    loss = model.p_losses(target_actions, cond, t=torch.randint(0, denoising_steps, (batch_size,), device=device))
    loss.backward()
    optimizer.step()

    sampled = _sample_actions(model, batch_size=2, cond_steps=cond_steps, obs_dim=obs_dim)

    #print(f"Training loss after one step: {loss.item():.6f}")
    #print(f"Sampled trajectories shape: {tuple(sampled.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke test for the diffusion model")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the test on (default: auto-detect GPU)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke_test(device=args.device)
