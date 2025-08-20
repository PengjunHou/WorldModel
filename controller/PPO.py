# -*- coding: utf-8 -*-
"""
PPO + GNN + CNN (centralized head-car) to output a score map (continuous actions in [0,1]).
Design:
- 5 vehicles each send its 1-channel confidence map (C=1, HxW aligned in global BEV) and the 5x5 adjacency matrix to the head car.
- The model encodes each map with a shared CNN ->
    (1) a global vector z_i (node feature for GNN), and
    (2) a low-resolution feature map F_i keeping spatial structure (adaptively pooled to low_h x low_w).
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelConfig:
    C: int = 1              # input channels per vehicle (confidence map channels)
    H: int = 128            # input height (arbitrary; H != W allowed)
    W: int = 128            # input width  (arbitrary; H != W allowed)
    num_vehicles: int = 5

    enc_channels: int = 64  # channels after CNN encoder
    low_h: int = 16         # target low-res spatial height after encoder (adaptive pool to this)
    low_w: int = 16         # target low-res spatial width after encoder  (adaptive pool to this)

    gnn_hidden: int = 128   # node embedding dim for GNN
    gnn_layers: int = 2

    # Action (patch) grid. Actions are continuous in [0,1] per patch.
    patch_h: int = 16       # number of patches vertically (no need to divide low_h)
    patch_w: int = 16       # number of patches horizontally (no need to divide low_w)

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utility: down/up helpers
# -----------------------------
class ConvEncoder(nn.Module):
    """Shared CNN encoder for each vehicle's map.
    Input:  [B, C=1, H, W]
    Output: global vector z_i [B, gnn_hidden], low-res feature map F_i [B, enc_channels, low_h, low_w]
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Three conv blocks; strides (2,2,1) as before
        self.backbone = nn.Sequential(
            nn.Conv2d(cfg.C, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, cfg.enc_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # NEW: adaptively pool to fixed (low_h, low_w) regardless of input H/W
        self.adapt = nn.AdaptiveAvgPool2d((cfg.low_h, cfg.low_w))

        # Global aggregator -> node embedding for GNN
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(cfg.enc_channels, cfg.gnn_hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 1, H, W] arbitrary H/W
        F_low = self.backbone(x)               # [B, enc_channels, h', w']
        F_low = self.adapt(F_low)              # [B, enc_channels, low_h, low_w]  <-- fixed now
        g = self.gap(F_low).squeeze(-1).squeeze(-1)  # [B, enc_channels]
        z = self.proj(g)                       # [B, gnn_hidden]
        return z, F_low


class GraphSAGE(nn.Module):
    """Simple mean-aggregator GraphSAGE layer for fixed small graphs.
    h' = sigma( W_self h + W_nei mean(A*h) )
    A is adjacency (0/1 or weighted), with zero diag assumed; we'll add self-term separately.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h:   [B, N, D]
        # adj: [B, N, N]
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, N, 1]
        nei = torch.matmul(adj, h) / deg                    # [B, N, D]
        out = self.lin_self(h) + self.lin_nei(nei)
        return self.act(out)


class GraphEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        d_in = cfg.gnn_hidden
        for _ in range(cfg.gnn_layers):
            self.layers.append(GraphSAGE(d_in, d_in))

    def forward(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # z: [B, N, D], adj: [B, N, N]
        h = z
        for layer in self.layers:
            h = layer(h, adj)
        return h  # [B, N, D]


class VehicleAttentionFusion(nn.Module):
    """Compute per-vehicle weights alpha_i (softmax over N) from node embeddings,
    then fuse per-vehicle low-res feature maps via weighted sum after a 1x1 conv alignment.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.aligner = nn.Conv2d(cfg.enc_channels, cfg.enc_channels, kernel_size=1)
        self.score_mlp = nn.Sequential(
            nn.Linear(cfg.gnn_hidden, cfg.gnn_hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.gnn_hidden, 1)
        )

    def forward(self, F_list: List[torch.Tensor], z_prime: torch.Tensor) -> torch.Tensor:
        # F_list: list of [B, Cenc, low_h, low_w] for each vehicle, length N
        # z_prime: [B, N, D]
        B, N, D = z_prime.shape
        Cenc, h, w = F_list[0].shape[1:]
        # compute alpha over vehicles
        scores = self.score_mlp(z_prime)          # [B, N, 1]
        alpha = F.softmax(scores, dim=1)          # [B, N, 1]
        # align & fuse
        F_aligned = [self.aligner(Fi) for Fi in F_list]  # each [B, Cenc, h, w]
        F_stack = torch.stack(F_aligned, dim=1)          # [B, N, Cenc, h, w]
        alpha_ = alpha.view(B, N, 1, 1, 1)
        F_fused = (alpha_ * F_stack).sum(dim=1)          # [B, Cenc, h, w]
        return F_fused


class ActorDecoder(nn.Module):
    """Decode fused low-res features to per-patch Beta parameters (alpha,beta >= 1.1).
    Actions live on a (P x Q) patch grid. You can upsample later to HxW if needed.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.dec = nn.Sequential(
            nn.Conv2d(cfg.enc_channels, cfg.enc_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.enc_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 2, kernel_size=1)  # -> [alpha_raw, beta_raw]

    def forward(self, F_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, h, w = F_fused.shape
        x = self.dec(F_fused)
        # resize to (patch_h, patch_w)
        x = F.interpolate(x, size=(self.cfg.patch_h, self.cfg.patch_w), mode="bilinear", align_corners=False)
        out = self.head(x)  # [B, 2, P, Q]
        alpha_raw, beta_raw = out[:, 0], out[:, 1]  # [B, P, Q]
        # map to >= 1.1 to avoid extreme betas early in training
        alpha = F.softplus(alpha_raw) + 1.1
        beta = F.softplus(beta_raw) + 1.1
        return alpha, beta


class CriticHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(cfg.enc_channels, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, F_fused: torch.Tensor) -> torch.Tensor:
        B, C, h, w = F_fused.shape
        g = self.pool(F_fused).view(B, C)
        v = self.mlp(g).squeeze(-1)  # [B]
        return v


class ActorCritic(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg)
        self.gnn = GraphEncoder(cfg)
        self.fuse = VehicleAttentionFusion(cfg)
        self.actor = ActorDecoder(cfg)
        self.critic = CriticHead(cfg)

    def forward(self, maps: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        maps: [B, N, C=1, H, W]
        adj:  [B, N, N]
        Returns: dict with alpha,beta (Beta params), value V, and fused features.
        """
        B, N, C, H, W = maps.shape
        zs = []
        Fs = []
        for i in range(N):
            z_i, F_i = self.encoder(maps[:, i])  # F_i now fixed to [B, enc_channels, low_h, low_w]
            zs.append(z_i)
            Fs.append(F_i)
        Z = torch.stack(zs, dim=1)  # [B, N, D]
        Zp = self.gnn(Z, adj)       # [B, N, D]
        F_fused = self.fuse(Fs, Zp) # [B, Cenc, low_h, low_w]
        alpha, beta = self.actor(F_fused)  # [B, P, Q] each
        value = self.critic(F_fused)       # [B]
        return {"alpha": alpha, "beta": beta, "value": value, "F_fused": F_fused}


# -----------------------------
# PPO pieces: rollout buffer, GAE, update
# -----------------------------
class RolloutBuffer:
    def __init__(self, capacity: int, patch_shape: Tuple[int, int], device: str):
        self.capacity = capacity
        self.P, self.Q = patch_shape
        self.device = device
        self.reset()

    def reset(self):
        self.states = []      # placeholders to keep references (maps, adj)
        self.actions = []     # [B, P, Q]
        self.logps = []       # [B]
        self.values = []      # [B]
        self.rewards = []     # [B]
        self.dones = []       # [B]

    def add(self, state, action, logp, value, reward, done):
        self.states.append(state)
        self.actions.append(action.detach())
        self.logps.append(logp.detach())
        self.values.append(value.detach())
        self.rewards.append(torch.as_tensor(reward).detach())
        self.dones.append(torch.as_tensor(done).detach())

    def stack(self) -> Dict[str, torch.Tensor]:
        actions = torch.stack(self.actions)        # [T, B, P, Q]
        logps = torch.stack(self.logps)            # [T, B]
        values = torch.stack(self.values)          # [T, B]
        rewards = torch.stack(self.rewards)        # [T, B]
        dones = torch.stack(self.dones)            # [T, B]
        return {
            "actions": actions,
            "logps": logps,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "states": self.states,  # as a python list of tuples
        }


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """GAE on tensors shaped [T, B]. Returns advantages [T, B] and returns [T, B]."""
    T, B = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_val = values[t+1] if t < T-1 else torch.zeros_like(values[0])
        next_nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_nonterm - values[t]
        last_gae = delta + gamma * lam * next_nonterm * last_gae
        adv[t] = last_gae
    ret = adv + values
    return adv, ret


class PPO:
    def __init__(self, model: ActorCritic, cfg: ModelConfig):
        self.model = model
        self.cfg = cfg
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def update(self, batch: Dict[str, torch.Tensor], get_new_logp_fn):
        cfg = self.cfg
        actions = batch["actions"]           # [T, B, P, Q]
        old_logps = batch["logps"]           # [T, B]
        values = batch["values"]             # [T, B]
        rewards = batch["rewards"]           # [T, B]
        dones = batch["dones"]               # [T, B]

        T, B, P, Q = actions.shape
        device = actions.device

        adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        # flatten time-batch dims for SGD
        adv_flat = adv.view(T*B)
        ret_flat = ret.view(T*B)
        old_logp_flat = old_logps.view(T*B)

        # advantage normalization
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std(unbiased=False).clamp(min=1e-8)
        adv_flat = (adv_flat - adv_mean) / adv_std

        # recompute new log-probs given stored states+actions
        new_logps = []
        new_values = []
        entropies = []
        for t in range(T):
            maps, adj = batch["states"][t]
            out = self.model(maps, adj)
            alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
            dist = Beta(alpha, beta)
            a_t = actions[t].clamp(1e-6, 1-1e-6)  # 防止0/1导致log_prob出现 -inf/NaN
            logp = dist.log_prob(a_t).sum(dim=(1,2))  # [B]
            entropy = dist.entropy().sum(dim=(1,2))   # [B]
            new_logps.append(logp)
            new_values.append(value)
            entropies.append(entropy)

        new_logps = torch.stack(new_logps).view(T*B)
        new_values = torch.stack(new_values).view(T*B)
        entropy = torch.stack(entropies).view(T*B)

        ratio = torch.exp(new_logps - old_logp_flat)
        surr1 = ratio * adv_flat
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        # value loss (no clip for simplicity; can add value clip)
        value_loss = F.mse_loss(new_values, ret_flat)

        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.opt.step()

        with torch.no_grad():
            approx_kl = (old_logp_flat - new_logps).mean().clamp_min(0.0)
        info = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "approx_kl": approx_kl.item(),
        }
        return info


# -----------------------------
# Dummy env interface (you'll replace with your simulator/evaluator)
# -----------------------------
class DummyFleetEnv:
    """A placeholder environment that:
    - Observes: 5 vehicle maps and adjacency matrix.
    - Accepts: patch-level scores in [0,1].
    - Computes a synthetic reward combining (i) overlap-aware novelty gain and (ii) bandwidth penalty.
    Replace `step()` with your cooperative perception evaluator that computes AP/mIoU increments, etc.
    """
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.reset()

    def reset(self, B: int = 4):
        cfg = self.cfg
        # Random maps: [B, N, 1, H, W]
        maps = torch.rand(B, cfg.num_vehicles, cfg.C, cfg.H, cfg.W, device=self.device)
        # Simple ring adjacency
        adj = torch.zeros(B, cfg.num_vehicles, cfg.num_vehicles, device=self.device)
        for b in range(B):
            for i in range(cfg.num_vehicles):
                adj[b, i, (i+1)%cfg.num_vehicles] = 1.0
                adj[b, (i+1)%cfg.num_vehicles, i] = 1.0
        self.state = (maps, adj)
        return self.state
    
    @torch.no_grad()
    def step(self, score_patches: torch.Tensor):
        (maps, adj) = self.state
        B = maps.shape[0]

        P, Q = self.cfg.patch_h, self.cfg.patch_w
        low_maps = F.interpolate(
            maps.view(B * self.cfg.num_vehicles, 1, self.cfg.H, self.cfg.W),
            size=(P, Q), mode="bilinear", align_corners=False
        )
        low_maps = low_maps.view(B, self.cfg.num_vehicles, 1, P, Q)
        nov = low_maps.std(dim=1).squeeze(1)  # [B, P, Q]

        budget = 0.2 * P * Q
        usage = score_patches.sum(dim=(1, 2))
        penalty = (usage - budget).clamp(min=0.0)

        reward = (score_patches * nov).mean(dim=(1, 2)) - 0.01 * penalty
        done = torch.zeros(B, device=maps.device)
        next_state = self.reset(B)
        return next_state, reward, done


# -----------------------------
# Training loop (toy demo)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig()
    device = cfg.device

    model = ActorCritic(cfg).to(device)
    ppo = PPO(model, cfg)
    env = DummyFleetEnv(cfg)

    B = 4                 # batch of parallel episodes
    T_horizon = 32        # rollout length
    iters = 5             # a few iterations for smoke test

    for it in range(iters):
        maps, adj = env.reset(B)
        buffer = RolloutBuffer(T_horizon, (cfg.patch_h, cfg.patch_w), device)
        for t in range(T_horizon):
            out = model(maps, adj)
            alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
            dist = Beta(alpha, beta)
            action = dist.rsample()                        # [B,P,Q], reparameterized sample
            action = action.clamp(1e-6, 1-1e-6)           # 避免边界数值问题
            logp = dist.log_prob(action).sum(dim=(1,2))   # [B]

            # Step environment
            (maps_next, adj_next), reward, done = env.step(action)

            # store
            buffer.add((maps, adj), action, logp, value, reward, done)

            maps, adj = maps_next, adj_next

        batch = buffer.stack()
        info = ppo.update(batch, get_new_logp_fn=None)
        print(f"Iter {it}: loss={info['loss']:.4f} policy={info['policy_loss']:.4f} value={info['value_loss']:.4f} ent={info['entropy']:.2f} KL={info['approx_kl']:.4f}")

    print("Training loop finished (toy demo). Replace DummyFleetEnv.step with your evaluator.")
