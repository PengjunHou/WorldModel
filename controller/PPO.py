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
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

import os
import time
import shutil

# -----------------------------
# Config
# -----------------------------
@dataclass
class ModelConfig:
    C: int = 2             # input channels per vehicle (confidence map channels)
    H: int = 42            # input height (arbitrary; H != W allowed)
    W: int = 44           # input width  (arbitrary; H != W allowed)
    num_vehicles: int = 5

    enc_channels: int = 64  # channels after CNN encoder
    low_h: int = 16         # target low-res spatial height after encoder (adaptive pool to this)
    low_w: int = 16         # target low-res spatial width after encoder  (adaptive pool to this)

    gnn_hidden: int = 128   # node embedding dim for GNN
    gnn_layers: int = 2

    # Action (patch) grid. Actions are continuous in [0,1] per patch.
    patch_h: int = 42       # number of patches vertically (no need to divide low_h)
    patch_w: int = 44      # number of patches horizontally (no need to divide low_w)

    # budget hidden
    film_hidden: int = 128

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.1
    vf_coef: float = 0.001
    max_grad_norm: float = 0.5
    lr: float = 0.00001

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utility: down/up helpers
# -----------------------------
class ConvEncoder(nn.Module):
    """Shared CNN encoder for each vehicle's map, including local map and fused map
    Input:  [B, C=2, H, W]
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
        # x: [B, 2, H, W] 
        F_low = self.backbone(x)               # [B, enc_channels, h', w']
        F_low = self.adapt(F_low)              # [B, enc_channels, low_h, low_w]  <-- fixed now
        g = self.gap(F_low).squeeze(-1).squeeze(-1)  # [B, enc_channels]
        z = self.proj(g)                       # [B, gnn_hidden]
        return z, F_low


class GraphSAGE(nn.Module):
    """Mean-aggregator GraphSAGE layer for fixed small graphs.
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


# -----------------------------
# Residual + SE blocks (用于更深的 Actor 和 Critic)
# -----------------------------
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, max(4, c // r)), nn.ReLU(inplace=True),
            nn.Linear(max(4, c // r), c), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            nn.GroupNorm(8, c),
            nn.Conv2d(c, c, 3, padding=1)
        )
        self.se = SE(c)
    def forward(self, x):
        y = self.net(x)
        y = self.se(y)
        return F.relu(x + y, inplace=True)


# -----------------------------
# Stronger ActorDecoder
# -----------------------------
class ActorDecoder(nn.Module):
    """Deeper conv decoder with residual blocks + two-stage upsampling."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.enc_channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            ResBlock(C), ResBlock(C)
        )
        # 先到一半分辨率
        mid_h, mid_w = max(1, cfg.patch_h // 2), max(1, cfg.patch_w // 2)
        self.mid_h, self.mid_w = mid_h, mid_w

        self.stage2 = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            ResBlock(C), nn.Conv2d(C, 96, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(inplace=True),
            ResBlock(64)
        )
        self.head = nn.Conv2d(64, 2, kernel_size=1)
        self.cfg = cfg

    def forward(self, F_fused: torch.Tensor):
        x = self.stage1(F_fused)  
        # 到中尺度
        x = F.interpolate(x, size=(self.mid_h, self.mid_w), mode="bilinear", align_corners=False)
        x = self.stage2(x)
        # 到目标尺寸
        x = F.interpolate(x, size=(self.cfg.patch_h, self.cfg.patch_w), mode="bilinear", align_corners=False)
        x = self.stage3(x)
        out = self.head(x)  # [B,2,H,W]
        alpha_raw, beta_raw = out[:, 0], out[:, 1]
        alpha = F.softplus(alpha_raw) + 1.1
        beta  = F.softplus(beta_raw)  + 1.1
        return alpha, beta


# -----------------------------
# Stronger CriticHead
# -----------------------------
class CriticHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.enc_channels
        self.tower = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
            ResBlock(C)
        )
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * C, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    def forward(self, F_fused: torch.Tensor):
        x = self.tower(F_fused)
        a = self.pool_avg(x).flatten(1)
        m = self.pool_max(x).flatten(1)
        g = torch.cat([a, m], dim=1)      # [B, 2C]
        v = self.mlp(g).squeeze(-1)
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

        # FiLM: 输入为2个标量 [prev_budget, curr_budget] -> 2*enc_channels (gamma,beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(2, cfg.film_hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.film_hidden, cfg.enc_channels * 2)
        )

    def _budget_pair_for_vehicle(self, prev_budget, curr_budget, i: int, B: int, norm: float):
        """
        支持 prev/curr_budget 形状为 [B] 或 [B,N]。
        返回 [B,2]，并做归一化到 [0,1]（除以 P*Q）。
        """
        if prev_budget is None:
            pb = torch.zeros(B, device=self.cfg.device)
        else:
            pb = prev_budget if prev_budget.dim() == 1 else prev_budget[:, i]
        if curr_budget is None:
            cb = torch.zeros(B, device=self.cfg.device)
        else:
            cb = curr_budget if curr_budget.dim() == 1 else curr_budget[:, i]
        pair = torch.stack([pb / norm, cb / norm], dim=-1).clamp(0, 1)  # [B,2]
        return pair

    def forward(
        self,
        maps: torch.Tensor,                  # [B,N,1,H,W]  (local maps)
        adj: torch.Tensor,                   # [B,N,N]
        prev_fused_per_veh: torch.Tensor,    # [B,N,1,H,W]
        prev_budget: Optional[torch.Tensor] = None,  # [B] 或 [B,N]
        curr_budget: Optional[torch.Tensor] = None,  # [B] 或 [B,N]
    ) -> Dict[str, torch.Tensor]:

        B, N, C, H, W = maps.shape
        low_h, low_w = self.cfg.low_h, self.cfg.low_w
        norm = float(self.cfg.patch_h * self.cfg.patch_w)

        zs, Fs = [], []
        for i in range(N):
            # 早期融合输入: [B,2,H,W]
            x2 = torch.cat([maps[:, i], prev_fused_per_veh[:, i]], dim=1)
            z_i, F_i = self.encoder(x2)  # [B,enc,low_h,low_w]

            # FiLM 条件化: 基于该车的(prev,curr)预算 -> (gamma,beta)
            pair = self._budget_pair_for_vehicle(prev_budget, curr_budget, i, B, norm)  # [B,2]
            gb = self.film_mlp(pair)                    # [B, 2*enc]
            gamma, beta = torch.chunk(gb, 2, dim=-1)    # [B,enc], [B,enc]
            gamma = gamma.view(B, self.cfg.enc_channels, 1, 1)
            beta  = beta.view(B, self.cfg.enc_channels, 1, 1)
            F_i = gamma * F_i + beta                    # FiLM 调制

            zs.append(z_i)
            Fs.append(F_i)

        Z = torch.stack(zs, dim=1)          # [B,N,D]
        Zp = self.gnn(Z, adj)               # [B,N,D]
        F_fused = self.fuse(Fs, Zp)         # [B,enc,low_h,low_w]

        alpha, beta = self.actor(F_fused)   # [B,P,Q]
        value = self.critic(F_fused)        # [B]
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
        self.states = []      # list of tuples: (maps, adj, prev_fused_per_veh, prev_budget, curr_budget)
        self.actions = []
        self.logps = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, logp, value, reward, done):
        # state: tuple of tensors
        def det(x):
            return x.detach() if torch.is_tensor(x) else x
        self.states.append(tuple(det(s) for s in state))
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
            "states": self.states,
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
        # 可调
        self.num_epochs = 4
        self.target_kl = 1e-3   # 过大则提前停止本批

    def _forward_all(self, batch):
        """用当前模型前向整批，返回 new_logps/new_values/entropy/alpha+beta 和便于日志的统计。"""
        actions = batch["actions"]           # [T,B,P,Q]
        states  = batch["states"]            # list of T tuples (maps, adj)
        T, B, P, Q = actions.shape

        new_logps, new_values, entropies = [], [], []
        alphas, betas = [], []
        for t in range(T):
            maps, adj, prev_fused_per_veh, prev_budget, curr_budget = states[t]
            out = self.model(maps, adj, prev_fused_per_veh, prev_budget, curr_budget)
            alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
            alphas.append(alpha); betas.append(beta)
            dist = Beta(alpha, beta)
            a_t = actions[t].clamp(1e-6, 1-1e-6)
            logp = dist.log_prob(a_t).sum(dim=(1,2))      # [B]
            entropy = dist.entropy().sum(dim=(1,2))       # [B]
            new_logps.append(logp); new_values.append(value); entropies.append(entropy)

        alphas = torch.stack(alphas)                      # [T,B,P,Q]
        betas  = torch.stack(betas)
        new_logps = torch.stack(new_logps).view(T*B)     # [T*B]
        new_values = torch.stack(new_values).view(T*B)   # [T*B]
        entropy = torch.stack(entropies).view(T*B)       # [T*B]

        with torch.no_grad():
            ab_sum = (alphas + betas).mean().item()
            ent_mean = entropy.mean().item()
        return new_logps, new_values, entropy, ab_sum, ent_mean
    
    def action_select(self, states, deterministic=False):
        local_maps, adjs, fused_maps, prev_b, curr_b = states
        # ---- 1) 转成 torch.Tensor 并放到正确设备 ----
        # feature = torch.as_tensor(state, dtype=torch.float32, device=device)          # [B,N,1,H,W]
        # adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.float32, device=device)  # [B,N,N]

        # ---- 2) 前向网络，采样 scores_map（[B,P,Q]）----
        out = self.model(local_maps, adjs, fused_maps, prev_b, curr_b)
        alpha, beta, value = out["alpha"], out["beta"], out["value"]  # [B,P,Q], [B]
        dist = Beta(alpha, beta)
        scores_map = dist.rsample().clamp(1e-6, 1-1e-6)               # [B,P,Q]
        logp = dist.log_prob(scores_map).sum(dim=(1, 2))
        return scores_map, logp, value


    def update(self, batch: Dict[str, torch.Tensor], get_new_logp_fn=None):
        cfg = self.cfg
        actions = batch["actions"]           # [T,B,P,Q]
        old_logps = batch["logps"].view(-1)  # [T*B] 采样时保存的 logp
        values = batch["values"]             # [T,B]
        rewards = batch["rewards"]           # [T,B]
        dones = batch["dones"]               # [T,B]
        T, B, P, Q = actions.shape

        adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv_flat = adv.view(T*B)
        ret_flat = ret.view(T*B)

        # 原始 GAE 统计
        with torch.no_grad():
            adv_raw = adv_flat
            gae_mean = adv_raw.mean().item(); gae_std = adv_raw.std(unbiased=False).item()
            gae_nonzero = (adv_raw.abs() > 1e-8).float().mean().item()

        # 归一化 advantage
        adv_flat = (adv_flat - adv_flat.mean()) / adv_flat.std(unbiased=False).clamp(min=1e-8)

        # ===== 多个 epoch 的更新 =====
        kl_reached = False
        last_policy_loss = last_value_loss = last_entropy = 0.0
        for epoch in range(self.num_epochs):
            new_logps, new_values, entropy, ab_sum, ent_mean = self._forward_all(batch)

            ratio = torch.exp(new_logps - old_logps)             # [T*B]
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(new_values, ret_flat)
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.opt.step()

            with torch.no_grad():
                approx_kl = (old_logps - new_logps).mean().clamp_min(0.0).item()
            last_policy_loss = policy_loss.item()
            last_value_loss = value_loss.item()
            last_entropy = ent_mean

            # 早停：KL 超标则不再继续本批
            if approx_kl > self.target_kl:
                kl_reached = True
                break

        # 训练完这一批后，再前向一次用于“训练后”的统计（这次肯定 != 1 / 0）
        with torch.no_grad():
            post_new_logps, _, post_entropy, ab_sum, ent_mean = self._forward_all(batch)
            post_ratio = torch.exp(post_new_logps - old_logps)
            ratio_mean = post_ratio.mean().item()
            ratio_std  = post_ratio.std(unbiased=False).item()
            ratio_min  = post_ratio.min().item()
            ratio_max  = post_ratio.max().item()
            ratio_close1 = ((post_ratio - 1).abs() < 1e-6).float().mean().item()
            approx_kl = (old_logps - post_new_logps).mean().clamp_min(0.0).item()

        info = {
            "loss": last_policy_loss + cfg.vf_coef * last_value_loss - cfg.ent_coef * last_entropy,
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": ent_mean,
            "approx_kl": approx_kl,
            "gae_mean": gae_mean,
            "gae_std": gae_std,
            "gae_nonzero": gae_nonzero,
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "ratio_close1": ratio_close1,
            "alpha_beta_sum_mean": ab_sum,
            "kl_earlystop": kl_reached,
        }
        return info


def save_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    cfg: Any,
                    out_dir: str = "./checkpoints",
                    step: Optional[int] = None,
                    is_best: bool = False,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """
    Args:
        model: 你的 ActorCritic 模型或任意 nn.Module
        optimizer: 优化器（可为 None）
        cfg: 配置对象（支持 dataclass 或普通 dict/对象）
        out_dir: 输出目录
        step: 当前训练步数/迭代（用于命名；None 则记作 'final'）
        is_best: 是否同时更新 best.pt
        extra: 额外想保存的字典（比如本次评估指标 info）

    Returns:
        ckpt_path: 本次保存的 checkpoint 文件路径
    """
    os.makedirs(out_dir, exist_ok=True)

    tag = f"step{int(step):06d}" if step is not None else "final"
    ckpt_path = os.path.join(out_dir, f"ppo_actorcritic_{tag}.pt")

    # 尝试把 dataclass 配置转成 dict；否则直接保存对象（可 pickled）
    try:
        cfg_to_save = asdict(cfg)
    except Exception:
        cfg_to_save = cfg

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "cfg": cfg_to_save,
        "step": step,
        "timestamp": time.time(),
        "extra": extra or {},
    }

    # 保存随机数状态，便于完全复现
    try:
        payload["rng_state"] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    except Exception:
        pass

    torch.save(payload, ckpt_path)
    latest_path = os.path.join(out_dir, "latest.pt")
    shutil.copyfile(ckpt_path, latest_path)
    if is_best:
        best_path = os.path.join(out_dir, "best.pt")
        shutil.copyfile(ckpt_path, best_path)

    return ckpt_path

def load_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    ckpt_path: str = "./checkpoints/latest.pt",
                    map_location: Optional[str] = None,
                    strict: bool = True,
                    load_optimizer: bool = True,
                    resume_rng: bool = False,
                    dataclass_type: Optional[type] = None,
                    move_optimizer_to_device: bool = True,
                    verbose: bool = True) -> Dict[str, Any]:
    """
    从 checkpoint 恢复模型/优化器/配置/随机数状态。

    Args:
        model: 你的 ActorCritic（或任意 nn.Module）
        optimizer: 优化器（可为 None）
        ckpt_path: 路径或目录；若是目录，自动拼接 latest.pt
        map_location: torch.load 的 map_location（默认自动：模型所在设备）
        strict: load_state_dict 的 strict
        load_optimizer: 是否加载优化器状态
        resume_rng: 是否恢复 Python/Torch/CUDA 的随机数状态
        dataclass_type: 若提供且保存的 cfg 是 dict，则用它还原为 dataclass（如 ModelConfig）
        move_optimizer_to_device: 将优化器 state tensor 移到模型参数所在设备
        verbose: 打印简要信息

    Returns:
        info: 包含 { 'cfg', 'step', 'timestamp', 'extra', 'path', 'payload_keys' }
    """
    # 1) 解析路径
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "latest.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 2) 决定 map_location（默认跟随模型参数设备）
    if map_location is None:
        try:
            device = next(model.parameters()).device
            map_location = device.type if device.type != "cuda" else f"cuda:{device.index or 0}"
        except StopIteration:
            map_location = "cpu"

    payload = torch.load(ckpt_path, map_location=map_location, weights_only = False)

    # 3) 加载模型参数
    model.load_state_dict(payload["model_state"], strict=strict)

    # 4) 加载优化器并把 state tensor 移到模型设备
    if load_optimizer and optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        if move_optimizer_to_device:
            model_device = next(model.parameters()).device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(model_device)

    # 5) 还原 cfg（如果要求 dataclass）
    cfg_loaded = payload.get("cfg", None)
    if dataclass_type is not None and isinstance(cfg_loaded, dict):
        try:
            cfg_loaded = dataclass_type(**cfg_loaded)
        except Exception:
            # 如果字段不匹配就保留 dict，避免抛异常
            pass

    # 6) 恢复随机数状态（可选）
    if resume_rng and "rng_state" in payload:
        try:
            import random as pyrandom
            rng = payload["rng_state"]
            if rng.get("python") is not None:
                pyrandom.setstate(rng["python"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            # 忽略随机数恢复失败
            pass

    if verbose:
        step = payload.get("step", None)
        #print(f"[load] loaded '{ckpt_path}' (step={step}, strict={strict}, map_location={map_location})")

    return {
        "cfg": cfg_loaded,
        "step": payload.get("step", None),
        "timestamp": payload.get("timestamp", None),
        "extra": payload.get("extra", {}),
        "path": ckpt_path,
        "payload_keys": list(payload.keys()),
    }

# -----------------------------
# Dummy env interface (you'll replace with your simulator/evaluator)
# -----------------------------
class DummyFleetEnv:
    """
    占位环境：
    - 维护 per-vehicle prev_fused_per_veh: [B,N,1,H,W]
    - 维护 prev_budget/curr_budget: [B,N]
    - 奖励 = 新颖性覆盖 - 超预算惩罚
    """
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.prev_fused_per_veh = None  # [B,N,1,H,W]
        self.prev_budget = None         # [B,N]
        self.curr_budget = None         # [B,N]
        self.state = None

    def reset(self, B: int = 4):
        cfg = self.cfg
        maps = torch.rand(B, cfg.num_vehicles, 1, cfg.H, cfg.W, device=self.device)
        adj = torch.zeros(B, cfg.num_vehicles, cfg.num_vehicles, device=self.device)
        for b in range(B):
            for i in range(cfg.num_vehicles):
                adj[b, i, (i+1) % cfg.num_vehicles] = 1.0
                adj[b, (i+1) % cfg.num_vehicles, i] = 1.0

        if self.prev_fused_per_veh is None:
            self.prev_fused_per_veh = torch.zeros(B, cfg.num_vehicles, 1, cfg.H, cfg.W, device=self.device)
        if self.prev_budget is None:
            base = int(0.2 * cfg.patch_h * cfg.patch_w)
            self.prev_budget = torch.full((B, cfg.num_vehicles), base, device=self.device, dtype=torch.float32)

        # 随机扰动生成当前预算
        noise = torch.empty_like(self.prev_budget).uniform_(0.9, 1.1)
        maxb = cfg.patch_h * cfg.patch_w
        self.curr_budget = (self.prev_budget * noise).clamp(0, maxb)

        self.state = (maps, adj)
        return maps, adj

    @torch.no_grad()
    def step(self, score_patches: torch.Tensor):
        (maps, adj) = self.state
        B = maps.shape[0]; N = self.cfg.num_vehicles
        P, Q = self.cfg.patch_h, self.cfg.patch_w
        cfg = self.cfg

        # 低分辨率下的各车图
        low = F.interpolate(
            maps.view(B*N, 1, cfg.H, cfg.W),
            size=(P, Q), mode="bilinear", align_corners=False
        ).view(B, N, 1, P, Q)  # [B,N,1,P,Q]

        # 新颖性： across vehicles 的 std
        nov = low.std(dim=1).squeeze(1)  # [B,P,Q]

        # 带宽使用（简单按 head 车动作的和）
        usage = score_patches.sum(dim=(1, 2))  # [B]
        budget = self.curr_budget.mean(dim=1)  # [B] 简化：平均预算
        penalty = (usage - budget).clamp(min=0.0)
        reward = (score_patches * nov).mean(dim=(1, 2)) - 0.01 * penalty
        done = torch.zeros(B, device=maps.device)

        # 生成每车的“当前 fused”（这里只是示例：用自身低分辨率图上采样回 HxW）
        fused_low_per_veh = low  # 你可以替换成真实的按邻居融合逻辑
        fused_up_per_veh = F.interpolate(
            fused_low_per_veh.view(B*N, 1, P, Q),
            size=(cfg.H, cfg.W), mode="bilinear", align_corners=False
        ).view(B, N, 1, cfg.H, cfg.W)

        # 更新到“下一时刻的 prev”
        self.prev_fused_per_veh = fused_up_per_veh.detach()
        self.prev_budget = self.curr_budget.detach().clone()

        # 下一状态
        maps_next, adj_next = self.reset(B)
        self.state = (maps_next, adj_next)
        return (maps_next, adj_next), reward, done


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

    B = 4
    T_horizon = 16
    iters = 3

    for it in range(iters):
        maps, adj = env.reset(B)
        buffer = RolloutBuffer(T_horizon, (cfg.patch_h, cfg.patch_w), device)

        for t in range(T_horizon):
            out = model(
                maps, adj,
                prev_fused_per_veh=env.prev_fused_per_veh,  # [B,N,1,H,W]
                prev_budget=env.prev_budget,                # [B,N]
                curr_budget=env.curr_budget,                # [B,N]
            )
            alpha, beta, value = out["alpha"], out["beta"], out["value"]
            dist = Beta(alpha, beta)
            action = dist.rsample().clamp(1e-6, 1-1e-6)
            logp = dist.log_prob(action).sum(dim=(1, 2))

            (maps_next, adj_next), reward, done = env.step(action)

            state_tuple = (maps, adj, env.prev_fused_per_veh, env.prev_budget, env.curr_budget)
            buffer.add(state_tuple, action, logp, value, reward, done)

            maps, adj = maps_next, adj_next

        batch = buffer.stack()
        info = ppo.update(batch, get_new_logp_fn=None)
        print(f"Iter {it}: loss={info['loss']:.4f} policy={info['policy_loss']:.4f} "
              f"value={info['value_loss']:.4f} ent={info['entropy']:.2f} KL={info['approx_kl']:.4f} "
              f"ratio_mean={info['ratio_mean']:.4f} ratio_std={info['ratio_std']:.4f}")