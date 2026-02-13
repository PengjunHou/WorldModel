import os
import numpy as np
from PIL import Image
from math import radians, sin, cos
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchdiffeq import odeint
from evaluate import rollout_one_batch, compare_rollout
import csv
import matplotlib
matplotlib.use("Agg")   # 必须在 import pyplot 之前
import matplotlib.pyplot as plt



import random
from torch.utils.data import Subset

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_dataset(ds, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Returns: train_ds, val_ds, test_ds (Subset)
    """
    assert 0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)

@torch.no_grad()
def evaluate(model, dl, device, H, dt, solver="dopri5", lambda_z=0.01):
    """
    Latent版评估：
      - 必算：state trajectory loss（pred_state vs states_future）
      - 可选：latent loss（pred_z vs z_future），如果 batch 里有 z_future

    期望 batch 包含：
      z_past: (B,L,N,Z)
      states_past: (B,L,N,6)
      states_future: (B,H,N,6)
      group_id: (B,N)
      node_mask: (B,N)
      (optional) z_future: (B,H,N,Z)

    model.forward 返回：
      pred_state: (B,H,N,6)
      pred_z:     (B,H,N,Z)
    """
    model.eval()
    total = 0.0
    count = 0

    for batch in dl:
        z_past = batch["z_past"].to(device)
        x = batch["states_past"].to(device)
        y = batch["states_future"].to(device)
        gid = batch["group_id"].to(device)
        node_mask = batch["node_mask"].to(device)

        pred_state, pred_z = model(
            z_past, x, gid, node_mask,
            H=H, dt=dt, k=2, solver=solver
        )

        # # 轨迹 loss（你原先已有的）
        # loss_state = traj_loss(pred_state, y, node_mask)

        # latent loss（如果有 z_future）
        if ("z_future" in batch) and (batch["z_future"] is not None):
            gt_z = batch["z_future"].to(device)
            # 你如果已经有 loss_total，就用 loss_total；没有的话用一个简单的 masked MSE
            # loss_z = ((pred_z - gt_z) ** 2).mean(dim=-1)          # (B,H,N)
            # loss_z = loss_z[:, :, node_mask].mean() if node_mask.any() else loss_z.mean()
            loss, info = loss_total(pred_state, y, node_mask, pred_z=pred_z, gt_z=gt_z, lambda_z=lambda_z)
        else:
            loss, info = loss_total(pred_state, y, node_mask)

        # print(f"Eval batch: loss={loss.item():.4f}, info={info}")
        bs = z_past.size(0)
        total += float(loss.item()) * bs
        count += bs

    return total / max(count, 1)


def save_checkpoint(save_path, model, opt, epoch, best_val, extra=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_val": best_val,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
    }
    if extra is not None:
        ckpt["extra"] = extra
    torch.save(ckpt, save_path)

def load_checkpoint(path, model, opt=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", float("inf"))
    return epoch, best_val, ckpt.get("extra", None)


def append_loss_csv(csv_path: str, epoch: int, train_loss: float, val_loss: float):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["epoch", "train_loss", "val_loss"])
        w.writerow([epoch, float(train_loss), float(val_loss)])


def plot_loss_curve(csv_path: str, out_png: str):
    epochs, tr, va = [], [], []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            # val_loss 可能是 nan
            try:
                va.append(float(row["val_loss"]))
            except:
                va.append(float("nan"))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure()
    plt.plot(epochs, tr, marker="o")
    plt.plot(epochs, va, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training/Validation Loss")
    plt.legend(["train", "val"])
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


# -------------------------
# Dataset: npz (paths) -> load images + tensors
# -------------------------
class TrajDatasetLatents(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True)
        self.z_past = self.data["z_past"].astype(np.float32)          # (B,L,N,Z)
        self.states_past = self.data["states_past"].astype(np.float32) # (B,L,N,6)
        self.states_future = self.data["states_future"].astype(np.float32) # (B,H,N,6)
        self.group_id = self.data["group_id"].astype(np.int32)         # (B,N)
        self.node_mask = self.data["node_mask"].astype(np.uint8)       # (B,N)

        self.L = self.states_past.shape[1]
        self.H = self.states_future.shape[1]
        self.Z = self.z_past.shape[-1]

        self.has_z_future = "z_future" in self.data.files
        if self.has_z_future:
            self.z_future = self.data["z_future"].astype(np.float32)   # (B,H,N,Z)

    def __len__(self):
        return self.states_past.shape[0]

    def __getitem__(self, idx):
        out = {
            "z_past": torch.from_numpy(self.z_past[idx]).float(),           # (L,N,Z)
            "states_past": torch.from_numpy(self.states_past[idx]).float(), # (L,N,6)
            "states_future": torch.from_numpy(self.states_future[idx]).float(), # (H,N,6)
            "group_id": torch.from_numpy(self.group_id[idx]).long(),        # (N,)
            "node_mask": torch.from_numpy(self.node_mask[idx]).bool(),      # (N,)
        }
        if self.has_z_future:
            out["z_future"] = torch.from_numpy(self.z_future[idx]).float()  # (H,N,Z)
        return out


# -------------------------
# Graph building: same-group kNN (k=2) from xy at t0
# -------------------------
@torch.no_grad()
def build_edges_knn_same_group(xy: torch.Tensor, group_id: torch.Tensor, mask: torch.Tensor, k: int = 2):
    """
    xy: (N,2), group_id: (N,), mask: (N,)
    Return directed edges (src->dst): (2,E)
    """
    device = xy.device
    N = xy.size(0)
    edges_src, edges_dst = [], []

    if mask.sum() <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    for g in torch.unique(group_id[mask]):
        nodes = torch.where(mask & (group_id == g))[0]
        m = nodes.numel()
        if m <= 1:
            continue
        pts = xy[nodes]                     # (m,2)
        dist = torch.cdist(pts, pts)        # (m,m)
        dist.fill_diagonal_(1e9)
        kk = min(k, m - 1)
        nn_idx = torch.topk(dist, k=kk, largest=False).indices  # (m,kk)

        for i in range(m):
            src = nodes[i].item()
            for j in nn_idx[i]:
                dst = nodes[j].item()
                edges_src.append(src)
                edges_dst.append(dst)

    if len(edges_src) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    return torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)

def edge_attr_from_state(s_last: torch.Tensor, edge_index: torch.Tensor):
    """
    s_last: (N,6) [x,y,vx,vy,sin,cos]
    edge_attr: (E,5) = [dx,dy,dvx,dvy,dist]
    """
    if edge_index.numel() == 0:
        return torch.empty(0, 5, dtype=s_last.dtype, device=s_last.device)
    src, dst = edge_index[0], edge_index[1]
    xy = s_last[:, 0:2]
    vv = s_last[:, 2:4]
    dxy = xy[src] - xy[dst]
    dv  = vv[src] - vv[dst]
    dist = torch.norm(dxy, dim=-1, keepdim=True)
    return torch.cat([dxy, dv, dist], dim=-1)

# -------------------------
# Encoders
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.proj(h)

class LatentEncoder(nn.Module):
    def __init__(self, z_dim=128, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, z_past):
        # z_past: (B,L,N,Z) -> (B,N,H)
        B, L, N, Z = z_past.shape
        x = z_past.permute(0, 2, 1, 3).contiguous().reshape(B*N, L, Z)
        _, h = self.gru(x)
        return h.squeeze(0).reshape(B, N, -1)

class StateEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, states_past):
        """
        states_past: (B,L,N,6) -> (B,N,hidden_dim)
        """
        B, L, N, D = states_past.shape
        x = states_past.permute(0, 2, 1, 3).contiguous().reshape(B * N, L, D)
        _, h = self.gru(x)
        return h.squeeze(0).reshape(B, N, -1)

# -------------------------
# ODEFunc (GNN drift)
# -------------------------
class GNNODEFunc(nn.Module):
    """
    Implements dh/dt = f(h, graph).
    graph is fixed during rollout: edge_index, edge_attr.
    """
    def __init__(self, h_dim=256, edge_dim=5, msg_dim=256):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * h_dim + edge_dim, msg_dim),
            nn.ReLU(),
            nn.Linear(msg_dim, msg_dim),
            nn.ReLU(),
        )
        self.upd = nn.Sequential(
            nn.Linear(h_dim + msg_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

        # set per-forward
        self.edge_index = None
        self.edge_attr = None
        self.node_mask = None

    def set_graph(self, edge_index, edge_attr, node_mask):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_mask = node_mask  # (N,) bool

    def forward(self, t, h):
        """
        t: scalar (ignored)
        h: (N, h_dim)
        return dh/dt: (N, h_dim)
        """
        if (self.edge_index is None) or (self.edge_index.numel() == 0):
            dh = torch.zeros_like(h)
        else:
            src, dst = self.edge_index[0], self.edge_index[1]
            hs = h[src]
            hd = h[dst]
            m = self.msg(torch.cat([hd, hs, self.edge_attr], dim=-1))
            agg = torch.zeros(h.size(0), m.size(-1), device=h.device, dtype=h.dtype)
            agg.index_add_(0, dst, m)
            dh = self.upd(torch.cat([h, agg], dim=-1))

        if self.node_mask is not None:
            dh = dh * self.node_mask[:, None].float()  # no drift for padded nodes
        return dh

# -------------------------
# Decoder: h(t) -> state
# -------------------------
class StateDecoder(nn.Module):
    def __init__(self, h_dim=256, out_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, h_seq):
        """
        h_seq: (T,N,h_dim) -> (T,N,6)
        """
        T, N, D = h_seq.shape
        return self.net(h_seq.reshape(T * N, D)).reshape(T, N, -1)

# -------------------------
# Full model: encode -> odeint -> decode
# -------------------------
class TrajGODEModel(nn.Module):
    def __init__(self, z_dim=128, z_enc_dim=128, st_dim=128, h_dim=256, state_dim=6):
        super().__init__()
        self.z_enc = LatentEncoder(z_dim=z_dim, hidden_dim=z_enc_dim)
        self.st_enc = StateEncoder(in_dim=state_dim, hidden_dim=st_dim)

        self.fuse = nn.Sequential(
            nn.Linear(z_enc_dim + st_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

        self.odefunc = GNNODEFunc(h_dim=h_dim, edge_dim=5, msg_dim=h_dim)

        self.dec_state = StateDecoder(h_dim=h_dim, out_dim=state_dim)
        self.dec_z = nn.Sequential(           # ✅ 新增：预测未来视觉latent
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
        )

    def forward(self, z_past, states_past, group_id, node_mask, H, dt=0.1, k=2, solver="dopri5"):
        B, L, N, _ = states_past.shape
        hz = self.z_enc(z_past)              # (B,N,z_enc_dim)
        hs = self.st_enc(states_past)        # (B,N,st_dim)
        h0 = self.fuse(torch.cat([hz, hs], dim=-1))  # (B,N,h_dim)
        
        h0 = h0 * node_mask[..., None].float()
        hz = hz * node_mask[..., None].float()
        hs = hs * node_mask[..., None].float()

        s_last = states_past[:, -1]  # (B,N,6)
        t_eval = torch.arange(1, H+1, device=states_past.device, dtype=states_past.dtype) * dt

        pred_state, pred_z = [], []
        for b in range(B):
            mask_b = node_mask[b]
            gid_b = group_id[b]
            xy = s_last[b, :, 0:2]
            edge_index = build_edges_knn_same_group(xy, gid_b, mask_b, k=k)
            edge_attr = edge_attr_from_state(s_last[b], edge_index)
            self.odefunc.set_graph(edge_index, edge_attr, mask_b)

            h_seq = odeint(self.odefunc, h0[b], t_eval, method=solver, rtol=1e-3, atol=1e-4)  # (H,N,h)

            y_state = self.dec_state(h_seq)                      # (H,N,6)
            y_state = y_state * mask_b[None,:,None].float()
            pred_state.append(y_state)

            y_z = self.dec_z(h_seq.reshape(-1, h_seq.size(-1))).reshape(H, N, -1)  # (H,N,Z)
            y_z = y_z * mask_b[None,:,None].float()
            pred_z.append(y_z)

        return torch.stack(pred_state, 0), torch.stack(pred_z, 0)  # (B,H,N,6), (B,H,N,Z)


    def trajectory(
        self,
        z_past: torch.Tensor,         # (1,L,N,Z)
        states_past: torch.Tensor,     # (1,L,N,6)
        group_id: torch.Tensor,        # (1,N)
        node_mask: torch.Tensor,       # (1,N) bool/0-1
        t_end: float,
        num_points: int,
        solver: str = "dopri5",
        return_z: bool = True,
    ):
        device = states_past.device
        self.eval()

        # 1) encode -> h0
        hz = self.z_enc(z_past)               # (1,N,z_enc_dim)
        hs = self.st_enc(states_past)         # (1,N,st_dim)

        # optional: mask out padded nodes early
        m = node_mask.float()
        hz = hz * m[..., None]
        hs = hs * m[..., None]

        h0 = self.fuse(torch.cat([hz, hs], dim=-1)).squeeze(0)  # (N,h_dim)

        # 2) build graph from last state
        s_last = states_past[:, -1].squeeze(0)   # (N,6)
        xy = s_last[:, 0:2]

        edge_index = build_edges_knn_same_group(
            xy,
            group_id.squeeze(0),
            node_mask.squeeze(0),
            k=2
        )
        edge_attr = edge_attr_from_state(s_last, edge_index)
        self.odefunc.set_graph(edge_index, edge_attr, node_mask.squeeze(0))

        # 3) time grid (allow negative t_end)
        t = torch.linspace(0.0, float(t_end), int(num_points), device=device, dtype=h0.dtype)

        # 4) integrate
        h_traj = odeint(self.odefunc, h0, t, method=solver, rtol=1e-3, atol=1e-4)  # (T,N,h)

        # 5) decode
        traj_state = self.dec_state(h_traj)  # (T,N,6)
        traj_state = traj_state * node_mask.squeeze(0)[None, :, None].float()

        if not return_z:
            return traj_state

        traj_z = self.dec_z(h_traj.reshape(-1, h_traj.size(-1))).reshape(t.size(0), h_traj.size(1), -1)  # (T,N,Z)
        traj_z = traj_z * node_mask.squeeze(0)[None, :, None].float()
        return (traj_state, traj_z)


# -------------------------
# Loss
# -------------------------
def loss_total(pred_state, gt_state, node_mask, pred_z=None, gt_z=None, lambda_z=0.01):
    # state loss（你已有的权重版也可以继续用）
    mask = node_mask[:, None, :, None].float()
    w = torch.tensor([2.0,2.0,1.0,1.0,0.5,0.5], device=gt_state.device)[None,None,None,:]
    L_state = ((pred_state - gt_state)**2 * w * mask).sum() / (mask.sum()*gt_state.size(-1) + 1e-6)

    if (pred_z is None) or (gt_z is None):
        return L_state, {"L_state": float(L_state.item())}

    L_z = ((pred_z - gt_z)**2 * mask).sum() / (mask.sum()*gt_z.size(-1) + 1e-6)
    L = L_state + lambda_z * L_z
    return L, {"L_state": float(L_state.item()), "L_z": float(L_z.item()), "lambda_z": lambda_z}


# -------------------------
# Train loop
# -------------------------
def train(
    npz_path,
    epochs=10,
    batch_size=4,
    lr=3e-4,
    device="cuda",
    solver="dopri5",
    vis_every=10,
    seed=42,
    train_ratio=0.5,
    val_ratio=0.1,
    ckpt_dir="checkpoints",
    resume_path=None,
    lambda_z=0.01,
):
    seed_everything(seed)
    os.makedirs(ckpt_dir, exist_ok=True)

    ds_all = TrajDatasetLatents(npz_path)
    train_ds, val_ds, test_ds = split_dataset(
        ds_all, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, drop_last=False)

    # z_dim 取 dataset 的
    model = TrajGODEModel(z_dim=ds_all.Z).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    H = ds_all.H
    dt = 0.1  # 你可以从数据里读，如果有保存 dt

    # 固定一批用于可视化（从 train 里取）
    vis_batch = next(iter(dl_train)) if len(train_ds) > 0 else None

    start_epoch = 1
    best_val = float("inf")

    # ===== resume =====
    if resume_path is not None and os.path.exists(resume_path):
        last_epoch, best_val0, _ = load_checkpoint(resume_path, model, opt, map_location=device)
        start_epoch = last_epoch + 1
        best_val = best_val0
        print(f"[Resume] {resume_path} | start_epoch={start_epoch} | best_val={best_val:.6f}")

    loss_csv = os.path.join(ckpt_dir, "loss.csv")

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for batch in dl_train:
            z_past = batch["z_past"].to(device)              # (B,L,N,Z)
            x = batch["states_past"].to(device)              # (B,L,N,6)
            y = batch["states_future"].to(device)            # (B,H,N,6)
            gid = batch["group_id"].to(device)               # (B,N)
            node_mask = batch["node_mask"].to(device)        # (B,N)

            # forward
            pred_state, pred_z = model(z_past, x, gid, node_mask, H=H, dt=dt, solver=solver)

            # loss
            if "z_future" in batch:
                gt_z = batch["z_future"].to(device)          # (B,H,N,Z)
                loss, log = loss_total(pred_state, y, node_mask, pred_z, gt_z, lambda_z=lambda_z)
            else:
                loss, log = loss_total(pred_state, y, node_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = z_past.size(0)
            total += float(loss.item()) * bs
            count += bs

        train_loss = total / max(count, 1)

        # val
        val_loss = evaluate(model, dl_val, device=device, H=H, dt=dt, solver=solver) \
            if len(val_ds) > 0 else float("nan")

        print(f"Epoch {ep}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        append_loss_csv(loss_csv, ep, train_loss, val_loss)

        # ===== 保存 last =====
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(last_path, model, opt, ep, best_val, extra={"npz": npz_path})

        # ===== 保存 best =====
        if len(val_ds) > 0 and (val_loss < best_val):
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, model, opt, ep, best_val, extra={"npz": npz_path})
            print(f"  [Best] val_loss -> {best_val:.6f} saved: {best_path}")

        # ===== 可视化 rollout（latent版）=====
        if vis_batch is not None and (ep % 1 == 0):
            model.eval()
            with torch.no_grad():
                z0   = vis_batch["z_past"][0:1].to(device)
                x0   = vis_batch["states_past"][0:1].to(device)
                y0   = vis_batch["states_future"][0:1].to(device)
                gid0 = vis_batch["group_id"][0:1].to(device)
                m0   = vis_batch["node_mask"][0:1].to(device)

                H_vis = y0.shape[1]

                traj_state, traj_z = model.trajectory(
                    z_past=z0,
                    states_past=x0,
                    group_id=gid0,
                    node_mask=m0,
                    t_end=H_vis * dt,
                    num_points=H_vis + 1,
                    solver=solver,
                    return_z=True
                )  # (H+1, N, 6)

                pred = traj_state[1:, :, :].unsqueeze(0).cpu().numpy()  # (1,H,N,6)
                gt   = y0.cpu().numpy()
                mask_np = m0.cpu().numpy()

                metrics = compare_rollout(
                    pred, gt, mask_np,
                    states_past=x0.cpu().numpy(),
                    save_dir=os.path.join(ckpt_dir, "rollout_vis"),
                    sample_idx=0,
                    max_vehicles=6,
                    prefix=f"ep{ep:03d}_"
                )
                print("Rollout metrics:", metrics)

    # ===== 最终 test（用 best）=====
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path) and len(test_ds) > 0:
        load_checkpoint(best_path, model, opt=None, map_location=device)
        test_loss = evaluate(model, dl_test, device=device, H=H, dt=dt, solver=solver)
        print(f"[Test] best.pt | test_loss={test_loss:.6f}")

    # loss curve
    loss_png = os.path.join(ckpt_dir, "loss_curve.png")
    if os.path.exists(loss_csv):
        plot_loss_curve(loss_csv, loss_png)
        print(f"[Saved] loss curve -> {loss_png}")

    return model



if __name__ == "__main__":
    npz_path = "/home/peh324/Codes/WorldModel/data/GODE/dataset_latents.npz"  # 改成你的npz路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = '/home/peh324/Codes/WorldModel/result/checkpoints/gode_vis'
    
    train(
        npz_path,
        epochs=100,
        batch_size=4,
        lr=3e-4,
        device=device,
        solver="dopri5",
        ckpt_dir=ckpt_dir,
        resume_path=None,   # 想断点续训就填 checkpoints_gode/last.pt
        seed=42
    )
