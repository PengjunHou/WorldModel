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
def evaluate(model, dl, device, H, dt, solver="dopri5"):
    model.eval()
    total = 0.0
    count = 0
    for batch in dl:
        imgs = batch["images_past"].to(device)
        x = batch["states_past"].to(device)
        y = batch["states_future"].to(device)
        gid = batch["group_id"].to(device)
        node_mask = batch["node_mask"].to(device)

        y_hat = model(imgs, x, gid, node_mask, H=H, dt=dt, k=2, solver=solver)
        loss = traj_loss(y_hat, y, node_mask)

        bs = imgs.size(0)
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
class TrajDatasetPaths(Dataset):
    def __init__(self, npz_path, img_size=(256, 144)):
        self.data = np.load(npz_path, allow_pickle=True)
        self.image_paths_past = self.data["image_paths_past"]   # (B,L,N) unicode
        self.states_past = self.data["states_past"].astype(np.float32)         # (B,L,N,6)
        self.states_future = self.data["states_future"].astype(np.float32)     # (B,H,N,6)
        self.group_id = self.data["group_id"].astype(np.int32)                # (B,N)
        self.node_mask = self.data["node_mask"].astype(np.uint8)              # (B,N)
        self.img_size = img_size

        self.L = int(self.data["L"][0]) if "L" in self.data.files else self.states_past.shape[1]
        self.H = int(self.data["H"][0]) if "H" in self.data.files else self.states_future.shape[1]

    def __len__(self):
        return self.states_past.shape[0]

    def _load_rgb(self, path: str):
        img_w, img_h = self.img_size  # 注意：PIL resize 是 (W,H)

        if (path is None) or (len(path) == 0) or (not isinstance(path, str)) or (not os.path.exists(path)):
            return torch.zeros(3, img_h, img_w, dtype=torch.float32)  # (3,H,W)

        img = Image.open(path).convert("RGB").resize((img_w, img_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        ten = torch.from_numpy(arr).permute(2, 0, 1)     # (3,H,W)
        return ten

    def __getitem__(self, idx):
        x = torch.from_numpy(self.states_past[idx]).float()       # (L,N,6)
        y = torch.from_numpy(self.states_future[idx]).float()     # (H,N,6)
        gid = torch.from_numpy(self.group_id[idx]).long()         # (N,)
        mask = torch.from_numpy(self.node_mask[idx]).bool()       # (N,)
        
        paths = self.image_paths_past[idx]       
        L, N = paths.shape

        img_w, img_h = self.img_size
        imgs = torch.zeros(L, N, 3, img_h, img_w, dtype=torch.float32)  # (L,N,3,H,W)

        for t in range(L):
            for i in range(N):
                if mask[i]:
                    imgs[t, i] = self._load_rgb(str(paths[t, i]))

        return {
            "images_past": imgs,     # (L,N,3,H,W)
            "states_past": x,        # (L,N,6)
            "states_future": y,      # (H,N,6)
            "group_id": gid,         # (N,)
            "node_mask": mask,       # (N,)
        }

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

class CameraEncoder(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=128):
        super().__init__()
        self.cnn = SmallCNN(out_dim=feat_dim)
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, images_past):
        """
        images_past: (B,L,N,3,H,W) -> (B,N,hidden_dim)
        """
        B, L, N, C, H, W = images_past.shape
        x = images_past.reshape(B * L * N, C, H, W)
        f = self.cnn(x).reshape(B, L, N, -1)      # (B,L,N,feat)
        f = f.permute(0, 2, 1, 3).contiguous()    # (B,N,L,feat)
        f = f.reshape(B * N, L, -1)
        _, h = self.gru(f)
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
    def __init__(self, cam_dim=128, st_dim=128, h_dim=256, state_dim=6):
        super().__init__()
        self.cam_enc = CameraEncoder(feat_dim=cam_dim, hidden_dim=cam_dim)
        self.st_enc = StateEncoder(in_dim=state_dim, hidden_dim=st_dim)

        self.fuse = nn.Sequential(
            nn.Linear(cam_dim + st_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
        )

        self.odefunc = GNNODEFunc(h_dim=h_dim, edge_dim=5, msg_dim=h_dim)
        self.dec = StateDecoder(h_dim=h_dim, out_dim=state_dim)

    def forward(self, images_past, states_past, group_id, node_mask, H: int, dt: float = 1.0, k: int = 2,
                solver="dopri5", rtol=1e-3, atol=1e-4):
        """
        images_past: (B,L,N,3,H,W)
        states_past: (B,L,N,6)
        group_id:    (B,N)
        node_mask:   (B,N) bool
        Returns y_hat: (B,H,N,6)
        """
        B, L, N, D = states_past.shape
        z_cam = self.cam_enc(images_past)                # (B,N,cam_dim)
        z_st  = self.st_enc(states_past)                 # (B,N,st_dim)
        h0 = self.fuse(torch.cat([z_cam, z_st], dim=-1)) # (B,N,h_dim)

        s_last = states_past[:, -1]  # (B,N,6)

        # Integrate at times: t1..tH (exclude t0 to align with future steps)
        # odeint returns values at each time in t_eval
        t_eval = torch.arange(1, H + 1, device=states_past.device, dtype=states_past.dtype) * dt  # (H,)

        preds = []
        for b in range(B):
            mask_b = node_mask[b]
            gid_b = group_id[b]
            xy = s_last[b, :, 0:2]

            edge_index = build_edges_knn_same_group(xy, gid_b, mask_b, k=k)
            edge_attr = edge_attr_from_state(s_last[b], edge_index)

            self.odefunc.set_graph(edge_index, edge_attr, mask_b)

            # odeint expects (N,h_dim) and returns (T,N,h_dim)
            h_seq = odeint(self.odefunc, h0[b], t_eval, method=solver, rtol=rtol, atol=atol)  # (H,N,h_dim)
            y_hat = self.dec(h_seq)  # (H,N,6)
            y_hat = y_hat * mask_b[None, :, None].float()
            preds.append(y_hat)

        return torch.stack(preds, dim=0)  # (B,H,N,6)

    def trajectory(self, images_past: torch.Tensor, states_past: torch.Tensor, group_id: torch.Tensor,
                node_mask: torch.Tensor, t_end: float, num_points: int, solver: str = "dopri5"):
        """
        从当前状态推演到任意时间点

        参数
        ----
        images_past: (1,L,N,3,H,W)
        states_past: (1,L,N,6)
        group_id:    (1,N)
        node_mask:   (1,N)
        t_end:       推演到多少秒（例如5.0）
        num_points:  采样多少个时间点（例如50）
        solver:      ode求解器

        返回
        ----
        traj: (T,N,6)  每个时间点的预测状态
        """

        device = states_past.device
        self.eval()

        # ========= 1. encode得到初始h0 =========
        z_cam = self.cam_enc(images_past)                 # (1,N,cam_dim)
        z_st  = self.st_enc(states_past)                  # (1,N,st_dim)
        h0 = self.fuse(torch.cat([z_cam, z_st], dim=-1))  # (1,N,h_dim)
        h0 = h0.squeeze(0)                                # (N,h_dim)

        # ========= 2. 构图（用最后一帧state） =========
        s_last = states_past[:, -1].squeeze(0)            # (N,6)
        xy = s_last[:, 0:2]

        edge_index = build_edges_knn_same_group( xy, group_id.squeeze(0), node_mask.squeeze(0), k=2)

        edge_attr = edge_attr_from_state(s_last, edge_index)

        self.odefunc.set_graph(edge_index, edge_attr, node_mask.squeeze(0))

        # ========= 3. 构建时间轴 =========
        # 从0开始到t_end
        integration_time = torch.linspace(0, t_end, num_points, device=device, dtype=h0.dtype)

        # ========= 4. ODE推演 =========
        h_traj = odeint(self.odefunc, h0, integration_time, method=solver, rtol=1e-3, atol=1e-4)  # (T,N,h_dim)

        # ========= 5. decode成state =========
        traj = self.dec(h_traj)   # (T,N,6)

        return traj


# -------------------------
# Loss
# -------------------------
def traj_loss(y_hat, y_true, node_mask):
    """
    y_hat, y_true: (B,H,N,6)
    node_mask: (B,N) bool
    """
    mask = node_mask[:, None, :, None].float()
    # position more important
    w = torch.tensor([2.0, 2.0, 1.0, 1.0, 0.5, 0.5], device=y_hat.device)[None, None, None, :]
    loss = ((y_hat - y_true) ** 2 * w * mask).sum() / (mask.sum() * y_hat.size(-1) + 1e-6)
    return loss

# -------------------------
# Train loop
# -------------------------
def train(npz_path, epochs=10, batch_size=4, lr=3e-4, device="cuda",
          img_size=(256,144), solver="dopri5", vis_every=10,
          seed=42,
          train_ratio=0.8, val_ratio=0.1,
          ckpt_dir="checkpoints",
          resume_path=None):

    seed_everything(seed)

    ds_all = TrajDatasetPaths(npz_path, img_size=img_size)
    train_ds, val_ds, test_ds = split_dataset(ds_all, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model = TrajGODEModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    H = ds_all.H
    dt = 0.1

    # 固定一批用于可视化（从 train 里取）
    vis_batch = next(iter(dl_train)) if len(train_ds) > 0 else None

    start_epoch = 1
    best_val = float("inf")

    # ===== resume =====
    if resume_path is not None and os.path.exists(resume_path):
        last_epoch, best_val0, _ = load_checkpoint(resume_path, model, opt, map_location=device)
        start_epoch = last_epoch + 1
        best_val = best_val0
        print(f"[Resume] from {resume_path} | start_epoch={start_epoch} | best_val={best_val:.6f}")

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for batch in dl_train:
            imgs = batch["images_past"].to(device)
            x = batch["states_past"].to(device)
            y = batch["states_future"].to(device)
            gid = batch["group_id"].to(device)
            node_mask = batch["node_mask"].to(device)

            y_hat = model(imgs, x, gid, node_mask, H=H, dt=dt, k=2, solver=solver)
            loss = traj_loss(y_hat, y, node_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = imgs.size(0)
            total += float(loss.item()) * bs
            count += bs

        train_loss = total / max(count, 1)
        val_loss = evaluate(model, dl_val, device=device, H=H, dt=dt, solver=solver) if len(val_ds) > 0 else float("nan")

        print(f"Epoch {ep}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        
        loss_csv = os.path.join(ckpt_dir, "loss.csv")
        append_loss_csv(loss_csv, ep, train_loss, val_loss)


        # ===== 保存 last =====
        last_path = os.path.join(ckpt_dir, "last.pt")
        save_checkpoint(# The code `last_path` is not doing anything in the provided snippet. It seems
        # to be a variable name or placeholder that is not being used or assigned any
        # value.
        last_path, model, opt, ep, best_val, extra={"npz": npz_path})

        # ===== 保存 best =====
        if len(val_ds) > 0 and val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, model, opt, ep, best_val, extra={"npz": npz_path})
            print(f"  [Best] val_loss improved -> {best_val:.6f}  saved to {best_path}")

        # ===== 可视化（可选） =====
        if vis_batch is not None and (ep % vis_every == 0):
            model.eval()
            with torch.no_grad():
                imgs0 = vis_batch["images_past"][0:1].to(device)
                x0    = vis_batch["states_past"][0:1].to(device)
                y0    = vis_batch["states_future"][0:1].to(device)
                gid0  = vis_batch["group_id"][0:1].to(device)
                m0    = vis_batch["node_mask"][0:1].to(device)

                H_vis = y0.shape[1]
                traj = model.trajectory(
                    images_past=imgs0, states_past=x0,
                    group_id=gid0, node_mask=m0,
                    t_end=H_vis * dt, num_points=H_vis + 1,
                    solver=solver,
                )

                pred = traj[1:, :, :].unsqueeze(0).cpu().numpy()
                gt   = y0.cpu().numpy()
                mask = m0.cpu().numpy()

                metrics = compare_rollout(
                    pred, gt, mask,
                    states_past=x0.cpu().numpy(),
                    save_dir="rollout_vis",
                    sample_idx=0,
                    max_vehicles=6,
                    prefix=f"ep{ep:03d}_traj_"
                )
                print("Trajectory rollout metrics:", metrics)

    # ===== 最终 test（用 best）=====
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path) and len(test_ds) > 0:
        load_checkpoint(best_path, model, opt=None, map_location=device)
        test_loss = evaluate(model, dl_test, device=device, H=H, dt=dt, solver=solver)
        print(f"[Test] best.pt | test_loss={test_loss:.6f}")
        
    loss_csv = os.path.join(ckpt_dir, "loss.csv")
    loss_png = os.path.join(ckpt_dir, "loss_curve.png")
    if os.path.exists(loss_csv):
        plot_loss_curve(loss_csv, loss_png)
        print(f"[Saved] loss curve -> {loss_png}")

    return model


def train2(npz_path, epochs=10, batch_size=4, lr=3e-4, device="cuda",
          img_size=(256,144), solver="dopri5", vis_every=1):

    ds = TrajDatasetPaths(npz_path, img_size=img_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = TrajGODEModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    H = ds.H
    dt = 0.1

    # 固定一批用于可视化（只取一次）
    vis_batch = next(iter(dl))

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0

        for batch in dl:
            imgs = batch["images_past"].to(device)          # (B,L,N,3,H,W)
            x = batch["states_past"].to(device)             # (B,L,N,6)
            y = batch["states_future"].to(device)           # (B,H,N,6)
            gid = batch["group_id"].to(device)              # (B,N)
            node_mask = batch["node_mask"].to(device)       # (B,N)

            y_hat = model(imgs, x, gid, node_mask, H=H, dt=dt, k=2, solver=solver)
            loss = traj_loss(y_hat, y, node_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())

        print(f"Epoch {ep}/{epochs} | loss={total/len(dl):.6f}")

        # ====== 用 trajectory 可视化 ======
        if ep % vis_every == 0:
            model.eval()
            with torch.no_grad():
                # 取vis_batch里第0条样本
                imgs0 = vis_batch["images_past"][0:1].to(device)
                x0    = vis_batch["states_past"][0:1].to(device)
                y0    = vis_batch["states_future"][0:1].to(device)
                gid0  = vis_batch["group_id"][0:1].to(device)
                m0    = vis_batch["node_mask"][0:1].to(device)

                H_vis = y0.shape[1]
                # 推演出 [0, dt, 2dt, ..., H*dt] 共 H+1 个点
                traj = model.trajectory(
                    images_past=imgs0,
                    states_past=x0,
                    group_id=gid0,
                    node_mask=m0,
                    t_end=H_vis * dt,
                    num_points=H_vis + 1,
                    solver=solver,
                )  # (H+1,N,6)

                # 对齐未来标签：去掉t=0
                pred = traj[1:, :, :].unsqueeze(0).cpu().numpy()   # (1,H,N,6)
                gt   = y0.cpu().numpy()
                mask = m0.cpu().numpy()

                metrics = compare_rollout(
                    pred, gt, mask,
                    states_past=x0.cpu().numpy(),
                    save_dir="rollout_vis",
                    sample_idx=0,
                    max_vehicles=6,
                    prefix=f"ep{ep:03d}_traj_"
                )
                print("Trajectory rollout metrics:", metrics)

    return model


if __name__ == "__main__":
    npz_path = "/home/peh324/Codes/WorldModel/data/GODE/dataset_paths.npz"  # 改成你的npz路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = '/home/peh324/Codes/WorldModel/result/checkpoints/gode'
    
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
