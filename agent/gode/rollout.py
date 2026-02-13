import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_gode import TrajGODEModel, load_checkpoint, TrajDatasetPaths, split_dataset

import csv
import matplotlib
matplotlib.use("Agg")   # 必须在 import pyplot 之前
import matplotlib.pyplot as plt

from PIL import Image

def load_episode_as_tensors(npz_path, img_size=(256,144)):
    data = np.load(npz_path, allow_pickle=True)
    paths = data["image_paths"]    # (T,N)
    states = torch.from_numpy(data["states"]).float()   # (T,N,6)
    group_id = torch.from_numpy(data["group_id"]).long()# (N,)
    node_mask_TN = torch.from_numpy(data["node_mask"]).bool()  # (T,N)

    T, N = paths.shape
    img_w, img_h = img_size
    images = torch.zeros(T, N, 3, img_h, img_w, dtype=torch.float32)

    for t in range(T):
        for i in range(N):
            p = str(paths[t, i])
            if (not node_mask_TN[t, i]) or (len(p) == 0) or (not os.path.exists(p)):
                continue
            img = Image.open(p).convert("RGB").resize((img_w, img_h))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            images[t, i] = torch.from_numpy(arr).permute(2,0,1)

    return images, states, group_id, node_mask_TN


def _make_window_from_episode(
    images_ep: torch.Tensor,   # (T,N,3,H,W)
    states_ep: torch.Tensor,   # (T,N,6)
    t_anchor: int,
    L: int
):
    """
    返回满足 model.trajectory 的输入窗口：
      images_past: (1,L,N,3,H,W)
      states_past: (1,L,N,6)
    规则：如果 t_anchor-L+1 < 0，则用最早帧 padding（重复第0帧）
    """
    T, N = states_ep.shape[0], states_ep.shape[1]
    t0 = t_anchor - L + 1
    if t0 < 0:
        pad = -t0
        # pad 用第0帧重复
        images_pad = images_ep[0:1].repeat(pad, 1, 1, 1, 1)  # (pad,N,3,H,W)
        states_pad = states_ep[0:1].repeat(pad, 1, 1)        # (pad,N,6)
        images_win = torch.cat([images_pad, images_ep[0:t_anchor+1]], dim=0)  # (L,N,3,H,W)
        states_win = torch.cat([states_pad, states_ep[0:t_anchor+1]], dim=0)  # (L,N,6)
    else:
        images_win = images_ep[t0:t_anchor+1]  # (L,N,3,H,W)
        states_win = states_ep[t0:t_anchor+1]  # (L,N,6)

    return images_win.unsqueeze(0), states_win.unsqueeze(0)


@torch.no_grad()
def compare_episode_bidirectional_rollout(
    model,
    images_ep,         # torch.Tensor (T,N,3,H,W)
    states_ep,         # torch.Tensor (T,N,6)
    group_id,          # torch.Tensor (N,)
    node_mask,         # torch.Tensor (N,) bool
    vehicle_idx: int,
    anchors,           # list[int]  e.g. [20, 60, 100]
    L: int,
    dt: float,
    forward_steps: int = 30,   # 从锚点向前推多少步
    backward_steps: int = 20,  # 从锚点向后推多少步
    solver: str = "dopri5",
    save_dir: str = "episode_vis",
    prefix: str = "",
    call_compare_rollout: bool = True,  # 是否复用已有 compare_rollout 做片段评估/画图
    max_segment_vehicles: int = 1,       # 片段图只画这一辆车（默认）
):
    """
    画整段 episode 中 vehicle_idx 的 GT 路径，并在多个 anchors 上做 forward/backward 推演叠加。

    - 整段GT：states_ep[:, vehicle_idx, 0:2]
    - 每个anchor：
        - 用过去 L 帧编码
        - forward：trajectory(t_end = forward_steps*dt)
        - backward：trajectory(t_end = -backward_steps*dt)  (负时间积分)
    - 可选：对每个 anchor 的 forward 片段，调用 compare_rollout(pred, gt, mask, states_past=...) 计算ADE/FDE并画片段图
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    # ====== move to device ======
    images_ep = images_ep.to(device)
    states_ep = states_ep.to(device)
    group_id = group_id.to(device)
    node_mask = node_mask.to(device)

    T, N = states_ep.shape[0], states_ep.shape[1]
    assert 0 <= vehicle_idx < N

    # ====== 整段 GT path ======
    gt_xy = states_ep[:, vehicle_idx, 0:2].detach().cpu().numpy()  # (T,2)

    # ====== 在一张图上叠加所有 anchor 的 pred ======
    # ====== 一张图：GT + 所有anchor rollout ======
    plt.figure(figsize=(8, 6))

    # ---- 整段GT ----
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], linewidth=3, color="black", label="GT full")

    # 起点终点
    plt.scatter(gt_xy[0, 0], gt_xy[0, 1], marker="o", s=60)
    plt.scatter(gt_xy[-1, 0], gt_xy[-1, 1], marker="x", s=60)
    
    all_metrics = []

    for a in anchors:
        if not (0 <= a < T):
            continue

        # ---- 4) anchor 点（GT） ----
        anchor_xy = states_ep[a, vehicle_idx, 0:2].detach().cpu().numpy()
        # ---- 5) 把预测轨迹画到同一张图上 ----
        plt.scatter(anchor_xy[0], anchor_xy[1], marker="s", label=f"anchor t={a}")
        
        # ---- 1) 取窗口作为编码输入 ----
        imgs_past, states_past = _make_window_from_episode(images_ep, states_ep, t_anchor=a, L=L)
        # shapes: (1,L,N,3,H,W), (1,L,N,6)
        # print(f"shape imgs_past {imgs_past.shape}, states_past {states_past.shape}, group_id {group_id.shape}, node_mask {node_mask.shape}")
        # ---- 2) forward rollout ----
        traj_f = model.trajectory(
            images_past=imgs_past,
            states_past=states_past,
            group_id=group_id.unsqueeze(0),
            node_mask=node_mask[a].unsqueeze(0),
            t_end=forward_steps * dt,
            num_points=forward_steps + 1,
            solver=solver,
        )  # (forward_steps+1, N, 6)

        pred_f_xy = traj_f[:, vehicle_idx, 0:2].detach().cpu().numpy()  # (Sf+1,2)

        # ---- 3) backward rollout（负时间） ----
        traj_b = model.trajectory(
            images_past=imgs_past,
            states_past=states_past,
            group_id=group_id.unsqueeze(0),
            node_mask=node_mask[a].unsqueeze(0),
            t_end=-backward_steps * dt,            # 关键：负数
            num_points=backward_steps + 1,
            solver=solver,
        )  # (Sb+1, N, 6)

        pred_b_xy = traj_b[:, vehicle_idx, 0:2].detach().cpu().numpy()  # (Sb+1,2)

        # backward 轨迹时间是从0到负数，点序是 [t=0, t=-dt, ...]
        # 为了在图上更直观，可以反转一下，让它看起来“从过去走到anchor”
        plt.plot(pred_b_xy[::-1, 0], pred_b_xy[::-1, 1], marker="x", alpha=0.8)

        # forward 轨迹从 anchor 往未来
        plt.plot(pred_f_xy[:, 0], pred_f_xy[:, 1], marker="*", alpha=0.8)

    plt.title(f"{prefix}Vehicle {vehicle_idx} | GT + rollout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    # plt.legend(loc="best")

    out_png = os.path.join(save_dir, f"{prefix}veh{vehicle_idx:03d}_rollout_full.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[Saved] rollout visualization -> {out_png}")

    # 把 metrics 写到文件方便你看
    if len(all_metrics) > 0:
        out_txt = os.path.join(save_dir, f"{prefix}veh{vehicle_idx:03d}_anchor_metrics.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for m in all_metrics:
                f.write(str(m) + "\n")

    return {
        "anchors_png": out_png,
        "anchor_metrics": all_metrics
    }


if __name__ == "__main__":
    print("This is a module for rollout visualization, not meant to be run directly.")
    # images_ep: (T,N,3,H,W)
    # states_ep: (T,N,6)
    # group_id:  (N,)
    # node_mask: (N,)
    ckpt_dir = '/home/peh324/Codes/WorldModel/result/checkpoints/bigode'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrajGODEModel()
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path) > 0:
        load_checkpoint(best_path, model, opt=None, map_location=device)

    npz_path = "/home/peh324/Codes/WorldModel/data/GODE/episode_paths.npz"
    img_size = (256, 144)
    images_ep, states_ep, group_id, node_mask = load_episode_as_tensors(npz_path, img_size=img_size)

    anchors = range(20, 560, 20)  # 你可以根据 T 和兴趣点调整这些锚点
    
    L = 20
    forward_steps = 30
    for vechiel_idx in range(6):
        res = compare_episode_bidirectional_rollout(
            model=model,
            images_ep=images_ep,
            states_ep=states_ep,
            group_id=group_id,
            node_mask=node_mask,
            vehicle_idx=vechiel_idx,
            anchors=anchors,  # 你可以根据 T 和兴趣点调整这些锚点
            L=L,
            dt=0.1,
            forward_steps=forward_steps,
            backward_steps=0,
            solver="dopri5",
            save_dir="episode_vis_bi",
            prefix=f"ep001_L{L}_For{forward_steps}_",
            call_compare_rollout=True
        )
        print(res)
