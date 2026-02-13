import os
import numpy as np
import matplotlib.pyplot as plt

import torch

@torch.no_grad()
def rollout_one_batch(model, batch, device, solver="dopri5", k=2, dt=1.0):
    """
    输入:
      model: TrajGODEModel
      batch: dataloader batch dict
    输出:
      pred: (B,H,N,6)
      gt:   (B,H,N,6)
      mask: (B,N) bool
      meta: dict (可选)
    """
    model.eval()

    imgs = batch["images_past"].to(device)          # (B,L,N,3,H,W)
    x = batch["states_past"].to(device)             # (B,L,N,6)
    y = batch["states_future"].to(device)           # (B,H,N,6)
    gid = batch["group_id"].to(device)              # (B,N)
    mask = batch["node_mask"].to(device)            # (B,N)

    H = y.shape[1]
    pred = model(
        imgs, x, gid, mask,
        H=H, dt=dt, k=k,
        solver=solver
    )  # (B,H,N,6)

    return pred.cpu(), y.cpu(), mask.cpu(), {
        "group_id": gid.cpu(),
        "states_past": x.cpu(),
    }


def compare_rollout(pred, gt, mask, states_past=None, save_dir="rollout_vis", sample_idx=0,
                    vehicle_indices=None, max_vehicles=10, prefix=""):
    """
    pred, gt: (B,H,N,6)
    mask: (B,N) bool
    states_past: (B,L,N,6) 可选，用来画历史轨迹
    sample_idx: 选择画 batch 里的第几个样本
    vehicle_indices: 指定要画的车辆 index list（基于 actor 顺序）
    max_vehicles: 不指定时最多画多少辆
    """
    os.makedirs(save_dir, exist_ok=True)

    pred = np.asarray(pred)
    gt = np.asarray(gt)
    mask = np.asarray(mask).astype(bool)
    # print(f"shape pred {pred.shape}, gt {gt.shape}, mask {mask.shape}")
    H = pred.shape[1]
    N = pred.shape[2]

    # 取一个样本
    p = pred[sample_idx]   # (H,N,6)
    g = gt[sample_idx]     # (H,N,6)
    m = mask[sample_idx]   # (N,)

    # 指标：只在有效节点上算
    # 位置误差
    p_xy = p[:, :, 0:2]   # (H,N,2)
    g_xy = g[:, :, 0:2]
    disp = np.linalg.norm(p_xy - g_xy, axis=-1)  # (H,N)
    ade = disp[:, m].mean() if m.any() else np.nan
    fde = disp[-1, m].mean() if m.any() else np.nan

    # 速度误差
    p_v = p[:, :, 2:4]
    g_v = g[:, :, 2:4]
    verr = np.linalg.norm(p_v - g_v, axis=-1)    # (H,N)
    v_ade = verr[:, m].mean() if m.any() else np.nan
    v_fde = verr[-1, m].mean() if m.any() else np.nan

    metrics = {"ADE": float(ade), "FDE": float(fde), "V_ADE": float(v_ade), "V_FDE": float(v_fde)}

    # 选择要画的车
    valid_ids = np.where(m)[0].tolist()
    if vehicle_indices is None:
        vehicle_indices = valid_ids[:max_vehicles]
    else:
        vehicle_indices = [i for i in vehicle_indices if (0 <= i < N) and m[i]]

    # 历史轨迹（可选）
    hist_xy = None
    if states_past is not None:
        xpast = np.asarray(states_past)[sample_idx]  # (L,N,6)
        hist_xy = xpast[:, :, 0:2]                   # (L,N,2)

    # 每辆车一张图
    for i in vehicle_indices:
        plt.figure()
        if hist_xy is not None:
            plt.plot(hist_xy[:, i, 0], hist_xy[:, i, 1], marker="o")  # history

        plt.plot(g_xy[:, i, 0], g_xy[:, i, 1], marker="o")           # gt future
        plt.plot(p_xy[:, i, 0], p_xy[:, i, 1], marker="o")           # pred future

        plt.title(f"{prefix}sample={sample_idx} vehicle_idx={i}  ADE={ade:.3f} FDE={fde:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["history", "gt_future", "pred_future"] if hist_xy is not None else ["gt_future", "pred_future"])
        plt.axis("equal")

        out_path = os.path.join(save_dir, f"{prefix}sample{sample_idx:03d}_veh{i:03d}.png")
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()

    # 还可以保存一个全车汇总图（可选）
    plt.figure(figsize=(7, 5))
    for i in vehicle_indices:
        plt.plot(g_xy[:, i, 0], g_xy[:, i, 1], alpha=0.8)
        plt.plot(p_xy[:, i, 0], p_xy[:, i, 1], alpha=0.8)
    plt.title(f"{prefix}sample={sample_idx}  ADE={ade:.3f} FDE={fde:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    out_path = os.path.join(save_dir, f"{prefix}sample{sample_idx:03d}_summary.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

    return metrics

