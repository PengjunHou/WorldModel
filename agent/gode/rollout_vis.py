import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from train_vis_gode import TrajGODEModel, load_checkpoint
from extract.ae_vis import load_ae_and_detector, encode_img_paths_to_z, visualize_predz_vs_gtimage_detection
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
    model: TrajGODEModel,
    images_ep,         # torch.Tensor (T,N,3,H,W) 这里暂时保留，不强依赖
    states_ep,         # torch.Tensor (T,N,6)
    group_id,          # torch.Tensor (N,)
    node_mask,         # torch.Tensor (T,N) or (N,)
    vehicle_idx: int,
    anchors,           # iterable[int]
    L: int,
    dt: float,
    forward_steps: int = 30,
    backward_steps: int = 0,   # 默认别用 backward
    solver: str = "dopri5",
    save_dir: str = "episode_vis",
    prefix: str = "",

    # recon/det
    ae_ckpt: str = None,
    z_dim: int = 128,
    recon_offsets=(1, 5, 10, 20),
    detector_prefer="yolo",
    img_size=(256,144),

    # must-have for recon + making z_past
    image_paths_ep=None,   # np.ndarray (T,N) str
):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()

    # ===== AE / detector =====
    ae, det = None, None
    if ae_ckpt is not None:
        ae, det = load_ae_and_detector(
            ae_ckpt=ae_ckpt,
            z_dim=z_dim,
            img_size=img_size,
            device=device,
            detector_prefer=detector_prefer,
        )

    # ===== move tensors =====
    # images_ep 可以不用搬上GPU（很大），但你代码里传进来就先保持行为一致
    # 如果你想更省显存，可以注释掉 images_ep.to(device) 这一行
    images_ep = images_ep.to(device) if isinstance(images_ep, torch.Tensor) else images_ep
    states_ep = states_ep.to(device)
    group_id = group_id.to(device)
    node_mask = node_mask.to(device)

    T, N = states_ep.shape[0], states_ep.shape[1]
    assert 0 <= vehicle_idx < N
    assert image_paths_ep is not None, "必须传 image_paths_ep (T,N)，用于 AE 编码生成 z_past"
    assert ae is not None, "latent rollout 需要 AE：请传 ae_ckpt"

    # node_mask could be (T,N) or (N,)
    def _mask_at(t_idx: int):
        return node_mask[t_idx] if node_mask.ndim == 2 else node_mask  # (N,)

    # ===== plot full GT =====
    gt_xy = states_ep[:, vehicle_idx, 0:2].detach().cpu().numpy()  # (T,2)
    plt.figure(figsize=(8, 6))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], linewidth=3, color="black", label="GT full")
    plt.scatter(gt_xy[0, 0], gt_xy[0, 1], marker="o", s=60)
    plt.scatter(gt_xy[-1, 0], gt_xy[-1, 1], marker="x", s=60)

    recon_logs = []

    for a in anchors:
        a = int(a)
        if not (0 <= a < T):
            continue

        mask_a = _mask_at(a)  # (N,)
        if mask_a.sum().item() <= 0:
            continue

        # ---- anchor GT point ----
        anchor_xy = states_ep[a, vehicle_idx, 0:2].detach().cpu().numpy()
        plt.scatter(anchor_xy[0], anchor_xy[1], marker="s", s=40, alpha=0.9)

        # ===== states_past window (1,L,N,6) =====
        # 复用你已有的 _make_window_from_episode，但它要求 images_ep。
        # 我们给一个“最小dummy images”以避免占用大显存；函数只为了 padding 逻辑。
        dummy_images = torch.zeros((T, N, 3, 1, 1), device=device, dtype=states_ep.dtype)
        _, states_past = _make_window_from_episode(
            images_ep=dummy_images,
            states_ep=states_ep,
            t_anchor=a,
            L=L
        )  # (1,L,N,6)

        # ===== image_paths window (L,N) =====
        t0 = a - L + 1
        if t0 < 0:
            pad = -t0
            paths_pad = np.repeat(image_paths_ep[0:1], pad, axis=0)          # (pad,N)
            paths_win = np.concatenate([paths_pad, image_paths_ep[0:a+1]], axis=0)  # (L,N)
        else:
            paths_win = image_paths_ep[t0:a+1]  # (L,N)

        # ===== z_past: (1,L,N,Z) =====
        z_win = encode_img_paths_to_z(
            ae=ae,
            paths_2d=paths_win,   # (L,N)
            img_size=img_size,
            device=device,
        ).to(device)                         # -> (L,N,Z) torch
        z_past = z_win.unsqueeze(0)  # (1,L,N,Z)

        # ===== forward rollout =====
        out_f = model.trajectory(
            z_past=z_past,
            states_past=states_past,
            group_id=group_id.unsqueeze(0),
            node_mask=mask_a.unsqueeze(0),
            t_end=forward_steps * dt,
            num_points=forward_steps + 1,
            solver=solver,
            return_z=True
        )

        traj_f, traj_f_z = out_f if isinstance(out_f, (tuple, list)) and len(out_f) >= 2 else (out_f, None)
        pred_f_xy = traj_f[:, vehicle_idx, 0:2].detach().cpu().numpy()
        plt.plot(pred_f_xy[:, 0], pred_f_xy[:, 1], marker="*", alpha=0.8)

        # ===== optional backward (不推荐，但保留) =====
        if backward_steps and backward_steps > 0:
            out_b = model.trajectory(
                z_past=z_past,
                states_past=states_past,
                group_id=group_id.unsqueeze(0),
                node_mask=mask_a.unsqueeze(0),
                t_end=-backward_steps * dt,
                num_points=backward_steps + 1,
                solver=solver,
            )
            traj_b = out_b[0] if isinstance(out_b, (tuple, list)) else out_b
            pred_b_xy = traj_b[:, vehicle_idx, 0:2].detach().cpu().numpy()
            plt.plot(pred_b_xy[::-1, 0], pred_b_xy[::-1, 1], marker="x", alpha=0.8)

        # ===== recon/det at selected offsets =====
        if traj_f_z is not None:
            for off in recon_offsets:
                off = int(off)
                if off < 0 or off >= traj_f_z.shape[0]:
                    continue
                t_gt = a + off
                if t_gt < 0 or t_gt >= T:
                    continue
                if not bool(mask_a[vehicle_idx].item()):
                    continue

                gt_path = "/home/peh324/Codes/WorldModel/" + str(image_paths_ep[t_gt, vehicle_idx])
                if (len(gt_path) == 0) or (not os.path.exists(gt_path)):
                    continue

                pred_z = traj_f_z[off, vehicle_idx]  # (Z,)
                out_png = os.path.join(
                    save_dir,
                    f"{prefix}veh{vehicle_idx:03d}_a{a:04d}_off{off:02d}_recon_det.png"
                )

                summary = visualize_predz_vs_gtimage_detection(
                    ae=ae,
                    detector=det,
                    gt_image_path=gt_path,
                    pred_z=pred_z,
                    out_png=out_png,
                    img_size=img_size,
                )

                recon_logs.append({
                    "anchor": int(a),
                    "offset": int(off),
                    "t_gt": int(t_gt),
                    "png": out_png,
                    **(summary if isinstance(summary, dict) else {"summary": str(summary)}),
                })

    plt.title(f"{prefix}Vehicle {vehicle_idx} | GT + rollout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    out_png = os.path.join(save_dir, f"{prefix}veh{vehicle_idx:03d}_rollout_full.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[Saved] rollout visualization -> {out_png}")

    if len(recon_logs) > 0:
        out_txt = os.path.join(save_dir, f"{prefix}veh{vehicle_idx:03d}_recon_det_logs.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for r in recon_logs:
                f.write(str(r) + "\n")
        print(f"[Saved] recon/det logs -> {out_txt}")

    return {"anchors_png": out_png, "recon_det": recon_logs}


if __name__ == "__main__":
    # 现在这个脚本就是要跑，所以别再写 not meant to be run
    print("[rollout] running episode rollout visualization...")

    ckpt_dir = "/home/peh324/Codes/WorldModel/result/checkpoints/gode_vis"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TrajGODEModel().to(device)
    best_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, opt=None, map_location=device)
        model.eval()
        print(f"[OK] loaded ckpt: {best_path}")
    else:
        raise FileNotFoundError(best_path)

    # ✅ 用“episode”文件，而不是 dataset_latents.npz
    ep_npz_path = "/home/peh324/Codes/WorldModel/data/GODE/episode_paths.npz"
    img_size = (256, 144)

    # 读出 episode 的图像tensor + states + mask
    images_ep, states_ep, group_id, node_mask_TN = load_episode_as_tensors(ep_npz_path, img_size=img_size)

    # ✅ 同时取出 image_paths_ep (T,N)，用于 AE 编码/重构/检测可视化
    data_ep = np.load(ep_npz_path, allow_pickle=True)
    image_paths_ep = data_ep["image_paths"]  # (T,N) str

    T = states_ep.shape[0]
    anchors = list(range(20, min(T - 1, 560), 20))  # 防越界

    L = 20
    forward_steps = 30

    ae_ckpt = "/home/peh324/Codes/WorldModel/result/checkpoints/ae/ae_best.pt"  # 你自己的AE路径

    for vehicle_idx in range(min(6, states_ep.shape[1])):
        res = compare_episode_bidirectional_rollout(
            model=model,
            images_ep=images_ep,          # 你签名里还有这个参数，就照传
            states_ep=states_ep,
            group_id=group_id,
            node_mask=node_mask_TN,       # (T,N) 也可以
            vehicle_idx=vehicle_idx,
            anchors=anchors,
            L=L,
            dt=0.1,
            forward_steps=forward_steps,
            backward_steps=0,
            solver="dopri5",
            save_dir="episode_vis_bi",
            prefix=f"ep001_L{L}_For{forward_steps}_veh{vehicle_idx:02d}_",

            # ✅ recon/det 相关
            ae_ckpt=ae_ckpt,
            z_dim=128,
            recon_offsets=(1, 5, 10, 20),
            detector_prefer="yolo",
            img_size=img_size,
            image_paths_ep=image_paths_ep,   # ✅ 必传
        )
        print(res)

