#!/usr/bin/env python3
# prepare_split_npz_dataset.py
import argparse, random
from pathlib import Path
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

# ---------- 1. 全局参数 ---------- #
IMG_SIZE  = 128          # 输出图像分辨率
N_PTS     = 4096         # 点云统一采样数
SEED      = 2025

IMG_EXT   = {".jpg", ".jpeg", ".png"}
img_tf = Compose([ToTensor()])

# ---------- 2. 文件读取 ---------- #
def load_image(p: Path) -> np.ndarray:
    """RGB → uint8 (3, IMG_SIZE, IMG_SIZE)"""
    return (img_tf(Image.open(p).convert("RGB")) * 255).byte().numpy()

def as_chw(arr: np.ndarray) -> np.ndarray:
    """把二维 / HWC / CHW / H**W**C(通道>4) 全部转成 CHW"""
    if arr.ndim == 2:                       # (H,W)
        arr = arr[None]                     # -> (1,H,W)
    elif arr.ndim == 3:
        # 把通道轴放到最前面：若 arr.shape[2] 像 256、3、4… 则认为 HWC
        if arr.shape[0] not in (1,3,4) and arr.shape[2] in (1,3,4,256):
            arr = arr.transpose(2,0,1)      # HWC -> CHW
        # 否则视为已是 CHW
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    return arr

def resize_tensor(chw: np.ndarray, target: int) -> np.ndarray:
    """(C,H,W) uint8/float → 双线性缩放到 target×target"""
    if target is None or chw.shape[1:] == (target, target):
        return chw
    t = torch.from_numpy(chw).unsqueeze(0).float()       # 1,C,H,W
    t = F.interpolate(t, size=target, mode="bilinear", align_corners=False)
    return t.squeeze(0).byte().numpy()

def load_npz(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        key = 'data' if 'data' in npz.files else npz.files[0]
        arr = npz[key]
    # 若是点云 / 其它非图像，你可以在这里直接 raise 跳过
    if arr.ndim == 2 and arr.shape[1] == 3:
        raise ValueError("skip point cloud")
    arr = as_chw(arr)                                     # -> CHW
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    # #print(f"arr shape: {arr.shape}")
    # arr = resize_tensor(arr, IMG_SIZE) # 尝试不做re'size，避免失真
    return arr     

def load_bin(p: Path) -> np.ndarray:
    """点云 .pcd.bin → (N_PTS, 5) float32（保留全部特征）"""
    raw = np.fromfile(p, dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f"Expected 5-channel point cloud, got size {raw.size}")
    
    pts = raw.reshape(-1, 5)
    if pts.shape[0] < N_PTS:
        #print(f"[warn] {p.name}: only {pts.shape[0]} points, padding to {N_PTS}")
        pad = np.zeros((N_PTS - pts.shape[0], 5), dtype=np.float32)
        pts = np.concatenate([pts, pad], axis=0)
    else:
        idx = np.random.choice(pts.shape[0], N_PTS, replace=False)
        pts = pts[idx]

    # optional: mean centering xyz only
    pts[:, :3] = pts[:, :3] - pts[:, :3].mean(0)
    return pts.astype("float32")  # shape: (2048, 5) 

# def load_bin(p: Path) -> np.ndarray:
#     """点云 .pcd.bin → (N_PTS,3) float32 (自动适配 3/4/5 通道)"""
#     raw = np.fromfile(p, dtype=np.float32)
#     if raw.size % 5 == 0: pts = raw.reshape(-1, 5)[:, :3]
#     elif raw.size % 4 == 0: pts = raw.reshape(-1, 4)[:, :3]
#     elif raw.size % 3 == 0: pts = raw.reshape(-1, 3)
#     else: raise ValueError(f"size {raw.size} not divisible by 3/4/5")
#     #print(f"pcd bin shape {raw.shape}")
#     idx = np.random.choice(
#         pts.shape[0], N_PTS, replace=pts.shape[0] < N_PTS)
#     return (pts[idx] - pts[idx].mean(0)).astype("float32")

def dispatcher(p: Path):
    suf = p.suffix.lower()
    if suf in IMG_EXT: return "img", load_image(p)
    if suf == ".bin":  return "pts", load_bin(p)
    if suf == ".npz":  return "arr", load_npz(p)
    raise ValueError(f"Unsupported file: {p}")

# ---------- 3. 主流程 ---------- #
def parse_args():
    ap = argparse.ArgumentParser(
        "整理本地数据为 train_data.npz / eval_data.npz (与 torchvision 脚本兼容)"
    )
    ap.add_argument("-i", "--input",  type=Path, default=Path("/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/"),
                    help="根目录，脚本会递归查找文件")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("../data/v2x_sim_mini/"),
                    help="输出目录")
    ap.add_argument("--train-ratio",  type=float, default=0.8,
                    help="train / eval 划分比")
    ap.add_argument("--seed",        type=int,   default=2025)
    return ap.parse_args()

def split_and_save(samples: List[np.ndarray],
                   name: str,
                   outdir: Path,
                   ratio: float):
    """随机划分并保存  train / eval"""
    if not samples:
        #print(f"[warn] 无合法 {name} 样本，跳过保存")
        return
    idx = list(range(len(samples)))
    random.shuffle(idx)
    n_tr = int(len(idx) * ratio)
    tr = np.stack([samples[i] for i in idx[:n_tr]])
    ev = np.stack([samples[i] for i in idx[n_tr:]])

    np.savez_compressed(outdir / f"{name}_train.npz", data=tr)
    np.savez_compressed(outdir / f"{name}_eval.npz",  data=ev)
    #print(f"✓ {name}: train {tr.shape}  eval {ev.shape}")

def main():
    args = parse_args()
    random.seed(args.seed)
    files = [p for p in args.input.rglob("*") if p.is_file()]
    #print(f"Found {len(files)} files → 加载中…")

    groups: Dict[str, List[np.ndarray]] = {"img": [], "pts": [], "arr": []}
    for p in tqdm(files):
        try:
            g, arr = dispatcher(p)
            groups[g].append(arr)
        except Exception as e:
            #print(f"[skip] {p.name}: {e}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for k in ("img", "pts", "arr"):
        split_and_save(groups[k], k, args.outdir, args.train_ratio)

    #print("全部完成，输出目录:", args.outdir.resolve())

if __name__ == "__main__":
    main()
