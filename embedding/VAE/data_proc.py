#!/usr/bin/env python3
import argparse, random
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, InterpolationMode

# ---------- 全局参数 ---------- #
IMG_SIZE = None
N_PTS = 4096
SEED = 2025

IMG_EXT = {".jpg", ".jpeg", ".png"}
# img_tf = Compose([Resize(IMG_SIZE), CenterCrop(IMG_SIZE), ToTensor()])
img_tf = Compose([
    Resize((225, 400)),
    ToTensor()
])

# ---------- 读取函数 ---------- #
def load_image(p: Path) -> np.ndarray:
    return (img_tf(Image.open(p).convert("RGB")) * 255).byte().numpy()

def as_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[None]
    elif arr.ndim == 3:
        if arr.shape[0] not in (1, 3, 4) and arr.shape[2] in (1, 3, 4, 256):
            arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"unsupported ndim {arr.ndim}")
    return arr

def load_npz(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        key = 'data' if 'data' in npz.files else npz.files[0]
        arr = npz[key]
    if arr.ndim == 2 and arr.shape[1] == 3:
        raise ValueError("skip point cloud")
    arr = as_chw(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype("uint8")
    return arr

def load_bin(p: Path) -> np.ndarray:
    raw = np.fromfile(p, dtype=np.float32)
    if raw.size % 5 != 0:
        raise ValueError(f"Expected 5-channel point cloud, got size {raw.size}")
    pts = raw.reshape(-1, 5)
    if pts.shape[0] < N_PTS:
        pad = np.zeros((N_PTS - pts.shape[0], 5), dtype=np.float32)
        pts = np.concatenate([pts, pad], axis=0)
    else:
        idx = np.random.choice(pts.shape[0], N_PTS, replace=False)
        pts = pts[idx]
    pts[:, :3] -= pts[:, :3].mean(0)
    return pts.astype("float32")

def get_loader(suffix: str):
    if suffix in IMG_EXT:
        return "img", load_image
    elif suffix == ".bin":
        return "pts", load_bin
    elif suffix == ".npz":
        return "arr", load_npz
    else:
        raise ValueError(f"Unsupported suffix: {suffix}")

# ---------- 按 shape 分组并保存 ---------- #
def write_split_npz_streaming(name: str, files: List[Path], loader_fn, outdir: Path, ratio: float):
    random.shuffle(files)
    n_train = int(len(files) * ratio)
    train_files, eval_files = files[:n_train], files[n_train:]

    def group_by_shape(subset_files: List[Path]):
        shape_to_data = {}
        for p in tqdm(subset_files, desc="Grouping by shape"):
            try:
                arr = loader_fn(p)
                shape = tuple(arr.shape)
                if shape not in shape_to_data:
                    shape_to_data[shape] = []
                shape_to_data[shape].append(arr)
            except Exception as e:
                #print(f"[skip] {p.name}: {e}")
        return shape_to_data

    def save_grouped_npz(data_dict: dict, prefix: str):
        for shape, arrays in data_dict.items():
            shape_str = "x".join(map(str, shape))
            filename = outdir / f"{name}_{prefix}_{shape_str}.npz"
            #print(f"Saving {len(arrays)} samples with shape {shape} → {filename.name}")
            np.savez_compressed(filename, data=np.stack(arrays))

    train_data = group_by_shape(train_files)
    eval_data = group_by_shape(eval_files)
    save_grouped_npz(train_data, "train")
    save_grouped_npz(eval_data, "eval")
    #print(f"✓ {name}: train {len(train_files)}, eval {len(eval_files)}")

# ---------- 主函数 ---------- #
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

def main():
    args = parse_args()
    random.seed(args.seed)
    all_files = [p for p in args.input.rglob("*") if p.is_file()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    file_groups = {"img": [], "pts": [], "arr": []}

    #print("分类中…")
    for p in tqdm(all_files):
        if not p.is_file():
            continue
        try:
            gtype, _ = get_loader(p.suffix.lower())
            file_groups[gtype].append(p)
        except:
            continue

    for k in ["img"]:  # 可改为 ["img", "pts", "arr"]
        if file_groups[k]:
            _, loader_fn = get_loader(file_groups[k][0].suffix.lower())
            write_split_npz_streaming(k, file_groups[k], loader_fn, args.outdir, args.train_ratio)

    #print("✅ 全部完成:", args.outdir.resolve())

if __name__ == "__main__":
    main()
