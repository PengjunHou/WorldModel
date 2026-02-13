# ae/build_latents_npz.py
import os
import numpy as np
from PIL import Image

import torch
from ae_model import AutoEncoder


def load_img(path: str, img_size=(256,144)):
    img_w, img_h = img_size
    if (path is None) or (len(str(path)) == 0) or (not os.path.exists(str(path))):
        return torch.zeros(3, img_h, img_w, dtype=torch.float32)
    img = Image.open(str(path)).convert("RGB").resize((img_w, img_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).contiguous()


@torch.no_grad()
def encode_paths_to_latents(ae: AutoEncoder, paths: np.ndarray, img_size=(256,144), batch_imgs=128, device="cuda"):
    """
    paths: (B,L,N) or (B,H,N)
    return z: same shape + z_dim (B,L,N,z_dim)
    """
    B, T, N = paths.shape
    z_dim = ae.enc.fc.out_features

    out = np.zeros((B, T, N, z_dim), dtype=np.float32)

    # 展平
    flat = paths.reshape(-1)
    total = flat.shape[0]

    # 分 batch 做编码
    idx = 0
    while idx < total:
        j = min(total, idx + batch_imgs)

        imgs = []
        for k in range(idx, j):
            imgs.append(load_img(flat[k], img_size=img_size))
        x = torch.stack(imgs, dim=0).to(device)

        z = ae.enc(x).detach().cpu().numpy()  # (bs,z_dim)
        out.reshape(-1, z_dim)[idx:j] = z

        idx = j

    return out


def build_latents_npz(
    in_npz: str,
    ae_ckpt: str,
    out_npz: str,
    z_dim: int = 128,
    img_size=(256,144),
    device="cuda",
    batch_imgs=128
):
    data = np.load(in_npz, allow_pickle=True)

    # load AE
    ae = AutoEncoder(z_dim=z_dim, out_hw=(img_size[1], img_size[0])).to(device)
    ck = torch.load(ae_ckpt, map_location=device)
    ae.load_state_dict(ck["model"])
    ae.eval()

    image_paths_past = data["image_paths_past"]  # (B,L,N)
    z_past = encode_paths_to_latents(ae, image_paths_past, img_size=img_size, batch_imgs=batch_imgs, device=device)

    out_dict = {}
    # 原字段全部拷贝
    for k in data.files:
        out_dict[k] = data[k]
    # 新字段
    out_dict["z_past"] = z_past.astype(np.float32)
    out_dict["z_dim"] = np.array([z_dim], dtype=np.int32)

    # 可选 future
    if "image_paths_future" in data.files:
        image_paths_future = data["image_paths_future"]  # (B,H,N)
        z_future = encode_paths_to_latents(ae, image_paths_future, img_size=img_size, batch_imgs=batch_imgs, device=device)
        out_dict["z_future"] = z_future.astype(np.float32)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **out_dict)
    print(f"[OK] saved latents dataset to: {out_npz}")
    print(f"z_past shape: {out_dict['z_past'].shape}")
    if "z_future" in out_dict:
        print(f"z_future shape: {out_dict['z_future'].shape}")


if __name__ == "__main__":
    in_npz  = "/home/peh324/Codes/WorldModel/data/GODE/dataset_paths.npz"
    ae_ckpt = "/home/peh324/Codes/WorldModel/result/checkpoints/ae/ae_best.pt"
    out_npz = "/home/peh324/Codes/WorldModel/data/GODE/dataset_latents.npz"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    build_latents_npz(
        in_npz=in_npz,
        ae_ckpt=ae_ckpt,
        out_npz=out_npz,
        z_dim=128,
        img_size=(256,144),
        device=device,
        batch_imgs=128
    )
