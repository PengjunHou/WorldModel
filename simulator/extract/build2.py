# ae/build_latents_npz.py
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from ae_model import AutoEncoder


def load_img(path: str, img_size=(256, 144)):
    """
    Returns: torch.FloatTensor (3, H, W) in [0,1]
    """
    img_w, img_h = img_size
    if (path is None) or (len(str(path)) == 0) or (not os.path.exists(str(path))):
        return torch.zeros(3, img_h, img_w, dtype=torch.float32)

    img = Image.open(str(path)).convert("RGB").resize((img_w, img_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)


class ImgPathDataset(Dataset):
    def __init__(self, flat_paths: np.ndarray, img_size=(256, 144)):
        self.flat_paths = flat_paths
        self.img_size = img_size

    def __len__(self):
        return int(self.flat_paths.shape[0])

    def __getitem__(self, i: int):
        return load_img(self.flat_paths[i], img_size=self.img_size)


@torch.no_grad()
def encode_paths_to_latents(
    ae: AutoEncoder,
    paths: np.ndarray,
    img_size=(256, 144),
    batch_imgs: int = 128,
    device: str = "cuda",
    num_workers: int = 8,
    use_amp: bool = True,
):
    """
    paths: (B,T,N)  (your past or future image paths)
    return z: (B,T,N,z_dim)
    """
    assert paths.ndim == 3, f"expected (B,T,N), got {paths.shape}"
    B, T, N = paths.shape
    z_dim = ae.enc.fc.out_features

    flat = paths.reshape(-1)  # length = B*T*N
    total = flat.shape[0]

    # DataLoader for parallel decode/resize on CPU
    ds = ImgPathDataset(flat, img_size=img_size)
    dl = DataLoader(
        ds,
        batch_size=batch_imgs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    ae = ae.to(device).eval()

    out = np.zeros((total, z_dim), dtype=np.float32)
    idx = 0

    for x in dl:
        # async H2D if pinned memory
        x = x.to(device, non_blocking=True)

        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = ae.enc(x)  # (bs,z_dim)
        else:
            z = ae.enc(x)

        z = z.float().cpu().numpy()
        bs = z.shape[0]
        out[idx : idx + bs] = z
        idx += bs

    return out.reshape(B, T, N, z_dim)


def build_latents_npz(
    in_npz: str,
    ae_ckpt: str,
    out_npz: str,
    z_dim: int = 128,
    img_size=(256, 144),
    device="cuda",
    batch_imgs=128,
    num_workers=8,
    use_amp=True,
):
    data = np.load(in_npz, allow_pickle=True)

    # load AE
    ae = AutoEncoder(z_dim=z_dim, out_hw=(img_size[1], img_size[0])).to(device)
    ck = torch.load(ae_ckpt, map_location=device)
    ae.load_state_dict(ck["model"])
    ae.eval()

    # past
    image_paths_past = data["image_paths_past"]  # (B,L,N)
    z_past = encode_paths_to_latents(
        ae,
        image_paths_past,
        img_size=img_size,
        batch_imgs=batch_imgs,
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
    )

    out_dict = {k: data[k] for k in data.files}
    out_dict["z_past"] = z_past.astype(np.float32)
    out_dict["z_dim"] = np.array([z_dim], dtype=np.int32)

    # optional future
    if "image_paths_future" in data.files:
        image_paths_future = data["image_paths_future"]  # (B,H,N)
        z_future = encode_paths_to_latents(
            ae,
            image_paths_future,
            img_size=img_size,
            batch_imgs=batch_imgs,
            device=device,
            num_workers=num_workers,
            use_amp=use_amp,
        )
        out_dict["z_future"] = z_future.astype(np.float32)

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, **out_dict)

    print(f"[OK] saved latents dataset to: {out_npz}")
    print(f"z_past shape: {out_dict['z_past'].shape}")
    if "z_future" in out_dict:
        print(f"z_future shape: {out_dict['z_future'].shape}")


if __name__ == "__main__":
    in_npz = "/home/peh324/Codes/WorldModel/data/GODE/dataset_paths.npz"
    ae_ckpt = "/home/peh324/Codes/WorldModel/result/checkpoints/ae/ae_best.pt"
    out_npz = "/home/peh324/Codes/WorldModel/data/GODE/dataset_latents2.npz"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tips:
    # - if you have many CPU cores: num_workers=8~16
    # - if GPU is strong: batch_imgs=256~512
    # - if GPU memory limited: lower batch_imgs
    build_latents_npz(
        in_npz=in_npz,
        ae_ckpt=ae_ckpt,
        out_npz=out_npz,
        z_dim=128,
        img_size=(256, 144),
        device=device,
        batch_imgs=256 if device.startswith("cuda") else 64,
        num_workers=8 if device.startswith("cuda") else 2,
        use_amp=True,
    )
