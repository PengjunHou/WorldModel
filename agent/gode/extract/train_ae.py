# ae/train_ae.py
import os, csv, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ae_model import AutoEncoder, ResNetPerceptual, feature_matching_loss


class ImagePathDataset(Dataset):
    """
    从 dataset_paths.npz 里把所有可用的图片路径摊平，训练AE用。
    """
    def __init__(self, npz_path: str, img_size=(256,144), max_images=None):
        data = np.load(npz_path, allow_pickle=True)
        paths = data["image_paths_past"]  # (B,L,N)
        paths = paths.reshape(-1)
        
        # print("npz keys:", data.files)
        # print("image_paths_past shape:", paths.shape)

        # # 打印前10个
        # flat = paths.reshape(-1)
        # print("sample raw paths:")
        # for i in range(10):
        #     print(i, repr(flat[i]))

        # 过滤空路径
        paths = [str(p) for p in paths if isinstance(p, (str, np.str_)) and len(str(p)) > 0]

        # 可选：限制数量
        if max_images is not None and len(paths) > max_images:
            random.shuffle(paths)
            paths = paths[:max_images]

        self.paths = paths
        self.img_w, self.img_h = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        #print(f"data: {}")
        p = self.paths[idx]
        if (not os.path.exists(p)):
            # 缺图直接返回零
            x = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
            return x

        # print(f"Loading image {idx}/{len(self.paths)}: {p}")
        img = Image.open(p).convert("RGB").resize((self.img_w, self.img_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2,0,1).contiguous()
        return x


def save_ckpt(path, model, opt, epoch, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss
    }, path)


def plot_loss(csv_path, out_png):
    import matplotlib.pyplot as plt
    steps, l_total, l_img, l_feat = [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            steps.append(int(r["step"]))
            l_total.append(float(r["loss_total"]))
            l_img.append(float(r["loss_img"]))
            l_feat.append(float(r["loss_feat"]))
    plt.figure()
    plt.plot(steps, l_total, label="total")
    plt.plot(steps, l_img, label="img")
    plt.plot(steps, l_feat, label="feat")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


def train_ae(
    dataset_npz: str,
    z_dim: int = 128,
    img_size=(256,144),
    batch_size=64,
    lr=3e-4,
    epochs=5,
    lambda_feat=0.1,
    device="cuda",
    num_workers=4,
    max_images=200000,
    save_dir = '/home/peh324/Codes/WorldModel',
):
    os.makedirs(os.path.join(save_dir, "result/checkpoints/ae"), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ds = ImagePathDataset(dataset_npz, img_size=img_size, max_images=max_images)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

    ae = AutoEncoder(z_dim=z_dim, out_hw=(img_size[1], img_size[0])).to(device)
    perceptual = ResNetPerceptual("resnet18").to(device)  # 冻结特征网络

    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-4)

    log_csv = os.path.join("logs", "ae_losses.csv")
    with open(log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss_total", "loss_img", "loss_feat"])
        writer.writeheader()

    best = 1e9
    step = 0
    ae.train()

    l1 = nn.L1Loss()

    for ep in range(1, epochs + 1):
        for x in dl:
            x = x.to(device, non_blocking=True)  # (B,3,H,W)
            # print("x stats:", x.min().item(), x.max().item(), x.mean().item())


            z, x_hat = ae(x)
            loss_img = l1(x_hat, x)
            loss_feat = feature_matching_loss(perceptual, x, x_hat)
            loss = loss_img + lambda_feat * loss_feat

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()

            # log
            if step % 20 == 0:
                print(f"[AE] ep={ep} step={step} loss={loss.item():.4f} img={loss_img.item():.4f} feat={loss_feat.item():.4f}")

            with open(log_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss_total", "loss_img", "loss_feat"])
                writer.writerow({
                    "step": step,
                    "epoch": ep,
                    "loss_total": float(loss.item()),
                    "loss_img": float(loss_img.item()),
                    "loss_feat": float(loss_feat.item())
                })

            # save
            if loss.item() < best:
                best = loss.item()
                save_ckpt(os.path.join(save_dir, "checkpoints/ae/ae_best.pt"), ae, opt, ep, best)
            if step % 200 == 0:
                save_ckpt(os.path.join(save_dir, "checkpoints/ae/ae_last.pt"), ae, opt, ep, best)

            step += 1

    save_ckpt(os.path.join(save_dir, "result/checkpoints/ae/ae_best.pt"), ae, opt, epochs, best)
    plot_loss(log_csv, os.path.join("logs", "ae_loss_curve.png"))
    print(f"[AE] done. best={best:.6f} log={log_csv} curve=logs/ae_loss_curve.png")


if __name__ == "__main__":
    dataset_npz = "/home/peh324/Codes/WorldModel/data/GODE/dataset_paths.npz"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ae(
        dataset_npz=dataset_npz,
        z_dim=128,
        img_size=(256,144),
        batch_size=64,
        lr=3e-4,
        epochs=10,
        lambda_feat=0.1,
        device=device,
        num_workers=4,
        max_images=200000,
        save_dir = '/home/peh324/Codes/WorldModel',
    )
