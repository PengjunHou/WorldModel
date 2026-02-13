# ae/ae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -------------------------
# Simple Conv AutoEncoder
# -------------------------
class ConvEncoder(nn.Module):
    def __init__(self, z_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),   # /2
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # /4
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(), # /8
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU() # /16
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, z_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, z_dim: int = 128, out_hw: Tuple[int, int] = (144, 256)):
        super().__init__()
        self.out_h, self.out_w = out_hw

        self.fc = nn.Linear(z_dim, 256 * 9 * 16)  # -> (256,9,16) as a start
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 18x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 36x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 72x128
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),    # 144x256
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 9, 16)
        x_hat = self.deconv(h)
        # 保证输出尺寸正确（有些输入尺寸变化时更稳）
        x_hat = F.interpolate(x_hat, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return x_hat


class AutoEncoder(nn.Module):
    def __init__(self, z_dim: int = 128, out_hw: Tuple[int, int] = (144, 256)):
        super().__init__()
        self.enc = ConvEncoder(z_dim=z_dim)
        self.dec = ConvDecoder(z_dim=z_dim, out_hw=out_hw)

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return z, x_hat


# -------------------------
# Perceptual Feature Extractor (frozen)
# Default: torchvision resnet features (no detector weights required)
# -------------------------
class ResNetPerceptual(nn.Module):
    """
    返回多个尺度的特征，用于 feature matching loss。
    默认用 torchvision resnet18/34/50 的前几层特征（可替换成 detector backbone）。
    """
    def __init__(self, variant="resnet18"):
        super().__init__()
        from torchvision import models

        if variant == "resnet18":
            m = models.resnet18(weights=None)  # 你如果本机有权重，可自己load_state_dict
        elif variant == "resnet34":
            m = models.resnet34(weights=None)
        elif variant == "resnet50":
            m = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # 拆出分层特征（conv1/bn1/relu/maxpool + layer1..4）
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1
        self.l2 = m.layer2
        self.l3 = m.layer3
        self.l4 = m.layer4

        # 冻结
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x) -> List[torch.Tensor]:
        feats = []
        h = self.stem(x); feats.append(h)
        h = self.l1(h);   feats.append(h)
        h = self.l2(h);   feats.append(h)
        h = self.l3(h);   feats.append(h)
        h = self.l4(h);   feats.append(h)
        return feats


def feature_matching_loss(perceptual_net: nn.Module, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """
    perceptual_net 必须冻结
    对原图 x 用 no_grad 提取特征，对重构图 x_hat 允许梯度回 AE
    """
    with torch.no_grad():
        fx = perceptual_net(x)
    fhat = perceptual_net(x_hat)  # 不要 no_grad
    loss = 0.0
    for a, b in zip(fx, fhat):
        loss = loss + (a - b).abs().mean()
    return loss
