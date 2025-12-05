"""
From Diffuser https://github.com/jannerm/diffuser

For MLP and UNet diffusion models.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample2d(nn.Module):
    def __init__(self, ch): 
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x): 
        return self.op(x)

class Upsample2d(nn.Module):
    def __init__(self, ch): 
        super().__init__() 
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x): 
        return self.op(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
    ):
        super().__init__()
        if activation_type == "Mish":
            act = nn.Mish()
        elif activation_type == "ReLU":
            act = nn.ReLU()
        else:
            raise "Unknown activation type for Conv1dBlock"

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            (
                Rearrange("batch channels horizon -> batch channels 1 horizon")
                if n_groups is not None
                else nn.Identity()
            ),
            (
                nn.GroupNorm(n_groups, out_channels, eps=eps)
                if n_groups is not None
                else nn.Identity()
            ),
            (
                Rearrange("batch channels 1 horizon -> batch channels horizon")
                if n_groups is not None
                else nn.Identity()
            ),
            act,
        )

    def forward(self, x):
        return self.block(x)


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
    ):
        super().__init__()
        padding = kernel_size // 2

        # 选择激活函数
        if activation_type == "Mish":
            act = nn.Mish()
        elif activation_type == "ReLU":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation type for Conv2dBlock, {activation_type}")

        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, padding=padding),
            (
                nn.GroupNorm(n_groups, out_channels, eps=eps)
                if n_groups is not None
                else nn.BatchNorm2d(out_channels)
            ),
            act,
        )

    def forward(self, x):
        return self.block(x)
    


# -------- 下/上采样（2D） --------
class Downsample2d(nn.Module):
    def __init__(self, ch): super().__init__(); self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)

class Upsample2d(nn.Module):
    def __init__(self, ch): super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)

# class Downsample2d(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.op = nn.AvgPool2d(kernel_size=2, stride=2)  # 或者 MaxPool2d
#     def forward(self, x):
#         return self.op(x)

# class Upsample2d(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.op = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
#     def forward(self, x, target_hw=None):
#         x = self.op(x)
#         # 强制对齐到 skip 的大小，避免 (10 vs 11) 的情况
#         if target_hw is not None:
#             x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
#         return x

