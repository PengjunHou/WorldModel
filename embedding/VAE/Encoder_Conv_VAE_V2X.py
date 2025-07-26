import torch
from torch import nn
from typing import List
import math

from pythae.models import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder


class Encoder_Conv_VAE_V2X(BaseEncoder):
    """
    通用卷积 Encoder：
    - 根据 `args.input_dim = (C, H, W)` 自动确定首通道数、下采样层数
    - 每层 stride=2，把最小边一路除 2，直到 <=4 像素时停止
    - GAP 压到 1×1，再映射到 `latent_dim`
    """

    def __init__(self, args: BaseAEConfig, base_channels: int = 64, max_channels: int = 1024):
        super().__init__()

        # 1. 基本属性
        print(f"input dim {args.input_dim}")
        print(f"lantet dim: {args.latent_dim}")
        self.input_dim   = args.input_dim            # (C, H, W)
        self.latent_dim  = args.latent_dim
        self.n_channels  = self.input_dim[0]

        C, H, W          = self.input_dim
        min_hw           = min(H, W)

        # 2. 计算需要多少个 stride‑2 下采样块
        #    直到特征最小边 <= 4
        n_down = max(1, int(math.ceil(math.log2(min_hw / 4))))  # e.g. 128→6, 64→4, 32→3

        # 3. 构造卷积层
        layers  = nn.ModuleList()
        in_c    = self.n_channels
        out_c   = base_channels

        for _ in range(n_down):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            )
            in_c  = out_c
            out_c = min(out_c * 2, max_channels)    # 通道逐级翻倍，上限 max_channels

        self.layers      = layers
        self.depth       = len(layers)

        # 4. 全局池化 + 线性映射
        self.gap         = nn.AdaptiveAvgPool2d(1)
        self.embedding   = nn.Linear(in_c, self.latent_dim)
        self.log_var     = nn.Linear(in_c, self.latent_dim)

    # ---------------- 前向传播 ---------------- #
    def forward(self,
                x: torch.Tensor,
                output_layer_levels: List[int] = None) -> ModelOutput:

        out_dict  = ModelOutput()
        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(self.depth >= l > 0 or l == -1 for l in output_layer_levels), \
                f"invalid layer index {output_layer_levels}"
            if -1 not in output_layer_levels:
                max_depth = max(output_layer_levels)

        out = x
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None and (i + 1) in output_layer_levels:
                out_dict[f"embedding_layer_{i+1}"] = out

        z_feat = self.gap(out).view(out.size(0), -1)        # (B, in_c)
        out_dict["embedding"]      = self.embedding(z_feat)
        out_dict["log_covariance"] = self.log_var(z_feat)
        return out_dict
