import torch
from torch import nn
from typing  import List
from pythae.models import BaseAEConfig
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder


import math

# -------- ResBlock 保持不变 -------- #
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, 1),
        )
    def forward(self, x):
        return x + self.conv_block(x)


# -------- 通用 / 任意分辨率 Decoder -------- #
class Decoder_ResNet_VAE_V2X(BaseDecoder):
    """
    兼容任意 (C_out,H,W) 的 ResNet‑风格解码器。
    latent_dim → FC→(C0,4,4) → ResBlocks → n×ConvT2d(×2) → interpolate→(H,W) → Conv2d→Sigmoid
    """
    def __init__(self,
                 args,
                 base_channels: int = 64,
                 max_channels:   int = 128,
                 interp_mode:    str = "bilinear"):
        super().__init__()

        C_out, H, W = args.input_dim
        assert H >= 4 and W >= 4, "H, W 必须 ≥ 4 像素"

        self.latent_dim = args.latent_dim
        self.target_hw  = (H, W)
        self.interp_mode = interp_mode

        # 需要多少次 ×2 才能覆盖 max(H,W)
        self.start_sz = 4
        self.n_up = math.ceil(math.log2(max(H, W) / self.start_sz))

        # 起始通道 = base_channels * 2^(n_up-1)（不超过 max_channels）
        self.start_ch = min(max_channels, base_channels * 2 ** (self.n_up - 1))

        # 1) latent → (C0,4,4)
        self.fc = nn.Linear(self.latent_dim,
                            self.start_ch * self.start_sz * self.start_sz)

        # 2) ResBlocks at 4×4
        blocks = [nn.Sequential(
            ResBlock(self.start_ch, max(self.start_ch // 4, 32)),
            ResBlock(self.start_ch, max(self.start_ch // 4, 32))
        )]

        # 3) 逐步反卷积上采样
        in_c = self.start_ch
        for _ in range(self.n_up):
            out_c = max(base_channels, in_c // 2)
            blocks.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)
            )
            in_c = out_c

        self.up_blocks = nn.ModuleList(blocks)

        # 4) 输出卷积到 C_out
        self.conv_out = nn.Conv2d(in_c, C_out, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()   # 若数据归一化到 [-1,1] 可改成 nn.Tanh()

        # 供 Pythae 打印
        self.input_dim = args.input_dim


    # ---------- forward ---------- #
    def forward(self,
                z: torch.Tensor,
                output_layer_levels: List[int] = None) -> ModelOutput:

        out_dict  = ModelOutput()
        depth     = len(self.up_blocks)
        max_depth = depth

        if output_layer_levels is not None:
            assert all(depth >= l > 0 or l == -1 for l in output_layer_levels), \
                f"invalid layer index {output_layer_levels}"
            if -1 not in output_layer_levels:
                max_depth = max(output_layer_levels)

        # FC → 4×4 feature map
        out = self.fc(z).view(z.size(0), self.start_ch, self.start_sz, self.start_sz)

        for i in range(max_depth):
            out = self.up_blocks[i](out)
            if output_layer_levels is not None and (i + 1) in output_layer_levels:
                out_dict[f"reconstruction_layer_{i+1}"] = out

        # 尺寸未必刚好，插值 / 裁剪到目标大小
        if out.shape[2:] != self.target_hw:
            out = F.interpolate(out,
                                size=self.target_hw,
                                mode=self.interp_mode,
                                align_corners=False if self.interp_mode in ("bilinear", "bicubic") else None)

        out = self.conv_out(out)
        out = self.output_act(out)
        out_dict["reconstruction"] = out        # (B,C_out,H,W)
        return out_dict


# ---------- 通用 ResBlock ---------- #
class ResBlock2(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, 1),
        )
    def forward(self, x):
        return x + self.conv_block(x)


# ---------- 通用 Decoder ---------- #
class Decoder_ResNet_VAE_V2X2(BaseDecoder):
    """
    自动适配输入维度的 ResNet‑风格解码器：
      latent_dim → FC → (C0,4,4) → ResBlocks → N×ConvTrans2d(×2) → 输出 (C_out,H,W)
    """

    def __init__(self,
                 args: BaseAEConfig,
                 base_channels: int = 64,
                 max_channels: int = 128):
        """
        base_channels: C0 基数；max_channels: 起始最大通道（防止过大）
        """
        super().__init__()

        C_out, H, W   = args.input_dim
        assert H % 4 == 0 and W % 4 == 0, "H,W 应该是 4 的倍数 (power-of-two 最佳)"

        # 上采样次数：从 4×4 反卷积到原始 H×W
        n_up_h = int(math.log2(H // 4))
        n_up_w = int(math.log2(W // 4))
        assert 2 ** n_up_h * 4 == H and 2 ** n_up_w * 4 == W, \
            "H,W 需为 4×2^k 形式"

        self.n_up   = max(n_up_h, n_up_w)          # 通常二者相等，方形图像
        self.latent_dim = args.latent_dim
        self.start_sz   = 4

        # 起始通道：base_channels * 2^(n_up-1)，但不超 max_channels
        self.start_ch = min(max_channels, base_channels * 2 ** (self.n_up - 1))

        # 1️⃣  latent → (C0,4,4)
        self.fc = nn.Linear(self.latent_dim,
                            self.start_ch * self.start_sz * self.start_sz)

        up_layers = nn.ModuleList()

        # 2️⃣  两个 ResBlock 在 4×4 处
        up_layers.append(
            nn.Sequential(
                ResBlock(self.start_ch, max(self.start_ch // 4, 32)),
                ResBlock(self.start_ch, max(self.start_ch // 4, 32))
            )
        )

        # 3️⃣  反卷积层
        in_c = self.start_ch
        for k in range(self.n_up):
            # 通道每次 /2，直到 base_channels
            out_c = max(base_channels, in_c // 2) if k < self.n_up - 1 else C_out
            up_layers.append(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1,
                                   output_padding=0)  #  spatial ×2
            )
            in_c = out_c

        self.up_layers = up_layers
        self.depth     = len(up_layers)
        self.output_act = nn.Sigmoid()   # 若想 [-1,1] 用 nn.Tanh()

        # 记录到属性，供 Pythae 打印
        self.input_dim = args.input_dim

    # ---------- forward ---------- #
    def forward(self,
                z: torch.Tensor,
                output_layer_levels: List[int] = None) -> ModelOutput:

        out_dict  = ModelOutput()
        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(self.depth >= l > 0 or l == -1 for l in output_layer_levels), \
                f"invalid layer index {output_layer_levels}"
            if -1 not in output_layer_levels:
                max_depth = max(output_layer_levels)

        # FC → feature map
        out = self.fc(z).view(z.size(0), self.start_ch,
                              self.start_sz, self.start_sz)

        for i in range(max_depth):
            out = self.up_layers[i](out)
            if output_layer_levels is not None and (i + 1) in output_layer_levels:
                out_dict[f"reconstruction_layer_{i+1}"] = out

        out = self.output_act(out)
        out_dict["reconstruction"] = out         # (B,C_out,H,W)
        return out_dict

