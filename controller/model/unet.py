"""
UNet implementation. Minorly modified from Diffusion Policy: https://github.com/columbia-ai-robotics/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conv1d_components.py

Set `smaller_encoder` to False for using larger observation encoder in ResidualBlock1D

"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import logging
from copy import deepcopy

BASEPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASEPATH)  

LOG = logging.getLogger(__name__)

from controller.model.modules import (
    SinusoidalPosEmb,
    Downsample2d,
    Upsample2d,
    Conv2dBlock,
)
from controller.model.common import ResidualMLP, SpatialEmb

class ResidualBlock2D(nn.Module):
    """
    与你1D版ResidualBlock1D逻辑一致：两层Conv2dBlock + FiLM条件（scale/bias）+ 残差捷径
    cond_dim 通常 = time_embed_dim + global_meta_dim（也可以拼其他条件）
    """
    def __init__(        
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=5,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        larger_encoder=False,
        groupnorm_eps=1e-5,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv2dBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    n_groups = n_groups,
                    activation_type = activation_type,
                    eps = groupnorm_eps,
                ),
                Conv2dBlock(
                    out_channels,
                    out_channels,
                    kernel_size,
                    n_groups = n_groups,
                    activation_type = activation_type,
                    eps = groupnorm_eps,
                ),
            ]
        )

        self.activateion_type = activation_type
        if self.activateion_type == "Mish":
            act = nn.Mish()
        elif self.activateion_type == "ReLU":
            act = nn.ReLU()
        else:
            raise "Unknown activation type for ConditionalResidualBlock2D"
        
        # FiLM Modulation
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels *= 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        if larger_encoder:
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, cond_channels),
                act,
                nn.Linear(cond_channels, cond_channels),
                act,
                nn.Linear(cond_channels, cond_channels),
            )
        else:
            self.cond_encoder = nn.Sequential(
                act,
                nn.Linear(cond_dim, cond_channels),
            )
        
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond_vec):
        """
        x: [B, C, H, W]
        cond_vec: [B, cond_dim]  （时间嵌入+全局meta等拼接）
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond_vec) # [B, out_ch*(1 or 2)]
        if self.cond_predict_scale:
            embed = embed.view(embed.shape[0], 2, self.out_channels, 1, 1)
            scale, bias = embed[:, 0], embed[:, 1]
            out = scale * out + bias
        else:
            bias = embed.view(embed.shape[0], self.out_channels, 1, 1)
            out = out + bias
        # #print(f"embed shape: {embed.shape}, out shape: {out.shape}")
        out = self.blocks[1](out)
        tmp = self.residual_conv(x)
        # #print(f"out shape: {out.shape}, tmp shape: {tmp.shape}")
        return out + tmp
    

class VisionUnet2D(nn.Module):
    """
    直接替换 DiffusionModel.network 使用。
    horizon_steps=1, action_dim=H*W
    输入 x: [B, 1, H*W]，内部 reshape 为 [B,1,H,W]。
    cond:
      - cond['x_in']: [B, C_ctx, H, W]
      - cond['global']: [B, D] (全局条件，可选)
    """
    def __init__(
        self,
        backbone,
        H, W,       # action map 的高宽
        ctx_channels,
        diffusion_step_embed_dim=32,    # 时间步嵌入维度
        dim=32,
        dim_mults=(1, 2, 4, 8),
        cond_dim=0,
        larger_encoder=False,
        cond_mlp_dims=None,
        kernel_size=3,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        groupnorm_eps=1e-5,
        spatial_emb=0,
        dropout=0,
        num_img=1,
        img_cond_steps = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.H, self.W = H, W
        self.ctx_channels = ctx_channels
        self.in_ch = ctx_channels + 1   # +1 for noisy x_t channel
        self.out_ch = 1
        self.img_cond_steps = img_cond_steps
        self.num_img = num_img

        # 视觉条件压缩
        if spatial_emb > 0:
            assert spatial_emb > 1
            if num_img > 1:  # TODO: 多图像情况
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim,
                    proj_dim=spatial_emb,
                    dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            visual_feature_dim = 128  # 设个默认维度
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim * num_img, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        # 通道金字塔
        chs = [dim * m for m in dim_mults]
        in_out = list(zip([self.in_ch, *chs[:-1]], chs))
        # #print(f"[VisionUnet2D] Channel dimensions: {in_out}")

        # 时间步 embedding
        dsed = diffusion_step_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        if cond_mlp_dims is not None:
            self.cond_mlp = ResidualMLP(
                dim_list=[cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            cond_block_dim = dsed + cond_mlp_dims[-1] + visual_feature_dim
        else:
            cond_block_dim = dsed + cond_dim + visual_feature_dim

        # Down path
        self.downs = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            self.downs.append(nn.ModuleList([
                ResidualBlock2D(c_in,  c_out, cond_block_dim, kernel_size, n_groups, activation_type, cond_predict_scale, larger_encoder, groupnorm_eps),
                ResidualBlock2D(c_out, c_out, cond_block_dim, kernel_size, n_groups, activation_type, cond_predict_scale, larger_encoder, groupnorm_eps),
                Downsample2d(c_out) if not is_last else nn.Identity(),
            ]))

        # Mid
        mid_ch = chs[-1]
        self.mid = nn.ModuleList([
            ResidualBlock2D(mid_ch, mid_ch, cond_block_dim, kernel_size, n_groups, activation_type, cond_predict_scale, larger_encoder, groupnorm_eps),
            ResidualBlock2D(mid_ch, mid_ch, cond_block_dim, kernel_size, n_groups, activation_type, cond_predict_scale, larger_encoder, groupnorm_eps),
        ])

        # Up path
        self.ups = nn.ModuleList()
        up_in_out = list(zip(reversed(chs[1:]), reversed(chs[:-1])))
        for i, (c_in, c_out) in enumerate(up_in_out):
            should_upsample = (len(up_in_out) == 1) or (i < len(up_in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResidualBlock2D(c_in*2, c_out, cond_block_dim, kernel_size, n_groups, activation_type, cond_predict_scale, larger_encoder, groupnorm_eps),
                ResidualBlock2D(c_out,   c_out, cond_block_dim, kernel_size, n_groups, activation_type, cond_predict_scale, larger_encoder, groupnorm_eps),
                Upsample2d(c_out),# if not should_upsample else nn.Identity(),
            ]))

        # Final head
        self.final = nn.Sequential(
            Conv2dBlock(chs[0], chs[0], kernel_size, n_groups, activation_type, groupnorm_eps),
            nn.Conv2d(chs[0], self.out_ch, 1),
        )


    def forward(self, x, time, cond):
        """
        x:    [B, 1, H*W]   —— DiffusionModel 提供的 x_t
        time: [B] or scalar —— 扩散步
        cond:
        - x_in:   [B, C_ctx, H, W]         （必需）
        - state:  [B, To, Do]              （可选）
        - rgb:    [B, To, 3, H_img, W_img] （可选, 需要 self.backbone）
        """
        #print(f"x shape: {x.shape}, time shape: {time.shape}")
        #print(f"cond keys: {list(cond.keys())}, cond['state_Ks'] shape: {cond['state_Ks'].shape if 'state_Ks' in cond else None}, cond['state_local_maps'] shape: {cond['state_local_maps'].shape if 'state_local_maps' in cond else None}, cond['state_fused_maps'] shape: {cond['state_fused_maps'].shape if 'state_fused_maps' in cond else None}")
        B = x.size(0)
        # 1) 把 x_t 还原为 2D，并和二维条件通道拼接
        x = x.view(B, -1, self.H, self.W)             # [B,1,H,W]
        # ctx = cond["x_in"]                              # [B,C_ctx,H,W]，xin是直接拼接的，在V2X中，local conf map应该作为rgb进入。也可以尝试直接拼接
        # x = torch.cat([x2d, ctx], dim=1)                # [B, C_ctx+1, H, W]

        # 2) 准备全局条件：state → (cond_mlp), rgb → backbone(+compress)
        state_vec = None
        if "state_Ks" in cond and cond["state_Ks"] is not None:
            state = cond["state_Ks"].view(B, -1)           # [B, To*Do]
            if hasattr(self, "cond_mlp"):
                state_vec = self.cond_mlp(state)        # [B, cond_mlp_dims[-1]]
            else:
                # 若没有 cond_mlp，就直接把展平的 state 当成全局向量
                state_vec = state

        visual_vec = None
        if hasattr(self, "backbone") and (cond.get("state_local_maps", None) is not None):
            rgb_local = cond["state_local_maps"].unsqueeze(1)                      # [B, T, N, 1, H_img, W_img], T = 1
            rgb_fused = cond["state_fused_maps"].unsqueeze(1)                        # [B, T, N, 1, H_img, W_img]
            rgb = torch.cat([rgb_local, rgb_fused], dim=2).squeeze(3)   # [B, 1, 2N, H_img, W_img]
            #print(f"rgb shape from cond: {rgb.shape}")
            B_rgb, T_rgb, C_img, H_img, W_img = rgb.shape
            # 只取最近 self.img_cond_steps 帧
            rgb = rgb[:, -self.img_cond_steps:]         # [B, T_used, 3, H_img, W_img]

            if self.num_img > 1:
                # 把 T_used 按 num_img 分组（比如 2 张图一起）
                rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H_img, W_img)
                # (b t n c h w) -> (b n (t*c) h w)   每个“相机位/图”合并时间-通道
                rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
                rgb1, rgb2 = rgb[:, 0].float(), rgb[:, 1].float()
                # 可选增强
                if getattr(self, "augment", False) and hasattr(self, "aug"):
                    rgb1, rgb2 = self.aug(rgb1), self.aug(rgb2)
                # backbone 特征
                feat1 = self.backbone(rgb1)             # 通常 [B, P, D]
                feat2 = self.backbone(rgb2)
                # 压缩（SpatialEmb / Linear）
                if isinstance(self.compress1, SpatialEmb):
                    # SpatialEmb 需要 state（prop）做条件
                    if state_vec is None:
                        # 若没有 state_vec，但 SpatialEmb 需要 prop_dim>0，会出错；此处简单兜底
                        prop = torch.zeros(B, self.compress1.prop_dim, device=x.device)
                    else:
                        # 若 state 是 cond_mlp 输出，这里通常希望用 raw state：按需求替换
                        prop = state.view(B, -1) if "state_Ks" in cond else state_vec
                    v1 = self.compress1(feat1, prop)    # [B, spatial_emb]
                    v2 = self.compress2(feat2, prop)
                else:
                    v1 = self.compress(feat1.flatten(1, -1))  # [B, visual_feature_dim]
                    v2 = self.compress(feat2.flatten(1, -1))
                visual_vec = torch.cat([v1, v2], dim=-1)       # [B, 2*spatial_emb] 或 [B, 2*visual_feature_dim]
            else:
                # 单图像：把 (b t c h w) -> (b (t c) h w)，再 backbone
                rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w").float()
                if getattr(self, "augment", False) and hasattr(self, "aug"):
                    rgb = self.aug(rgb)
                feat = self.backbone(rgb)                       # [B, P, D] 或其他
                if hasattr(self, "compress") and isinstance(self.compress, SpatialEmb):
                    if state_vec is None:
                        prop = torch.zeros(B, self.compress.prop_dim, device=x.device)
                    else:
                        prop = state.view(B, -1) if "state_Ks" in cond else state_vec
                    visual_vec = self.compress(feat, prop)      # [B, spatial_emb]
                else:
                    # 线性压缩到 visual_feature_dim
                    visual_vec = self.compress(feat.flatten(1, -1))  # [B, visual_feature_dim]

        # 3) 时间嵌入 + 拼 global_feature（FiLM 条件向量）
        if not torch.is_tensor(time):
            time = torch.tensor([time], dtype=torch.long, device=x.device)
        if time.ndim == 0:
            time = time[None].to(x.device)
        time = time.expand(B)
        t_emb = self.time_mlp(time)                               # [B, dsed]

        pieces = [t_emb]
        if state_vec is not None:
            pieces.append(state_vec)
        else:
            state_padding = torch.zeros(B, 0, device=x.device)
        if visual_vec is not None:
            pieces.append(visual_vec)
        global_feature = torch.cat(pieces, dim=-1)                # [B, cond_block_dim]（和 __init__ 的 cond_block_dim 对齐）
        # print(f"global_feature shape: {global_feature.shape}, t_emb shape: {t_emb.shape}, x shape: {x.shape}")

        # 4) U-Net 编解码
        skips = []
        # #print(f"x shape before U-Net: {x.shape}, global_feature: {global_feature.shape}")
        for res1, res2, down in self.downs:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            skips.append(x)
            x = down(x)

        for block in self.mid:
            x = block(x, global_feature)

        for (res1, res2, up), skip in zip(self.ups, reversed(skips)):
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            # #print(f"x shape before upsample: {x.shape}, skip shape: {skip.shape}")

            x = torch.cat([x, skip], dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = up(x)
        # #print(f"x shape before final: {x.shape}")
        # 5) 输出回到 [B, 1, H*W]
        out2d = self.final(x)                                      # [B,1,H,W]
        # #print(f"out2d shape: {out2d.shape}, self.H: {self.H}, self.W: {self.W}")
        out = out2d.view(B, 1, self.H * self.W)
        return out



if __name__ == "__main__":
    from vit import VitEncoder, VitEncoderConfig

    obs_shape = [3, 128, 128]
    backbone = VitEncoder(
        obs_shape=obs_shape,        #这里没用到
        cfg=VitEncoderConfig(),
        num_channel=10,              # 这里的channel数量是cond rgb的channel数量
        img_h=obs_shape[1],
        img_w=obs_shape[2],
    )
    
    enc = VisionUnet2D(
        backbone=backbone,
        H=16, W=16,
        ctx_channels= 0, #obs_shape[0],  # 这里的channel数量是去噪的x + x_in的channel数量
        diffusion_step_embed_dim=32,
        dim=16,
        dim_mults=(1, 2, 4, 8),
        cond_dim=10, 
        larger_encoder=False,
        cond_mlp_dims=None,
        kernel_size=3,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        groupnorm_eps=1e-5,
        spatial_emb=0,
        dropout=0.1,
        num_img=1,
    )
    #print(enc)
    x = torch.randn(4, 1, 16*16)
    time = torch.randint(0, 1000, (4,))
    cond = {"state_Ks": torch.randn(4, 5, 2), "state_local_maps": torch.randn(4, 5, 128, 128), "state_fused_maps": torch.randn(4, 5, 128, 128)}
    with torch.no_grad():
        y = enc(x, time, cond)
    #print(y.shape)  # [4,1,16*16]