"""
Implementation of Multi-layer Perception (MLP).

Residual model is taken from https://github.com/ALRhub/d3il/blob/main/agents/models/common/mlp.py

"""

import torch
from torch import nn
from collections import OrderedDict
import logging


activation_dict = nn.ModuleDict(
    {
        "ReLU": nn.ReLU(),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity(),
        "Softplus": nn.Softplus(),
    }
)


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="Tanh",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Construct module list: if use `Python List`, the modules are not
        # added to computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        self.append_layers = append_layers
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))

            # add activation function
            act = (
                activation_dict[activation_type]
                if idx != num_layer - 1
                else activation_dict[out_activation_type]
            )
            layers.append(("act_1", act))

            # re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x


class ResidualMLP(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
        self,
        dim_list,
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
    ):
        super(ResidualMLP, self).__init__()
        hidden_dim = dim_list[1]
        num_hidden_layers = len(dim_list) - 3
        assert num_hidden_layers % 2 == 0
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation_type=activation_type,
                    use_layernorm=use_layernorm,
                    dropout=dropout,
                )
                for _ in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        if use_layernorm_final:
            self.layers.append(nn.LayerNorm(dim_list[-1]))
        self.layers.append(activation_dict[out_activation_type])

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x


class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation_type="Mish",
        use_layernorm=False,
        dropout=0,
    ):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_dict[activation_type]
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-06)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-06)
        if dropout > 0:
            raise NotImplementedError("Dropout not implemented for residual MLP!")

    def forward(self, x):
        x_input = x
        if hasattr(self, "norm1"):
            x = self.norm1(x)
        x = self.l1(self.act(x))
        if hasattr(self, "norm2"):
            x = self.norm2(x)
        x = self.l2(self.act(x))
        return x + x_input


"""
Additional implementation of the ViT image encoder from https://github.com/hengyuan-hu/ibrl/tree/main

"""

class SpatialEmb(nn.Module):
    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super().__init__()

        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor):
        feat = feat.transpose(1, 2) # B x patch_dim x num_patch 

        '''
        🔹 3. 为什么不是拼在 patch_dim 上？

            这取决于我们希望 条件信息如何影响特征：

            (1) 拼在 patch 数 (N) 维度上（当前代码做法）：

            意味着条件被当成“额外的 token/patch”，和原始 patch 们一起处理；

            相当于 增加一些虚拟 patch 来携带条件信息；

            Linear 是在 “patch + cond” 空间做投影。

            (2) 拼在 embedding 维度 (D) 上：

            意味着条件信息作为 每个 patch embedding 的额外特征维度；

            相当于扩展每个 token 的表示维度，从 [D] → [D + cond_dim]；

            Linear 是在 “embedding 特征维度” 空间做投影。
        '''
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)

        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        z = self.dropout(z)
        return z            # [batch_size, proj_dim]，即每个样本被映射到一个 proj_dim 维的全局 embedding 向量。


class RandomShiftsAug:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )


# test random shift
if __name__ == "__main__":
    from PIL import Image
    import requests
    import numpy as np

    image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.resize((96, 96))

    image = torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
    aug = RandomShiftsAug(pad=4)
    image_aug = aug(image)
    image_aug = image_aug.squeeze().permute(1, 2, 0).numpy()
    image_aug = Image.fromarray(image_aug.astype(np.uint8))
    image_aug.show()
