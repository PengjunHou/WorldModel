"""
GODE网络模型 - 纯PyTorch实现(不依赖DGL)
用于车辆协同感知的融合map预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from torchdiffeq import odeint_adjoint, odeint


class GraphConvLayer(nn.Module):
    """
    图卷积层 - 纯PyTorch实现
    使用邻接矩阵进行消息传递
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim] 节点特征
            adj: [B, N, N] 邻接矩阵
        Returns:
            [B, N, out_dim] 更新后的节点特征
        """
        # 聚合邻居信息: [B, N, N] @ [B, N, in_dim] -> [B, N, in_dim]
        h_neigh = torch.bmm(adj, x)
        
        # 加上自身特征
        h_total = x + h_neigh
        
        # 线性变换
        out = self.linear(h_total)
        return F.relu(out)


class ControlledGDEFunc(nn.Module):
    """受控GDE函数 - 纯PyTorch实现"""
    def __init__(self, gnn_layers: nn.ModuleList, output_dim, hidden_dim):
        super().__init__()
        self.gnn = gnn_layers
        self.output_proj = nn.Linear(hidden_dim, output_dim)  # 256 -> 144
        self.h0 = None
        self.adj = None
        self.nfe = 0
    
    def set_graph(self, adj: torch.Tensor):
        """设置邻接矩阵"""
        self.adj = adj
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间
            x: [B*N, feature_dim] 展平的节点特征
        Returns:
            dx/dt: [B*N, feature_dim]
        """
        self.nfe += 1
        
        if self.h0 is None or self.adj is None:
            raise ValueError("h0 or adj not set!")
        
        # 拼接初始状态
        x = torch.cat([x, self.h0], dim=1)  # [B*N, feature_dim*2]
        
        # 获取batch大小和节点数
        B_N = x.shape[0]
        # 从adj推断batch和节点数
        B = self.adj.shape[0]
        N = self.adj.shape[1]
        
        # Reshape回[B, N, feature_dim*2]
        x = x.view(B, N, -1)
        
        # 通过GNN层
        for layer in self.gnn:
            x = layer(x, self.adj)
        
        # 投影回原始维度 (ODE要求输入输出维度一致)
        x = self.output_proj(x)  # [B, N, hidden_dim] -> [B, N, node_feature_dim]
        
        # Flatten回[B*N, feature_dim]
        x = x.view(B_N, -1)
        
        return x


class ODEBlock(nn.Module):
    """ODE求解器"""
    def __init__(
        self,
        odefunc: nn.Module,
        method: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True
    ):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        Args:
            x: [B*N, feature_dim] 初始状态
            T: 积分时间
        Returns:
            [B*N, feature_dim] 终态
        """
        integration_time = torch.tensor([0, T]).type_as(x)
        
        if self.adjoint_flag:
            out = odeint_adjoint(
                self.odefunc, x, integration_time,
                rtol=self.rtol, atol=self.atol, method=self.method
            )
        else:
            out = odeint(
                self.odefunc, x, integration_time,
                rtol=self.rtol, atol=self.atol, method=self.method
            )
        
        return out[-1]


class TemporalEncoder(nn.Module):
    """时序编码器 - 使用LSTM处理历史序列"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, feature_dim] 时序特征
        Returns:
            [B, hidden_dim] 编码后的特征
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class VehicleGODEPredictor(nn.Module):
    """
    单车GODE预测器 - 纯PyTorch实现
    
    输入:
    - 自己的历史local maps [T, H, W]
    - 自己的历史positions [T, 2]
    - 其他车辆的历史local maps [N-1, T, H, W]
    - 其他车辆的历史positions [N-1, T, 2]
    - 当前K值
    
    输出:
    - 预测的fused map [H, W]
    """
    
    def __init__(
        self,
        vae_encoder,
        map_size: Tuple[int, int] = (64, 64),
        vae_latent_dim: int = 32,
        position_dim: int = 2,
        temporal_hidden_dim: int = 128,
        gnn_hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_temporal_layers: int = 2,
        k_embed_dim: int = 16,
        comm_range: float = 100.0,
        ode_method: str = 'dopri5'
    ):
        super().__init__()
        self.map_size = map_size
        self.vae_latent_dim = vae_latent_dim
        self.comm_range = comm_range
        
        # 1. VAE编码器 (预训练,冻结)
        self.vae_encoder = vae_encoder
        for param in self.vae_encoder.parameters():
            param.requires_grad = False
        
        # 2. 位置编码器
        self.position_encoder = nn.Sequential(
            nn.Linear(position_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 3. K值嵌入
        self.k_embedding = nn.Sequential(
            nn.Linear(1, k_embed_dim),
            nn.ReLU()
        )
        
        # 4. 时序编码器
        self_temporal_input_dim = vae_latent_dim + 64
        self.self_temporal_encoder = TemporalEncoder(
            self_temporal_input_dim,
            temporal_hidden_dim,
            num_temporal_layers
        )
        
        other_temporal_input_dim = vae_latent_dim + 64
        self.other_temporal_encoder = TemporalEncoder(
            other_temporal_input_dim,
            temporal_hidden_dim,
            num_temporal_layers
        )
        
        # 5. GNN层
        node_feature_dim = temporal_hidden_dim + k_embed_dim
        gnn_input_dim = node_feature_dim * 2  # 拼接初始状态
        
        gnn_layers = nn.ModuleList([
            GraphConvLayer(
                gnn_input_dim if i == 0 else gnn_hidden_dim, 
                gnn_hidden_dim
            )
            for i in range(num_gnn_layers)
        ])
        
        self.gde_func = ControlledGDEFunc(gnn_layers,
                                        output_dim=node_feature_dim,  # 输出维度 = 输入维度
                                        hidden_dim=gnn_hidden_dim)
        self.ode_block = ODEBlock(self.gde_func, method=ode_method)
        
        # 6. 特征投影
        self.feature_proj = nn.Linear(node_feature_dim, vae_latent_dim)
        
        # 7. Map解码器
        self.decoder = self._build_decoder(vae_latent_dim, map_size)
    
    def _build_decoder(self, latent_dim: int, map_size: Tuple[int, int]) -> nn.Module:
        """构建解码器"""
        H, W = map_size
        start_size = 8
        channels = 256
        
        self.fc_decode = nn.Linear(latent_dim, channels * start_size * start_size)
        
        layers = []
        current_size = start_size
        
        while current_size < H:
            layers.extend([
                nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
                nn.ReLU()
            ])
            channels = channels // 2
            current_size *= 2
        
        layers.append(nn.Conv2d(channels, 1, 3, 1, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def encode_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """使用VAE编码maps"""
        original_shape = maps.shape
        
        if len(original_shape) == 4:  # [B, T, H, W]
            B, T, H, W = original_shape
            maps = maps.reshape(B * T, 1, H, W)
            latents = self.vae_encoder.get_latent(maps)
            latents = latents.reshape(B, T, -1)
        else:  # [B, H, W]
            B, H, W = original_shape
            maps = maps.unsqueeze(1)
            latents = self.vae_encoder.get_latent(maps)
        
        return latents
    
    def build_adjacency_matrix(
        self,
        positions: torch.Tensor,
        comm_range: float
    ) -> torch.Tensor:
        """
        构建邻接矩阵
        
        Args:
            positions: [B, N, 2] 节点位置
            comm_range: 通信范围
        Returns:
            adj: [B, N, N] 邻接矩阵
        """
        B, N, _ = positions.shape
        device = positions.device
        
        # 计算距离矩阵
        # [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
        pos_i = positions.unsqueeze(2)  # [B, N, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, N, 2]
        diff = pos_i - pos_j  # [B, N, N, 2]
        distances = torch.norm(diff, dim=-1)  # [B, N, N]
        
        # 构建邻接矩阵 (距离在通信范围内为1)
        adj = (distances <= comm_range).float()
        
        # 移除自环(对角线设为0)
        # 但在GCN中我们会加上自身,所以这里保留自环也可以
        # 这里选择移除,在GraphConvLayer中手动加上
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        adj = adj - eye
        
        # 归一化邻接矩阵 (degree normalization)
        # D^{-1/2} A D^{-1/2}
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6  # [B, N, 1]
        degree_inv_sqrt = torch.pow(degree, -0.5)
        adj_normalized = degree_inv_sqrt * adj * degree_inv_sqrt.transpose(1, 2)
        
        return adj_normalized
    
    def forward(
        self,
        self_history_maps: torch.Tensor,       # [B, T, H, W]
        self_history_positions: torch.Tensor,  # [B, T, 2]
        others_maps: torch.Tensor,              # [B, N-1, T, H, W]
        others_positions: torch.Tensor,         # [B, N-1, T, 2]
        K_value: torch.Tensor,                  # [B, 1]
        integration_time: float = 1.0
    ) -> torch.Tensor:
        """
        预测融合map
        
        Returns:
            predicted_fused_map: [B, H, W]
        """
        B = self_history_maps.shape[0]
        T = self_history_maps.shape[1]
        N_others = others_maps.shape[1]
        device = self_history_maps.device
        
        # 1. 编码自己的历史
        self_map_features = self.encode_maps(self_history_maps)  # [B, T, latent_dim]
        self_pos_features = self.position_encoder(self_history_positions)  # [B, T, 64]
        self_temporal_features = torch.cat([self_map_features, self_pos_features], dim=-1)
        self_encoded = self.self_temporal_encoder(self_temporal_features)  # [B, temporal_hidden_dim]
        
        # 2. 编码K值
        k_embed = self.k_embedding(K_value)  # [B, k_embed_dim]
        
        # 3. 自己的节点特征
        self_node_features = torch.cat([self_encoded, k_embed], dim=-1)
        
        # 4. 编码其他车辆
        B_N = B * N_others
        others_maps_flat = others_maps.reshape(B_N, T, *self.map_size)
        others_positions_flat = others_positions.reshape(B_N, T, 2)
        
        others_map_features = self.encode_maps(others_maps_flat)
        others_pos_features = self.position_encoder(others_positions_flat)
        others_temporal_features = torch.cat([others_map_features, others_pos_features], dim=-1)
        others_encoded = self.other_temporal_encoder(others_temporal_features)
        
        others_node_features = others_encoded.reshape(B, N_others, -1)
        
        # 5. 补齐维度
        others_padding = torch.zeros(B, N_others, k_embed.shape[-1]).to(device)
        others_node_features = torch.cat([others_node_features, others_padding], dim=-1)
        
        print(f"gode: own node {self_node_features.shape}, other {others_node_features.shape}")
        
        # 6. 拼接所有节点: [B, N_total, feature_dim]
        all_node_features = torch.cat([
            self_node_features.unsqueeze(1),  # [B, 1, feature_dim]
            others_node_features              # [B, N-1, feature_dim]
        ], dim=1)  # [B, N_total, feature_dim]
        
        print(f"all {all_node_features.shape}")
        N_total = 1 + N_others
        
        # 7. 构建邻接矩阵
        
        self_current_pos = self_history_positions[:, -1, :].unsqueeze(1)  # [B, 1, 2]
        others_current_pos = others_positions[:, :, -1, :]  # [B, N-1, 2]
        all_positions = torch.cat([self_current_pos, others_current_pos], dim=1)  # [B, N_total, 2]
        print(f"gode position {self_history_positions.shape}, {others_current_pos.shape}, {all_positions.shape}")
        
        adj = self.build_adjacency_matrix(all_positions, self.comm_range)  # [B, N_total, N_total]
        self.gde_func.set_graph(adj)
        
        # 8. GDE演化
        # Flatten: [B, N_total, feature_dim] -> [B*N_total, feature_dim]
        node_features_flat = all_node_features.reshape(B * N_total, -1)
        
        # 设置初始状态
        self.gde_func.h0 = node_features_flat
        
        # ODE积分
        evolved_features = self.ode_block(node_features_flat, T=integration_time)
        
        # 9. Reshape回来,取自己的特征
        evolved_features = evolved_features.reshape(B, N_total, -1)
        self_evolved = evolved_features[:, 0, :]  # [B, hidden_dim]
        
        # 10. 投影回latent空间
        latent_features = self.feature_proj(self_evolved)  # [B, vae_latent_dim]
        
        # 11. 解码
        h = self.fc_decode(latent_features)
        h = h.view(B, 256, 8, 8)
        predicted_map = self.decoder(h)
        predicted_map = predicted_map.squeeze(1)  # [B, H, W]
        
        return predicted_map


def count_parameters(model):
    """统计模型参数"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen (VAE): {frozen:,}")
    
    return total, trainable


# 测试代码
if __name__ == "__main__":
    from src.extract.vae_module import PerceptionMapVAE
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建VAE
    vae = PerceptionMapVAE(map_size=(64, 64), latent_dim=32).to(device)
    
    # 创建GODE
    model = VehicleGODEPredictor(
        vae_encoder=vae,
        map_size=(16, 16),
        vae_latent_dim=64,
        temporal_hidden_dim=128,
        gnn_hidden_dim=256,
        num_gnn_layers=3
    ).to(device)
    
    count_parameters(model)
    
    # 测试
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    B, T, N_others, H, W = 2, 5, 4, 64, 64
    
    self_history_maps = torch.randn(B, T, H, W).to(device)
    self_history_positions = torch.randn(B, T, 2).to(device) * 100
    others_maps = torch.randn(B, N_others, T, H, W).to(device)
    others_positions = torch.randn(B, N_others, T, 2).to(device) * 100
    K_value = torch.randint(10, 150, (B, 1)).float().to(device)
    
    print(f"\nInput shapes:")
    print(f"  self_history_maps: {self_history_maps.shape}")
    print(f"  self_history_positions: {self_history_positions.shape}")
    print(f"  others_maps: {others_maps.shape}")
    print(f"  others_positions: {others_positions.shape}")
    print(f"  K_value: {K_value.shape}")
    
    with torch.no_grad():
        predicted_map = model(
            self_history_maps,
            self_history_positions,
            others_maps,
            others_positions,
            K_value
        )
    
    print(f"\nOutput shape:")
    print(f"  predicted_map: {predicted_map.shape}")
    
    assert predicted_map.shape == (B, H, W), "Output shape incorrect!"
    
    print("\n✓ Forward pass successful!")
    print("✓ Pure PyTorch GODE implementation working!")