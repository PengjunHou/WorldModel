"""
时空图注意力网络（ST-GAT）模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import List


class SpatioTemporalGAT(nn.Module):
    """
    时空图注意力网络
    先在每个时间步内做空间GAT，再跨时间步做时间Transformer
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # ====== 空间注意力层（每个时间步内） ======
        self.spatial_gat_layers = nn.ModuleList()
        
        # 第一层
        self.spatial_gat_layers.append(
            GATConv(
                in_channels=feature_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True  # 拼接多头
            )
        )
        
        # 后续层
        for _ in range(num_spatial_layers - 1):
            self.spatial_gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
        
        # 空间层的LayerNorm
        self.spatial_layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_spatial_layers)
        ])
        
        # ====== 时间注意力层（跨时间步） ======
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # [T, N_v, D]格式
        )
        
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features_history: torch.Tensor,
        edge_indices_history: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features_history: [N_v, T, feature_dim] 每辆车每个时间步的特征
            edge_indices_history: List[edge_index]，长度为T，每个是[2, E_t]
        
        Returns:
            final_embeddings: [N_v, hidden_dim] 每辆车最终的时空嵌入
        """
        N_v, T, _ = features_history.shape
        
        # ====== 步骤1：空间建模（每个时间步独立） ======
        spatial_embeddings = []
        
        for t in range(T):
            # 当前时间步的特征和边
            x_t = features_history[:, t, :]  # [N_v, feature_dim]
            edge_index_t = edge_indices_history[t]  # [2, E_t]
            
            # 多层GAT
            h = x_t
            for gat_layer, layer_norm in zip(self.spatial_gat_layers, self.spatial_layer_norms):
                # GAT前向
                h_new = gat_layer(h, edge_index_t)  # [N_v, hidden_dim]
                
                # 残差连接（如果维度匹配）
                if h.shape[-1] == h_new.shape[-1]:
                    h = layer_norm(h + self.dropout(h_new))
                else:
                    h = layer_norm(h_new)
            
            spatial_embeddings.append(h)
        
        # 堆叠：[N_v, T, hidden_dim]
        spatial_embeddings = torch.stack(spatial_embeddings, dim=1)
        
        # ====== 步骤2：时间建模（每辆车独立） ======
        # 转换为Transformer所需的格式：[T, N_v, hidden_dim]
        spatial_embeddings = spatial_embeddings.transpose(0, 1)
        
        # Transformer编码时序依赖
        temporal_output = self.temporal_transformer(spatial_embeddings)  # [T, N_v, hidden_dim]
        
        # 取最后一个时间步的输出
        final_embeddings = temporal_output[-1]  # [N_v, hidden_dim]
        
        return final_embeddings


class EdgeBuilder:
    """
    构建通信边的工具类
    """
    @staticmethod
    def build_communication_edges(
        positions: torch.Tensor,
        comm_range: float
    ) -> torch.Tensor:
        """
        根据车辆位置构建通信边
        
        Args:
            positions: [N_v, 2] 车辆位置
            comm_range: float 通信范围
        
        Returns:
            edge_index: [2, E] 边索引
        """
        N_v = positions.shape[0]
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(positions, positions)
        
        # 在通信范围内且不是自己
        adj_matrix = (dist_matrix < comm_range) & (dist_matrix > 0)
        
        # 转换为边索引
        edge_index = adj_matrix.nonzero().t()
        
        return edge_index
