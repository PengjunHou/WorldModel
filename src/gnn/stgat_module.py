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
        dropout: float = 0.1,
        use_edge_weights: bool = True  # 是否使用边权重
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_edge_weights = use_edge_weights
        
        # ====== 空间注意力层（每个时间步内） ======
        self.spatial_gat_layers = nn.ModuleList()
        
        # 第一层
        self.spatial_gat_layers.append(
            GATConv(
                in_channels=feature_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True,  # 拼接多头
                edge_dim=1 if use_edge_weights else None  # 支持边权重
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
                    concat=True,
                    edge_dim=1 if use_edge_weights else None  # 支持边权重
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
        edge_indices_history: List[torch.Tensor],
        edge_weights_history: List[torch.Tensor] = None  # 新增：边权重历史
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features_history: [N_v, T, feature_dim] 每辆车每个时间步的特征
            edge_indices_history: List[edge_index]，长度为T，每个是[2, E_t]
            edge_weights_history: List[edge_weights]，长度为T，每个是[E_t]（可选）
        
        Returns:
            final_embeddings: [N_v, hidden_dim] 每辆车最终的时空嵌入
        """
        N_v, T, _ = features_history.shape
        
        # 如果没有提供边权重但模型要求使用，发出警告
        if self.use_edge_weights and edge_weights_history is None:
            import warnings
            warnings.warn("use_edge_weights=True但未提供edge_weights_history，将不使用边权重")
        
        # ====== 步骤1：空间建模（每个时间步独立） ======
        spatial_embeddings = []
        
        for t in range(T):
            # 当前时间步的特征和边
            x_t = features_history[:, t, :]  # [N_v, feature_dim]
            edge_index_t = edge_indices_history[t]  # [2, E_t]
            
            # 获取边权重（如果有）
            if self.use_edge_weights and edge_weights_history is not None:
                edge_attr_t = edge_weights_history[t].unsqueeze(-1)  # [E_t, 1]
            else:
                edge_attr_t = None
            
            # 多层GAT
            h = x_t
            for gat_layer, layer_norm in zip(self.spatial_gat_layers, self.spatial_layer_norms):
                # GAT前向（传入边权重）
                if self.use_edge_weights and edge_attr_t is not None:
                    h_new = gat_layer(h, edge_index_t, edge_attr=edge_attr_t)
                else:
                    h_new = gat_layer(h, edge_index_t)
                
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
    构建通信边的工具类（支持距离加权）
    """
    @staticmethod
    def build_communication_edges(
        positions: torch.Tensor,
        comm_range: float,
        return_weights: bool = False,
        weight_type: str = 'linear'
    ) -> tuple:
        """
        根据车辆位置构建通信边（支持边权重）
        
        Args:
            positions: [N_v, 2] 车辆位置
            comm_range: float 通信范围（米）
            return_weights: bool 是否返回边权重
            weight_type: str 权重计算方式
                - 'linear': 线性衰减，weight = 1 - distance/comm_range
                - 'exp': 指数衰减，weight = exp(-distance/sigma)
                - 'inverse': 反比例，weight = 1 / (1 + distance)
                - 'gaussian': 高斯核，weight = exp(-distance^2 / (2*sigma^2))
        
        Returns:
            如果 return_weights=False:
                edge_index: [2, E] 边索引
            如果 return_weights=True:
                (edge_index, edge_weights): 边索引和权重
                edge_index: [2, E] 
                edge_weights: [E] 范围0-1，距离越近权重越大
        """
        N_v = positions.shape[0]
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(positions, positions)
        
        # 在通信范围内且不是自己
        adj_matrix = (dist_matrix < comm_range) & (dist_matrix > 0)
        
        # 转换为边索引
        edge_index = adj_matrix.nonzero().t()
        
        if not return_weights:
            return edge_index
        
        # 计算边权重
        edge_distances = dist_matrix[edge_index[0], edge_index[1]]
        edge_weights = EdgeBuilder._compute_edge_weights(
            edge_distances, 
            comm_range,
            weight_type
        )
        
        return edge_index, edge_weights
    
    @staticmethod
    def _compute_edge_weights(
        distances: torch.Tensor,
        comm_range: float,
        weight_type: str = 'linear'
    ) -> torch.Tensor:
        """
        根据距离计算边权重
        
        Args:
            distances: [E] 边的距离
            comm_range: float 通信范围
            weight_type: str 权重类型
        
        Returns:
            weights: [E] 边权重（0-1之间，距离越近权重越大）
        """
        if weight_type == 'linear':
            # 线性衰减: weight = 1 - (distance / comm_range)
            # 距离0时权重=1，距离=comm_range时权重→0
            weights = 1.0 - (distances / comm_range)
            weights = torch.clamp(weights, min=0.0, max=1.0)
        
        elif weight_type == 'exp':
            # 指数衰减: weight = exp(-distance / sigma)
            sigma = comm_range / 3.0  # 3-sigma规则
            weights = torch.exp(-distances / sigma)
        
        elif weight_type == 'inverse':
            # 反比例: weight = 1 / (1 + distance)
            weights = 1.0 / (1.0 + distances)
            # 归一化到[0, 1]
            max_weight = 1.0 / 1.0  # 距离=0时的最大值
            weights = weights / max_weight
        
        elif weight_type == 'gaussian':
            # 高斯核: weight = exp(-distance^2 / (2*sigma^2))
            sigma = comm_range / 3.0
            weights = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
        
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        return weights


