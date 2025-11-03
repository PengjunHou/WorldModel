"""
GNN模型架构：异构图神经网络用于Q值预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, GCNConv, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, Optional


class VehicleEncoder(nn.Module):
    """车辆特征编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RegionEncoder(nn.Module):
    """区域特征编码器"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HeteroGNNLayer(nn.Module):
    """异构图卷积层"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 定义不同类型边的卷积操作
        self.convs = HeteroConv({
            # 车辆-车辆通信
            ('vehicle', 'communicates', 'vehicle'): GATConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout, add_self_loops=True
            ),
            # 区域-区域邻接
            ('region', 'adjacent', 'region'): GATConv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, dropout=dropout, add_self_loops=True
            ),
            # 车辆感知区域
            ('vehicle', 'perceives', 'region'): GATConv(
                (hidden_dim, hidden_dim), hidden_dim // num_heads,
                heads=num_heads, dropout=dropout
            ),
            # 区域被车辆感兴趣
            ('region', 'interests', 'vehicle'): GATConv(
                (hidden_dim, hidden_dim), hidden_dim // num_heads,
                heads=num_heads, dropout=dropout
            ),
        }, aggr='mean')
        
        self.layer_norm = nn.ModuleDict({
            'vehicle': nn.LayerNorm(hidden_dim),
            'region': nn.LayerNorm(hidden_dim)
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[tuple, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典 {'vehicle': [N_v, D], 'region': [N_r, D]}
            edge_index_dict: 边索引字典
        Returns:
            更新后的节点特征字典
        """
        # 异构图卷积
        x_dict_new = self.convs(x_dict, edge_index_dict)
        
        # 残差连接 + LayerNorm
        for node_type in x_dict.keys():
            if node_type in x_dict_new:
                x_dict_new[node_type] = x_dict[node_type] + self.dropout(x_dict_new[node_type])
                x_dict_new[node_type] = self.layer_norm[node_type](x_dict_new[node_type])
        
        return x_dict_new


class QNetworkGNN(nn.Module):
    """
    基于GNN的Q网络
    输入：车辆和区域的异构图
    输出：每个区域的Q值
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ============ 编码器 ============
        # 计算输入维度（需要根据graph_builder中的特征维度调整）
        vehicle_input_dim = (
            2 +  # 位置
            2 +  # 速度
            2 +  # 朝向(sin/cos)
            config.GRID_SIZE * config.GRID_SIZE  # 感知地图展平
        )
        
        region_input_dim = (
            1 +  # confidence
            1 +  # interest_counts
            2 +  # 位置
            config.SEMANTIC_MAP_CHANNELS  # 语义特征
        )
        
        self.vehicle_encoder = VehicleEncoder(
            vehicle_input_dim,
            config.GNN_HIDDEN_DIM,
            config.GNN_HIDDEN_DIM
        )
        
        self.region_encoder = RegionEncoder(
            region_input_dim,
            config.GNN_HIDDEN_DIM // 2,
            config.GNN_HIDDEN_DIM
        )
        
        # ============ GNN层 ============
        self.gnn_layers = nn.ModuleList([
            HeteroGNNLayer(
                config.GNN_HIDDEN_DIM,
                config.NUM_ATTENTION_HEADS,
                config.DROPOUT_RATE
            )
            for _ in range(config.GNN_NUM_LAYERS)
        ])
        
        # ============ Q值预测头 ============
        self.q_predictor = nn.Sequential(
            nn.Linear(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM // 2),
            nn.LayerNorm(config.GNN_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.GNN_HIDDEN_DIM // 2, config.GNN_HIDDEN_DIM // 4),
            nn.ReLU(),
            nn.Linear(config.GNN_HIDDEN_DIM // 4, 1)  # 输出单个Q值
        )
        
        # ============ 时间建模（可选）============
        self.temporal_rnn = nn.GRU(
            config.GNN_HIDDEN_DIM,
            config.GNN_HIDDEN_DIM,
            batch_first=True
        )
    
    def forward(
        self,
        data: HeteroData,
        return_embeddings: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: HeteroData图对象
            return_embeddings: 是否返回中间嵌入（用于可视化）
        Returns:
            q_values: [N_r] 每个区域的Q值
        """
        # ============ 初始编码 ============
        x_dict = {
            'vehicle': self.vehicle_encoder(data['vehicle'].x),
            'region': self.region_encoder(data['region'].x)
        }
        
        embeddings_history = []
        
        # ============ 多层GNN传播 ============
        for gnn_layer in self.gnn_layers:
            x_dict = gnn_layer(x_dict, data.edge_index_dict)
            embeddings_history.append(x_dict['region'].clone())
        
        # ============ 提取区域嵌入 ============
        region_embeddings = x_dict['region']  # [N_r, hidden_dim]
        
        # ============ 预测Q值 ============
        q_values = self.q_predictor(region_embeddings).squeeze(-1)  # [N_r]
        
        if return_embeddings:
            return q_values, {
                'region_embeddings': region_embeddings,
                'vehicle_embeddings': x_dict['vehicle'],
                'embeddings_history': embeddings_history
            }
        
        return q_values
    
    def compute_whittle_index(
        self,
        q_values: torch.Tensor,
        confidence_current: torch.Tensor,
        confidence_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        根据Q值计算Whittle Index
        
        Args:
            q_values: [N_r] Q值
            confidence_current: [N_r] 当前时刻的confidence
            confidence_prev: [N_r] 上一时刻的confidence
        Returns:
            whittle_indices: [N_r] Whittle Index值
        """
        beta = self.config.DISCOUNT_FACTOR
        
        # C_t = (r_t - β*r_{t-1}) * Q_t
        confidence_diff = confidence_current - beta * confidence_prev
        whittle_indices = confidence_diff * q_values
        
        return whittle_indices


class DuelingQNetwork(nn.Module):
    """
    Dueling架构的Q网络（可选的改进版本）
    将Q值分解为状态值V和优势函数A
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 共享的GNN backbone
        self.backbone = QNetworkGNN(config)
        
        # 移除原始的q_predictor
        self.backbone.q_predictor = nn.Identity()
        
        # 状态值流
        self.value_stream = nn.Sequential(
            nn.Linear(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.GNN_HIDDEN_DIM // 2, 1)
        )
        
        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.GNN_HIDDEN_DIM, config.GNN_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(config.GNN_HIDDEN_DIM // 2, 1)
        )
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Dueling架构前向传播
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        # 获取区域嵌入
        region_embeddings = self.backbone.region_encoder(data['region'].x)
        
        # 通过GNN层
        x_dict = {'region': region_embeddings, 'vehicle': self.backbone.vehicle_encoder(data['vehicle'].x)}
        for gnn_layer in self.backbone.gnn_layers:
            x_dict = gnn_layer(x_dict, data.edge_index_dict)
        
        region_embeddings = x_dict['region']
        
        # 计算V和A
        values = self.value_stream(region_embeddings).squeeze(-1)  # [N_r]
        advantages = self.advantage_stream(region_embeddings).squeeze(-1)  # [N_r]
        
        # 组合得到Q值
        q_values = values + (advantages - advantages.mean())
        
        return q_values


class EnsembleQNetwork(nn.Module):
    """
    集成Q网络（用于减少估计方差）
    """
    
    def __init__(self, config, num_ensemble: int = 3):
        super().__init__()
        self.config = config
        self.num_ensemble = num_ensemble
        
        # 创建多个Q网络
        self.q_networks = nn.ModuleList([
            QNetworkGNN(config) for _ in range(num_ensemble)
        ])
    
    def forward(self, data: HeteroData, use_mean: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: 输入图
            use_mean: 是否使用平均值（推理时），训练时返回所有值
        Returns:
            q_values: [N_r] 或 [num_ensemble, N_r]
        """
        q_values_list = [net(data) for net in self.q_networks]
        
        if use_mean:
            return torch.stack(q_values_list).mean(dim=0)
        else:
            return torch.stack(q_values_list)


def create_model(config, model_type: str = 'standard'):
    """
    工厂函数：创建指定类型的模型
    
    Args:
        config: 配置对象
        model_type: 模型类型 ('standard', 'dueling', 'ensemble')
    Returns:
        model: 创建的模型
    """
    if model_type == 'standard':
        return QNetworkGNN(config)
    elif model_type == 'dueling':
        return DuelingQNetwork(config)
    elif model_type == 'ensemble':
        return EnsembleQNetwork(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
