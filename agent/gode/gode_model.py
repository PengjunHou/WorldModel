"""
GODE-based Cooperative Perception Model for Autonomous Vehicles
基于图神经ODE的协同感知模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from typing import Dict, Tuple, Optional
import torchdiffeq


class ConfidenceMapEncoder(nn.Module):
    """
    将置信度地图(Confidence Map)编码为特征向量
    """
    def __init__(self, map_size: Tuple[int, int] = (100, 100), 
                 hidden_dim: int = 128, 
                 feature_dim: int = 64):
        super().__init__()
        self.map_size = map_size
        
        # CNN提取空间特征
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # -> 50x50
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> 25x25
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> 13x13
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> 4x4
        )
        
        # 全连接层压缩特征
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, conf_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conf_map: [batch, H, W] 或 [batch, 1, H, W]
        Returns:
            features: [batch, feature_dim]
        """
        if conf_map.dim() == 3:
            conf_map = conf_map.unsqueeze(1)  # [batch, 1, H, W]
            
        x = self.conv_layers(conf_map)  # [batch, 128, 4, 4]
        x = x.view(x.size(0), -1)  # [batch, 128*4*4]
        x = self.fc(x)  # [batch, feature_dim]
        return x


class VehicleStateEncoder(nn.Module):
    """
    编码车辆状态(位置、速度、方向等)
    """
    def __init__(self, state_dim: int = 6, hidden_dim: int = 32, feature_dim: int = 32):
        super().__init__()
        # state: [x, y, vx, vy, heading, speed]
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch, state_dim]
        Returns:
            features: [batch, feature_dim]
        """
        return self.fc(state)


class GCNLayer(nn.Module):
    """
    图卷积层,用于GODE
    """
    def __init__(self, in_feats: int, out_feats: int, 
                 activation=None, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: DGL graph
            h: [num_nodes, in_feats]
        Returns:
            h_out: [num_nodes, out_feats]
        """
        if self.dropout:
            h = self.dropout(h)
            
        # 特征变换
        h = torch.mm(h, self.weight)
        
        # 归一化
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).unsqueeze(1).to(h.device)
        h = h * norm
        
        # 消息传递
        g.ndata['h'] = h
        g.update_all(
            message_func=dgl.function.copy_u('h', 'm'),
            reduce_func=dgl.function.sum('m', 'h_agg')
        )
        h_agg = g.ndata.pop('h_agg')
        
        # 再次归一化
        h_agg = h_agg * norm
        
        # 加bias
        h_out = h_agg + self.bias
        
        if self.activation:
            h_out = self.activation(h_out)
            
        return h_out


class GCNEncoder(nn.Module):
    """
    多层GCN编码器
    """
    def __init__(self, num_layers: int, in_feats: int, 
                 hidden_feats: int, out_feats: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(
            GCNLayer(in_feats, hidden_feats, activation=F.relu, dropout=dropout)
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                GCNLayer(hidden_feats, hidden_feats, activation=F.relu, dropout=dropout)
            )
        
        # 输出层
        self.layers.append(
            GCNLayer(hidden_feats, out_feats, activation=None, dropout=0.0)
        )
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(g, h)
        return h


class GDEFunc(nn.Module):
    """
    Graph Differential Equation Function
    定义图上的连续动力学: dh/dt = GNN(h, G)
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gnn = GCNEncoder(
            num_layers=num_layers,
            in_feats=hidden_dim,
            hidden_feats=hidden_dim,
            out_feats=hidden_dim,
            dropout=dropout
        )
        self.nfe = 0  # Number of function evaluations
        self.graph = None
        
    def set_graph(self, g: dgl.DGLGraph):
        """设置图结构"""
        self.graph = g
        
    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        """
        ODE右侧函数
        Args:
            t: 时间
            h: [num_nodes, hidden_dim] 节点隐状态
        Returns:
            dh_dt: [num_nodes, hidden_dim] 状态导数
        """
        self.nfe += 1
        return self.gnn(self.graph, h)


class ControlledGDEFunc(GDEFunc):
    """
    Controlled GDE: 保持初始输入信息的影响
    dh/dt = GNN([h, h0], G)
    """
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        # 输入维度加倍,因为要concat h0
        super().__init__(hidden_dim, num_layers, dropout)
        self.gnn = GCNEncoder(
            num_layers=num_layers,
            in_feats=hidden_dim * 2,  # 拼接当前状态和初始状态
            hidden_feats=hidden_dim,
            out_feats=hidden_dim,
            dropout=dropout
        )
        self.h0 = None
        
    def set_initial_state(self, h0: torch.Tensor):
        """设置初始状态h0"""
        self.h0 = h0
        
    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: 时间
            h: [num_nodes, hidden_dim]
        Returns:
            dh_dt: [num_nodes, hidden_dim]
        """
        self.nfe += 1
        # 拼接当前状态和初始状态
        h_concat = torch.cat([h, self.h0], dim=1)  # [num_nodes, hidden_dim*2]
        return self.gnn(self.graph, h_concat)


class ODEBlock(nn.Module):
    """
    ODE求解器模块
    """
    def __init__(self, odefunc: nn.Module, method: str = 'dopri5',
                 rtol: float = 1e-3, atol: float = 1e-4, adjoint: bool = True):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.rtol = rtol
        self.atol = atol
        
    def forward(self, h0: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        求解ODE从t=0到t=T
        Args:
            h0: [num_nodes, hidden_dim] 初始状态
            T: 终止时间
        Returns:
            h_T: [num_nodes, hidden_dim] 时间T的状态
        """
        integration_time = torch.tensor([0.0, T]).float().to(h0.device)
        
        if self.adjoint_flag:
            out = torchdiffeq.odeint_adjoint(
                self.odefunc, h0, integration_time,
                rtol=self.rtol, atol=self.atol, method=self.method
            )
        else:
            out = torchdiffeq.odeint(
                self.odefunc, h0, integration_time,
                rtol=self.rtol, atol=self.atol, method=self.method
            )
        
        return out[-1]  # 返回最后时刻的状态


class ConfidenceMapDecoder(nn.Module):
    """
    将隐状态解码回置信度地图
    """
    def __init__(self, hidden_dim: int, map_size: Tuple[int, int] = (100, 100)):
        super().__init__()
        self.map_size = map_size
        
        # 先用FC扩展维度
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 4 * 4)
        )
        
        # 反卷积恢复空间分辨率
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),     # 64x64 -> 128x128
            nn.Sigmoid()  # 输出[0,1]范围
        )
        
        # 最后调整到目标尺寸
        self.final_resize = nn.AdaptiveAvgPool2d(map_size)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, hidden_dim]
        Returns:
            conf_map: [batch, H, W]
        """
        x = self.fc(h)  # [batch, 128*4*4]
        x = x.view(-1, 128, 4, 4)  # [batch, 128, 4, 4]
        x = self.deconv_layers(x)  # [batch, 1, 128, 128]
        x = self.final_resize(x)  # [batch, 1, H, W]
        return x.squeeze(1)  # [batch, H, W]


class GODEVehicleModel(nn.Module):
    """
    完整的GODE车辆模型
    输入: 本地confidence map + 车辆状态 + 邻居信息(通过图传递)
    输出: 预测的future confidence map
    """
    def __init__(self, 
                 map_size: Tuple[int, int] = (100, 100),
                 state_dim: int = 6,
                 hidden_dim: int = 128,
                 num_gnn_layers: int = 2,
                 use_controlled_gde: bool = True,
                 ode_method: str = 'dopri5'):
        super().__init__()
        
        self.map_size = map_size
        self.hidden_dim = hidden_dim
        self.use_controlled_gde = use_controlled_gde
        
        # 编码器
        self.map_encoder = ConfidenceMapEncoder(
            map_size=map_size, 
            hidden_dim=128, 
            feature_dim=64
        )
        self.state_encoder = VehicleStateEncoder(
            state_dim=state_dim, 
            hidden_dim=32, 
            feature_dim=32
        )
        
        # 融合层: map_feature(64) + state_feature(32) -> hidden_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(64 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GDE核心
        if use_controlled_gde:
            self.gde_func = ControlledGDEFunc(
                hidden_dim=hidden_dim,
                num_layers=num_gnn_layers,
                dropout=0.1
            )
        else:
            self.gde_func = GDEFunc(
                hidden_dim=hidden_dim,
                num_layers=num_gnn_layers,
                dropout=0.1
            )
        
        self.ode_block = ODEBlock(
            odefunc=self.gde_func,
            method=ode_method,
            rtol=1e-3,
            atol=1e-4,
            adjoint=True
        )
        
        # 解码器
        self.map_decoder = ConfidenceMapDecoder(
            hidden_dim=hidden_dim,
            map_size=map_size
        )
        
    def forward(self, 
                local_maps: torch.Tensor,
                vehicle_states: torch.Tensor,
                graph: dgl.DGLGraph,
                prediction_horizon: float = 1.0) -> torch.Tensor:
        """
        前向传播
        Args:
            local_maps: [num_vehicles, H, W] 每辆车的本地置信度地图
            vehicle_states: [num_vehicles, state_dim] 车辆状态
            graph: DGLGraph 车辆间的通信拓扑
            prediction_horizon: 预测时间范围 (秒)
        Returns:
            pred_maps: [num_vehicles, H, W] 预测的置信度地图
        """
        num_vehicles = local_maps.size(0)
        
        # 1. 编码
        map_features = self.map_encoder(local_maps)  # [num_vehicles, 64]
        state_features = self.state_encoder(vehicle_states)  # [num_vehicles, 32]
        
        # 2. 融合
        features = torch.cat([map_features, state_features], dim=1)  # [num_vehicles, 96]
        h0 = self.fusion_fc(features)  # [num_vehicles, hidden_dim]
        
        # 3. 设置图和初始状态
        self.gde_func.set_graph(graph)
        if self.use_controlled_gde:
            self.gde_func.set_initial_state(h0)
        
        # 4. 通过ODE求解演化
        h_pred = self.ode_block(h0, T=prediction_horizon)  # [num_vehicles, hidden_dim]
        
        # 5. 解码为地图
        pred_maps = self.map_decoder(h_pred)  # [num_vehicles, H, W]
        
        return pred_maps
    
    def get_nfe(self):
        """获取ODE求解器的函数评估次数"""
        return self.gde_func.nfe
    
    def reset_nfe(self):
        """重置NFE计数器"""
        self.gde_func.nfe = 0


class ActionDecisionModule(nn.Module):
    """
    基于预测的confidence map做出加速/减速决策
    """
    def __init__(self, map_size: Tuple[int, int] = (100, 100), hidden_dim: int = 64):
        super().__init__()
        
        # 地图特征提取
        self.map_encoder = ConfidenceMapEncoder(
            map_size=map_size,
            hidden_dim=128,
            feature_dim=hidden_dim
        )
        
        # 决策网络: 输出连续的加速度值
        self.decision_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
    def forward(self, pred_map: torch.Tensor, 
                current_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_map: [batch, H, W] 预测的confidence map
            current_map: [batch, H, W] 当前的confidence map (可选,用于对比)
        Returns:
            action: [batch, 1] 加速度决策,范围[-1, 1]
                   正值表示加速,负值表示减速
        """
        # 如果提供了当前地图,可以计算差异
        if current_map is not None:
            # 计算预测地图与当前地图的差异
            map_diff = pred_map - current_map
            # 可以同时考虑预测地图和差异
            features = self.map_encoder(pred_map)
            diff_features = self.map_encoder(map_diff)
            combined = features + 0.5 * diff_features
        else:
            features = self.map_encoder(pred_map)
            combined = features
        
        # 决策
        action = self.decision_net(combined)
        
        return action


# ============ 工具函数 ============

def build_vehicle_graph(positions: np.ndarray, 
                       max_comm_range: float = 200.0,
                       adjacency_matrix: Optional[np.ndarray] = None) -> dgl.DGLGraph:
    """
    根据车辆位置和通信范围构建通信图
    
    Args:
        positions: [num_vehicles, 2] 车辆位置(x, y)
        max_comm_range: 最大通信距离
        adjacency_matrix: [num_vehicles, num_vehicles] 预定义的邻接矩阵(可选)
    Returns:
        graph: DGLGraph
    """
    num_vehicles = len(positions)
    
    if adjacency_matrix is not None:
        # 使用预定义的邻接矩阵
        edges_src, edges_dst = np.where(adjacency_matrix > 0)
    else:
        # 根据距离构建
        edges_src = []
        edges_dst = []
        
        for i in range(num_vehicles):
            for j in range(num_vehicles):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= max_comm_range:
                        edges_src.append(i)
                        edges_dst.append(j)
    
    # 创建图
    graph = dgl.graph((edges_src, edges_dst), num_nodes=num_vehicles)
    
    # 添加自环 (重要!)
    graph = dgl.add_self_loop(graph)
    
    return graph


def compute_map_quality(pred_map: torch.Tensor, 
                        gt_map: torch.Tensor,
                        metric: str = 'mse') -> torch.Tensor:
    """
    计算预测地图的质量
    
    Args:
        pred_map: [batch, H, W] 预测地图
        gt_map: [batch, H, W] 真实地图
        metric: 'mse', 'mae', 或 'iou'
    Returns:
        quality: [batch] 质量分数
    """
    if metric == 'mse':
        return F.mse_loss(pred_map, gt_map, reduction='none').mean(dim=[1, 2])
    elif metric == 'mae':
        return F.l1_loss(pred_map, gt_map, reduction='none').mean(dim=[1, 2])
    elif metric == 'iou':
        # 将confidence map二值化后计算IoU
        pred_binary = (pred_map > 0.5).float()
        gt_binary = (gt_map > 0.5).float()
        intersection = (pred_binary * gt_binary).sum(dim=[1, 2])
        union = (pred_binary + gt_binary).clamp(max=1).sum(dim=[1, 2])
        return intersection / (union + 1e-6)
    else:
        raise ValueError(f"Unknown metric: {metric}")
