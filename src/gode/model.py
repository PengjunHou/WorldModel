import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from typing import Dict, List, Tuple, Optional
from torchdiffeq import odeint_adjoint, odeint


class GCNLayer(nn.Module):
    """图卷积层"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.g = None  # 图结构将在forward前设置
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [N_nodes, in_dim] 节点特征
        Returns:
            [N_nodes, out_dim] 更新后的节点特征
        """
        if self.g is None:
            raise ValueError("Graph not set! Call set_graph() first")
        
        with self.g.local_scope():
            self.g.ndata['h'] = h
            # 消息传递: 邻居特征求和
            self.g.update_all(
                message_func=dgl.function.copy_u('h', 'm'),
                reduce_func=dgl.function.sum('m', 'h_neigh')
            )
            h_neigh = self.g.ndata['h_neigh']
            
            # 聚合自身特征和邻居特征
            h_total = h + h_neigh
            return F.relu(self.linear(h_total))


class GDEFunc(nn.Module):
    """GDE函数 - 定义节点特征的演化动力学"""
    def __init__(self, gnn: nn.ModuleList, vehicle_count: int):
        super().__init__()
        self.gnn = gnn
        self.vehicle_count = vehicle_count
        self.nfe = 0  # 函数评估次数计数器
    
    def set_graph(self, g: dgl.DGLGraph):
        """设置图结构"""
        for layer in self.gnn:
            layer.g = g
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        ODE右侧函数: dx/dt = f(t, x)
        
        Args:
            t: 时间 (标量)
            x: [N_vehicles, feature_dim] 当前状态
        Returns:
            dx/dt: [N_vehicles, feature_dim] 状态导数
        """
        self.nfe += 1
        for layer in self.gnn:
            x = layer(x)
        return x


class ControlledGDEFunc(GDEFunc):
    """受控GDE - 保留初始输入信息"""
    def __init__(self, gnn: nn.ModuleList, vehicle_count: int):
        super().__init__(gnn, vehicle_count)
        self.h0 = None  # 初始特征,在forward前设置
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        受控ODE: dx/dt = f(t, x, x0)
        通过拼接初始状态x0来保持输入信息
        """
        self.nfe += 1
        if self.h0 is None:
            raise ValueError("Initial state h0 not set!")
        
        # 拼接当前状态和初始状态
        x = torch.cat([x, self.h0], dim=1)
        for layer in self.gnn:
            x = layer(x)
        return x


class ODEBlock(nn.Module):
    """ODE求解器封装"""
    def __init__(
        self,
        odefunc: nn.Module,
        method: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        adjoint: bool = True
    ):
        """
        Args:
            odefunc: ODE函数
            method: 求解方法 {'euler', 'rk4', 'dopri5', 'adams'}
            rtol: 相对误差容忍度
            atol: 绝对误差容忍度
            adjoint: 是否使用伴随方法计算梯度
        """
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.adjoint_flag = adjoint
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, x: torch.Tensor, T: float = 1.0) -> torch.Tensor:
        """
        从t=0积分到t=T
        
        Args:
            x: [N_vehicles, feature_dim] 初始状态
            T: 积分时间
        Returns:
            [N_vehicles, feature_dim] 终态
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
        
        return out[-1]  # 返回终态
    
    def trajectory(self, x: torch.Tensor, T: float, num_points: int = 10) -> torch.Tensor:
        """
        返回完整轨迹
        
        Args:
            x: 初始状态
            T: 终止时间
            num_points: 轨迹采样点数
        Returns:
            [num_points, N_vehicles, feature_dim] 轨迹
        """
        integration_time = torch.linspace(0, T, num_points).type_as(x)
        out = odeint(
            self.odefunc, x, integration_time,
            rtol=self.rtol, atol=self.atol, method=self.method
        )
        
class GODEPredictor(nn.Module):
    """
    GODE预测器 - 用于车辆协同感知
    
    功能:
    1. 接收车队成员的region confidence scores
    2. 使用GDE预测融合后的confidence map
    3. 与实际融合对比,决定是否需要通信
    """
    def __init__(
        self,
        vehicle_count: int = 5,
        map_size: Tuple[int, int] = (128, 128),
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        ode_method: str = 'dopri5',
        use_controlled: bool = True
    ):
        """
        Args:
            vehicle_count: 车队车辆数量
            map_size: confidence map尺寸
            latent_dim: 潜在特征维度
            hidden_dim: GNN隐藏层维度
            num_gnn_layers: GNN层数
            ode_method: ODE求解方法
            use_controlled: 是否使用受控GDE
        """
        super().__init__()
        self.vehicle_count = vehicle_count
        self.map_size = map_size
        self.latent_dim = latent_dim
        self.use_controlled = use_controlled
        
        # Confidence map编码器/解码器
        self.encoder = ConfidenceMapEncoder(map_size, latent_dim)
        self.decoder = ConfidenceMapDecoder(latent_dim, map_size)
        
        # 位置编码 (将车辆位置嵌入到特征中)
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 64),  # (x, y) -> 64
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # GNN层
        if use_controlled:
            # 受控GDE需要双倍输入维度(拼接初始状态)
            gnn_layers = nn.ModuleList([
                GCNLayer(latent_dim * 2 if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_gnn_layers)
            ])
            self.gde_func = ControlledGDEFunc(gnn_layers, vehicle_count)
        else:
            gnn_layers = nn.ModuleList([
                GCNLayer(latent_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_gnn_layers)
            ])
            self.gde_func = GDEFunc(gnn_layers, vehicle_count)
        
        # ODE求解器
        self.ode_block = ODEBlock(self.gde_func, method=ode_method)
        
        # 特征映射回latent_dim
        self.feature_proj = nn.Linear(hidden_dim, latent_dim)
    
    def build_communication_graph(
        self,
        positions: torch.Tensor,
        comm_range: float = 100.0
    ) -> dgl.DGLGraph:
        """
        构建通信图 - 基于车辆位置和通信范围
        
        Args:
            positions: [N_vehicles, 2] 车辆位置 (x, y)
            comm_range: 通信范围(米)
        Returns:
            DGLGraph
        """
        N = positions.shape[0]
        
        # 计算距离矩阵
        pos_expanded = positions.unsqueeze(1)  # [N, 1, 2]
        pos_tiled = positions.unsqueeze(0)     # [1, N, 2]
        distances = torch.norm(pos_expanded - pos_tiled, dim=2)  # [N, N]
        
        # 构建邻接矩阵 (距离在通信范围内)
        adj_matrix = (distances <= comm_range).float()
        # 移除自环
        adj_matrix = adj_matrix - torch.eye(N).to(positions.device)
        
        # 构建DGL图
        edge_index = torch.nonzero(adj_matrix, as_tuple=True)
        src, dst = edge_index[0], edge_index[1]
        
        g = dgl.graph((src, dst), num_nodes=N)
        g = g.to(positions.device)
        
        return g
    
    def forward(
        self,
        local_conf_maps: torch.Tensor,
        positions: torch.Tensor,
        received_regions: Optional[Dict[int, List[Tuple[int, float]]]] = None,
        comm_range: float = 100.0,
        integration_time: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测融合后的confidence maps
        
        Args:
            local_conf_maps: [N_vehicles, 1, H, W] 各车辆本地confidence map
            positions: [N_vehicles, 2] 车辆位置
            received_regions: 收到的区域信息 {vehicle_id: [(region_id, score), ...]}
            comm_range: 通信范围
            integration_time: ODE积分时间
            
        Returns:
            predicted_fused_maps: [N_vehicles, 1, H, W] 预测的融合map
            latent_features: [N_vehicles, latent_dim] 潜在特征
        """
        device = local_conf_maps.device
        N = local_conf_maps.shape[0]
        
        # 1. 编码local confidence maps
        local_features = self.encoder(local_conf_maps)  # [N, latent_dim]
        
        # 2. 编码位置信息
        pos_features = self.pos_encoder(positions)  # [N, latent_dim]
        
        # 3. 融合局部感知和位置信息
        node_features = local_features + pos_features  # [N, latent_dim]
        
        # 4. 构建通信图
        comm_graph = self.build_communication_graph(positions, comm_range)
        self.gde_func.set_graph(comm_graph)
        
        # 5. 如果使用受控GDE,设置初始状态
        if self.use_controlled:
            self.gde_func.h0 = node_features
        
        # 6. 通过GDE演化特征
        evolved_features = self.ode_block(node_features, T=integration_time)  # [N, hidden_dim]
        
        # 7. 投影回latent空间
        latent_features = self.feature_proj(evolved_features)  # [N, latent_dim]
        
        # 8. 解码为confidence maps
        predicted_fused_maps = self.decoder(latent_features)  # [N, 1, H, W]
        
        return predicted_fused_maps, latent_features
    
    def predict_communication_gain(
        self,
        local_conf_map: torch.Tensor,
        predicted_fused_map: torch.Tensor,
        threshold: float = 0.05
    ) -> Tuple[bool, float]:
        """
        预测通信是否有价值
        
        Args:
            local_conf_map: [1, H, W] 本地map
            predicted_fused_map: [1, H, W] 预测的融合map
            threshold: 决策阈值
            
        Returns:
            should_communicate: 是否应该通信
            gain: 预期增益
        """
        # 计算预期增益 (融合map与本地map的差异)
        diff = predicted_fused_map - local_conf_map
        gain = torch.mean(torch.abs(diff)).item()
        
        # 如果增益大于阈值,则通信
        should_communicate = gain > threshold
        
        return should_communicate, gain
    
    def get_trajectory(
        self,
        local_conf_maps: torch.Tensor,
        positions: torch.Tensor,
        comm_range: float = 100.0,
        T: float = 1.0,
        num_points: int = 10
    ) -> torch.Tensor:
        """
        获取特征演化轨迹(用于可视化和分析)
        
        Returns:
            [num_points, N_vehicles, latent_dim] 轨迹
        """
        device = local_conf_maps.device
        
        # 编码
        local_features = self.encoder(local_conf_maps)
        pos_features = self.pos_encoder(positions)
        node_features = local_features + pos_features
        
        # 构建图
        comm_graph = self.build_communication_graph(positions, comm_range)
        self.gde_func.set_graph(comm_graph)
        
        if self.use_controlled:
            self.gde_func.h0 = node_features
        
        # 获取轨迹
        trajectory = self.ode_block.trajectory(node_features, T, num_points)
        
        return trajectory
