"""
GODE Environment Wrapper for CARLA
将GODE模型集成到CARLA通信环境中
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from collections import defaultdict

from .gode import (
    GODEVehicleModel, 
    ActionDecisionModule,
    build_vehicle_graph,
    compute_map_quality
)


class GODEVehicleAgent:
    """
    单个车辆的GODE智能体
    每辆车部署一个独立的GODE网络
    """
    def __init__(self, 
                 vehicle_id: int,
                 model: GODEVehicleModel,
                 action_module: ActionDecisionModule,
                 device: str = 'cuda'):
        self.vehicle_id = vehicle_id
        self.model = model.to(device)
        self.action_module = action_module.to(device)
        self.device = device
        
        # 历史信息缓存
        self.local_map_history = []
        self.state_history = []
        self.received_neighbor_info = {}
        
        # 预测缓存
        self.last_prediction_time = 0
        self.cached_prediction = None
        
    def update_local_observation(self, local_map: np.ndarray, vehicle_state: np.ndarray):
        """
        更新本地观测
        Args:
            local_map: [H, W] 本地置信度地图
            vehicle_state: [state_dim] 车辆状态 [x, y, vx, vy, heading, speed]
        """
        self.local_map_history.append(local_map)
        self.state_history.append(vehicle_state)
        
        # 只保留最近的N个历史
        max_history = 10
        if len(self.local_map_history) > max_history:
            self.local_map_history.pop(0)
            self.state_history.pop(0)
    
    def receive_neighbor_info(self, neighbor_id: int, neighbor_data: Dict):
        """
        接收邻居车辆的信息
        Args:
            neighbor_id: 邻居车辆ID
            neighbor_data: {'map': array, 'state': array, 'timestamp': float}
        """
        self.received_neighbor_info[neighbor_id] = neighbor_data
    
    def get_shareable_info(self) -> Dict:
        """
        获取可分享的信息
        Returns:
            info: {'map': array, 'state': array, 'timestamp': float}
        """
        if len(self.local_map_history) == 0:
            return None
        
        return {
            'map': self.local_map_history[-1],
            'state': self.state_history[-1],
            'timestamp': len(self.local_map_history)  # 简化的时间戳
        }
    
    def predict_future_map(self, 
                          graph: dgl.DGLGraph,
                          all_local_maps: torch.Tensor,
                          all_states: torch.Tensor,
                          prediction_horizon: float = 1.0) -> torch.Tensor:
        """
        预测未来的confidence map
        Args:
            graph: 车辆通信图
            all_local_maps: [num_vehicles, H, W] 所有车辆的本地地图
            all_states: [num_vehicles, state_dim] 所有车辆的状态
            prediction_horizon: 预测时间范围
        Returns:
            pred_map: [H, W] 本车预测的地图
        """
        self.model.eval()
        with torch.no_grad():
            pred_maps = self.model(
                local_maps=all_local_maps,
                vehicle_states=all_states,
                graph=graph,
                prediction_horizon=prediction_horizon
            )
        
        # 返回本车的预测 (假设车辆在图中的索引与vehicle_id对应)
        return pred_maps[self.vehicle_id]
    
    def decide_action(self, pred_map: torch.Tensor, current_map: torch.Tensor) -> float:
        """
        基于预测地图做出行动决策
        Args:
            pred_map: [H, W] 预测的confidence map
            current_map: [H, W] 当前的confidence map
        Returns:
            action: float, 加速度控制 [-1, 1]
        """
        self.action_module.eval()
        with torch.no_grad():
            pred_map_batch = pred_map.unsqueeze(0)  # [1, H, W]
            current_map_batch = current_map.unsqueeze(0)
            action = self.action_module(pred_map_batch, current_map_batch)
        
        return action.item()


class GODECoordinationManager:
    """
    管理车辆间的GODE协同
    """
    def __init__(self, 
                 config,
                 device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # 创建共享的GODE模型 (所有车辆使用相同架构)
        self.gode_model = GODEVehicleModel(
            map_size=tuple(config.conf_map.local_bev_config['grid_size']),
            state_dim=6,  # [x, y, vx, vy, heading, speed]
            hidden_dim=128,
            num_gnn_layers=2,
            use_controlled_gde=True,
            ode_method='dopri5'
        )
        
        self.action_module = ActionDecisionModule(
            map_size=tuple(config.conf_map.local_bev_config['grid_size']),
            hidden_dim=64
        )
        
        # 每辆车的智能体
        self.agents: Dict[int, GODEVehicleAgent] = {}
        
        # 通信时间管理
        self.T_comm = config.gode.T_comm if hasattr(config.gode, 'T_comm') else 10  # 通信周期
        self.T_action = config.gode.T_action if hasattr(config.gode, 'T_action') else 1  # 决策周期
        self.last_comm_time = 0
        self.last_action_time = 0
        
        # 性能统计
        self.stats = defaultdict(list)
        
    def register_vehicle(self, vehicle_id: int):
        """注册一个新车辆"""
        if vehicle_id not in self.agents:
            agent = GODEVehicleAgent(
                vehicle_id=vehicle_id,
                model=self.gode_model,
                action_module=self.action_module,
                device=self.device
            )
            self.agents[vehicle_id] = agent
            print(f"[GODE] Registered vehicle {vehicle_id}")
    
    def should_communicate(self, current_time: int) -> bool:
        """判断是否到了通信时刻"""
        return (current_time - self.last_comm_time) >= self.T_comm
    
    def should_decide_action(self, current_time: int) -> bool:
        """判断是否到了决策时刻"""
        return (current_time - self.last_action_time) >= self.T_action
    
    def perform_communication(self, 
                            vehicle_groups: Dict[int, List[int]],
                            current_time: int):
        """
        执行车辆间通信
        Args:
            vehicle_groups: {group_id: [vehicle_ids]} 车队分组
            current_time: 当前时间步
        """
        print(f"[GODE] Communication at time {current_time}")
        
        # 在每个组内进行通信
        for group_id, vehicle_ids in vehicle_groups.items():
            # 收集本组所有车辆的可分享信息
            shareable_info = {}
            for v_id in vehicle_ids:
                if v_id in self.agents:
                    info = self.agents[v_id].get_shareable_info()
                    if info is not None:
                        shareable_info[v_id] = info
            
            # 广播给组内其他车辆
            for v_id in vehicle_ids:
                if v_id in self.agents:
                    for neighbor_id, neighbor_data in shareable_info.items():
                        if neighbor_id != v_id:
                            self.agents[v_id].receive_neighbor_info(
                                neighbor_id, neighbor_data
                            )
        
        self.last_comm_time = current_time
    
    def predict_and_decide(self,
                          vehicle_groups: Dict[int, List[int]],
                          vehicle_positions: Dict[int, np.ndarray],
                          adjacency_matrix: np.ndarray,
                          current_time: int) -> Dict[int, float]:
        """
        预测环境并做出行动决策
        Args:
            vehicle_groups: 车队分组
            vehicle_positions: {vehicle_id: [x, y]} 车辆位置
            adjacency_matrix: [num_vehicles, num_vehicles] 邻接矩阵
            current_time: 当前时间步
        Returns:
            actions: {vehicle_id: acceleration} 每辆车的加速度控制
        """
        actions = {}
        
        # 对每个组分别处理
        for group_id, vehicle_ids in vehicle_groups.items():
            if len(vehicle_ids) == 0:
                continue
            
            # 1. 构建本组的通信图
            group_positions = np.array([vehicle_positions[v_id] for v_id in vehicle_ids])
            group_graph = build_vehicle_graph(
                positions=group_positions,
                max_comm_range=self.config.gode.max_comm_range if hasattr(self.config.gode, 'max_comm_range') else 200.0
            ).to(self.device)
            
            # 2. 收集本组所有车辆的观测
            local_maps = []
            vehicle_states = []
            valid_vehicles = []
            
            for v_id in vehicle_ids:
                if v_id in self.agents and len(self.agents[v_id].local_map_history) > 0:
                    local_maps.append(self.agents[v_id].local_map_history[-1])
                    vehicle_states.append(self.agents[v_id].state_history[-1])
                    valid_vehicles.append(v_id)
            
            if len(local_maps) == 0:
                continue
            
            # 转为tensor
            local_maps_tensor = torch.FloatTensor(np.array(local_maps)).to(self.device)
            vehicle_states_tensor = torch.FloatTensor(np.array(vehicle_states)).to(self.device)
            
            # 3. GODE预测
            self.gode_model.eval()
            with torch.no_grad():
                pred_maps = self.gode_model(
                    local_maps=local_maps_tensor,
                    vehicle_states=vehicle_states_tensor,
                    graph=group_graph,
                    prediction_horizon=self.T_action * 0.1  # 假设每步0.1秒
                )
            
            # 4. 决策
            for idx, v_id in enumerate(valid_vehicles):
                pred_map = pred_maps[idx]
                current_map = local_maps_tensor[idx]
                
                action = self.agents[v_id].decide_action(pred_map, current_map)
                actions[v_id] = action
                
                # 记录统计
                self.stats['nfe'].append(self.gode_model.get_nfe())
                
            # 重置NFE计数
            self.gode_model.reset_nfe()
        
        self.last_action_time = current_time
        
        return actions
    
    def train_step(self, 
                  batch_data: Dict,
                  optimizer: torch.optim.Optimizer,
                  criterion: torch.nn.Module) -> float:
        """
        训练一步
        Args:
            batch_data: {
                'local_maps': [batch, num_vehicles, H, W],
                'states': [batch, num_vehicles, state_dim],
                'graphs': [DGLGraph],
                'target_maps': [batch, num_vehicles, H, W]
            }
            optimizer: 优化器
            criterion: 损失函数
        Returns:
            loss: 损失值
        """
        self.gode_model.train()
        self.action_module.train()
        
        local_maps = batch_data['local_maps'].to(self.device)
        states = batch_data['states'].to(self.device)
        target_maps = batch_data['target_maps'].to(self.device)
        graphs = batch_data['graphs']
        
        batch_size, num_vehicles = local_maps.shape[:2]
        
        total_loss = 0
        
        for b in range(batch_size):
            # 预测
            pred_maps = self.gode_model(
                local_maps=local_maps[b],
                vehicle_states=states[b],
                graph=graphs[b].to(self.device),
                prediction_horizon=1.0
            )
            
            # 计算损失
            loss = criterion(pred_maps, target_maps[b])
            
            total_loss += loss
        
        # 平均损失
        avg_loss = total_loss / batch_size
        
        # 反向传播
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        return avg_loss.item()
    
    def save_model(self, save_dir: str, epoch: int):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'gode_model_state_dict': self.gode_model.state_dict(),
            'action_module_state_dict': self.action_module.state_dict(),
            'stats': dict(self.stats)
        }, os.path.join(save_dir, f'gode_checkpoint_epoch_{epoch}.pth'))
        
        print(f"[GODE] Model saved at epoch {epoch}")
    
    def load_model(self, checkpoint_path: str):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.gode_model.load_state_dict(checkpoint['gode_model_state_dict'])
        self.action_module.load_state_dict(checkpoint['action_module_state_dict'])
        
        print(f"[GODE] Model loaded from {checkpoint_path}")
        return checkpoint.get('epoch', 0)


def extract_vehicle_state(actor, world) -> np.ndarray:
    """
    从CARLA actor提取车辆状态
    Args:
        actor: CARLA vehicle actor
        world: WorldManager
    Returns:
        state: [6] numpy array [x, y, vx, vy, heading, speed]
    """
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    
    x = transform.location.x
    y = transform.location.y
    vx = velocity.x
    vy = velocity.y
    heading = np.radians(transform.rotation.yaw)  # 转为弧度
    speed = np.sqrt(vx**2 + vy**2)
    
    return np.array([x, y, vx, vy, heading, speed], dtype=np.float32)
