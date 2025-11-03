"""
车联网环境模拟器
"""
import torch
import numpy as np
from typing import Dict, Tuple


class VehicularEnvironment:
    """
    车联网协同感知环境模拟器
    """
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self) -> Dict[str, torch.Tensor]:
        """重置环境"""
        # 初始化车辆
        self.vehicles = self._initialize_vehicles()
        
        # 初始化历史数据
        self.perception_history = []
        self.interest_history = []
        self.position_history = []
        
        self.time_step = 0
        
        # 收集初始历史（填充相同的初始状态）
        initial_state = self._get_current_state()
        for _ in range(self.config.HISTORY_LENGTH):
            self.perception_history.append(initial_state['perception_maps'])
            self.interest_history.append(initial_state['interest_map'])
            self.position_history.append(initial_state['positions'])
        
        return self._get_state()
    
    def _initialize_vehicles(self) -> list:
        """初始化车辆"""
        vehicles = []
        
        for i in range(self.config.NUM_VEHICLES):
            # 随机位置
            position = np.random.rand(2) * self.config.PERCEPTION_RANGE
            
            # 随机速度（5-15 m/s）
            speed = np.random.rand() * 10 + 5
            heading = np.random.rand() * 2 * np.pi
            velocity = np.array([
                speed * np.cos(heading),
                speed * np.sin(heading)
            ])
            
            vehicle = {
                'id': i,
                'position': position,
                'velocity': velocity,
                'heading': heading
            }
            
            vehicles.append(vehicle)
        
        return vehicles
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict, float, bool, Dict]:
        """
        环境步进
        
        Args:
            actions: [N_v, H, W] 每辆车选择的区域mask
        
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 更新车辆位置
        self._update_vehicles()
        
        # 获取当前状态
        current_state = self._get_current_state()
        
        # 计算奖励
        reward = self._compute_reward(actions, current_state)
        
        # 更新历史
        self.perception_history.append(current_state['perception_maps'])
        self.interest_history.append(current_state['interest_map'])
        self.position_history.append(current_state['positions'])
        
        # 保持历史长度
        if len(self.perception_history) > self.config.HISTORY_LENGTH:
            self.perception_history.pop(0)
            self.interest_history.pop(0)
            self.position_history.pop(0)
        
        # 检查是否结束
        self.time_step += 1
        done = self.time_step >= self.config.MAX_EPISODE_LENGTH
        
        # 额外信息
        info = {
            'time_step': self.time_step,
            'num_shared_regions': actions.sum().item(),
            'avg_perception_quality': current_state['perception_maps'].mean().item()
        }
        
        return self._get_state(), reward, done, info
    
    def _update_vehicles(self):
        """更新车辆位置"""
        dt = 0.1  # 时间步长
        
        for vehicle in self.vehicles:
            # 更新位置
            vehicle['position'] += vehicle['velocity'] * dt
            
            # 边界处理（循环边界）
            vehicle['position'] = vehicle['position'] % self.config.PERCEPTION_RANGE
            
            # 小概率改变方向
            if np.random.rand() < 0.05:
                vehicle['heading'] += np.random.randn() * 0.3
                speed = np.linalg.norm(vehicle['velocity'])
                vehicle['velocity'] = np.array([
                    speed * np.cos(vehicle['heading']),
                    speed * np.sin(vehicle['heading'])
                ])
    
    def _get_current_state(self) -> Dict:
        """获取当前时刻的状态"""
        N_v = len(self.vehicles)
        H, W = self.config.MAP_HEIGHT, self.config.MAP_WIDTH
        
        # 车辆位置
        positions = np.array([v['position'] for v in self.vehicles])
        
        # 生成每辆车的感知质量map
        perception_maps = np.zeros((N_v, H, W))
        for i, vehicle in enumerate(self.vehicles):
            perception_maps[i] = self._generate_perception_map(vehicle['position'])
        
        # 生成兴趣度map
        interest_map = self._generate_interest_map(positions)
        
        return {
            'perception_maps': torch.FloatTensor(perception_maps),
            'interest_map': torch.FloatTensor(interest_map),
            'positions': torch.FloatTensor(positions)
        }
    
    def _generate_perception_map(self, vehicle_position: np.ndarray) -> np.ndarray:
        """
        生成车辆的感知质量map
        基于距离衰减
        """
        H, W = self.config.MAP_HEIGHT, self.config.MAP_WIDTH
        perception_map = np.zeros((H, W))
        
        cell_size_x = self.config.PERCEPTION_RANGE / W
        cell_size_y = self.config.PERCEPTION_RANGE / H
        
        for i in range(H):
            for j in range(W):
                # 网格中心位置
                cell_center = np.array([
                    (j + 0.5) * cell_size_x,
                    (i + 0.5) * cell_size_y
                ])
                
                # 距离
                distance = np.linalg.norm(vehicle_position - cell_center)
                
                # 距离衰减
                if distance < self.config.PERCEPTION_RANGE:
                    perception_map[i, j] = max(0, 1 - distance / self.config.PERCEPTION_RANGE)
                    # 添加噪声
                    perception_map[i, j] += np.random.randn() * 0.05
                    perception_map[i, j] = np.clip(perception_map[i, j], 0, 1)
        
        return perception_map
    
    def _generate_interest_map(self, positions: np.ndarray) -> np.ndarray:
        """
        生成兴趣度map
        每个网格的值 = 有多少车辆对该网格感兴趣
        """
        H, W = self.config.MAP_HEIGHT, self.config.MAP_WIDTH
        interest_map = np.zeros((H, W))
        
        cell_size_x = self.config.PERCEPTION_RANGE / W
        cell_size_y = self.config.PERCEPTION_RANGE / H
        
        for i in range(H):
            for j in range(W):
                cell_center = np.array([
                    (j + 0.5) * cell_size_x,
                    (i + 0.5) * cell_size_y
                ])
                
                # 计算有多少车辆对这个网格感兴趣
                # 规则：在感知范围内且感知质量 > 阈值
                count = 0
                for pos in positions:
                    distance = np.linalg.norm(pos - cell_center)
                    if distance < self.config.PERCEPTION_RANGE * 0.7:
                        perception_quality = max(0, 1 - distance / self.config.PERCEPTION_RANGE)
                        if perception_quality > 0.3:  # 阈值
                            count += 1
                
                interest_map[i, j] = count / self.config.NUM_VEHICLES  # 归一化
        
        return interest_map
    
    def _compute_reward(
        self,
        actions: torch.Tensor,
        current_state: Dict
    ) -> float:
        """
        计算奖励
        reward = 感知质量提升 - 通信开销
        """
        # 感知质量：被选择区域的平均质量 × 兴趣度
        perception_maps = current_state['perception_maps']  # [N_v, H, W]
        interest_map = current_state['interest_map']  # [H, W]
        
        # 被选择区域的质量
        selected_quality = (perception_maps * actions.cpu()).sum()
        
        # 被选择区域的兴趣度
        selected_interest = (interest_map.unsqueeze(0) * actions.cpu()).sum()
        
        # 感知质量奖励
        quality_reward = (selected_quality * selected_interest).item()
        
        # 通信开销惩罚
        num_shared = actions.sum().item()
        comm_cost = num_shared * 0.01
        
        # 总奖励
        reward = quality_reward - comm_cost
        
        return reward
    
    def _get_state(self) -> Dict[str, torch.Tensor]:
        """获取完整状态（包含历史）"""
        # 堆叠历史
        perception_history = torch.stack(self.perception_history, dim=1)  # [N_v, T, H, W]
        interest_history = torch.stack(self.interest_history, dim=0)  # [T, H, W]
        position_history = torch.stack(self.position_history, dim=1)  # [N_v, T, 2]
        
        return {
            'perception_maps_history': perception_history,
            'interest_maps_history': interest_history,
            'positions_history': position_history
        }
