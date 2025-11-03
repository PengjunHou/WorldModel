"""
数据生成和加载：模拟车联网环境，生成训练数据
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random

from graph_builder import CooperativePerceptionGraph


class VehicularEnvironment:
    """
    车联网环境模拟器
    模拟车辆移动、感知数据生成、通信等
    """
    
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self) -> Dict:
        """
        重置环境
        Returns:
            初始状态
        """
        # 初始化车辆
        self.vehicles = self._initialize_vehicles()
        
        # 初始化区域
        self.regions = self._initialize_regions()
        
        # 时间步
        self.time_step = 0
        
        return self._get_state()
    
    def _initialize_vehicles(self) -> List[Dict]:
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
                'heading': heading,
                'perception_history': [],  # 存储历史感知地图
            }
            
            vehicles.append(vehicle)
        
        return vehicles
    
    def _initialize_regions(self) -> List[Dict]:
        """初始化感知区域网格"""
        regions = []
        grid_size = self.config.GRID_SIZE
        cell_size = self.config.PERCEPTION_RANGE / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 区域中心位置
                position = np.array([
                    (i + 0.5) * cell_size,
                    (j + 0.5) * cell_size
                ])
                
                region = {
                    'id': i * grid_size + j,
                    'position': position,
                    'confidence': np.random.rand(),  # 初始confidence
                    'interest_count': 0,  # 感兴趣的车辆数
                    'semantic_type': np.random.choice(['road', 'building', 'tree', 'vehicle'])
                }
                
                regions.append(region)
        
        return regions
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict, torch.Tensor, bool, Dict]:
        """
        环境步进
        
        Args:
            actions: [N_r] 二值向量，表示哪些区域被选择共享
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 1. 更新车辆位置
        self._update_vehicles()
        
        # 2. 更新区域confidence
        self._update_regions(actions)
        
        # 3. 计算奖励
        reward = self._compute_reward(actions)
        
        # 4. 检查是否结束
        self.time_step += 1
        done = self.time_step >= self.config.MAX_EPISODE_LENGTH
        
        # 5. 获取新状态
        next_state = self._get_state()
        
        # 6. 额外信息
        info = {
            'time_step': self.time_step,
            'num_shared_regions': actions.sum().item(),
            'avg_confidence': np.mean([r['confidence'] for r in self.regions])
        }
        
        return next_state, reward, done, info
    
    def _update_vehicles(self):
        """更新车辆位置"""
        dt = 0.1  # 时间步长
        
        for vehicle in self.vehicles:
            # 更新位置
            vehicle['position'] += vehicle['velocity'] * dt
            
            # 边界处理（循环边界）
            vehicle['position'] = vehicle['position'] % self.config.PERCEPTION_RANGE
            
            # 小概率改变方向（模拟转向）
            if np.random.rand() < 0.05:
                vehicle['heading'] += np.random.randn() * 0.3
                speed = np.linalg.norm(vehicle['velocity'])
                vehicle['velocity'] = np.array([
                    speed * np.cos(vehicle['heading']),
                    speed * np.sin(vehicle['heading'])
                ])
            
            # 更新感知地图历史
            perception_map = self._generate_perception_map(vehicle)
            vehicle['perception_history'].append(perception_map)
            
            # 只保留最近K个
            if len(vehicle['perception_history']) > self.config.K_HISTORY:
                vehicle['perception_history'].pop(0)
    
    def _generate_perception_map(self, vehicle: Dict) -> np.ndarray:
        """
        生成车辆的感知地图
        简化版本：基于距离衰减
        """
        grid_size = self.config.GRID_SIZE
        perception_map = np.zeros((grid_size, grid_size))
        
        cell_size = self.config.PERCEPTION_RANGE / grid_size
        vehicle_pos = vehicle['position']
        
        for i in range(grid_size):
            for j in range(grid_size):
                cell_center = np.array([(i + 0.5) * cell_size, (j + 0.5) * cell_size])
                distance = np.linalg.norm(vehicle_pos - cell_center)
                
                # 距离衰减
                if distance < self.config.PERCEPTION_RANGE:
                    perception_map[i, j] = max(0, 1 - distance / self.config.PERCEPTION_RANGE)
                    # 添加一些噪声
                    perception_map[i, j] += np.random.randn() * 0.1
                    perception_map[i, j] = np.clip(perception_map[i, j], 0, 1)
        
        return perception_map
    
    def _update_regions(self, actions: torch.Tensor):
        """
        更新区域状态
        
        Args:
            actions: [N_r] 哪些区域被选择共享
        """
        for idx, region in enumerate(self.regions):
            # 计算有多少车辆对这个区域感兴趣
            interest_count = self._compute_interest_count(region)
            region['interest_count'] = interest_count
            
            # 如果该区域被共享，confidence提升
            if actions[idx] > 0.5:
                # 共享数据会提升confidence
                region['confidence'] = min(1.0, region['confidence'] + 0.1)
            else:
                # 不共享，confidence随时间衰减
                region['confidence'] = max(0.0, region['confidence'] - 0.05)
    
    def _compute_interest_count(self, region: Dict) -> int:
        """计算有多少车辆对该区域感兴趣"""
        count = 0
        region_pos = region['position']
        
        for vehicle in self.vehicles:
            vehicle_pos = vehicle['position']
            distance = np.linalg.norm(vehicle_pos - region_pos)
            
            # 在感知范围内且不是建筑物类型（建筑物通常不需要共享）
            if distance < self.config.PERCEPTION_RANGE * 0.7:
                if region['semantic_type'] != 'building':
                    count += 1
        
        return count
    
    def _compute_reward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        计算奖励函数
        权衡：感知质量提升 vs 通信开销
        """
        # 1. 感知质量奖励：所有区域的confidence加权和
        quality_reward = 0
        for region in self.regions:
            interest = region['interest_count']
            confidence = region['confidence']
            # 被更多车辆感兴趣的区域，其confidence更重要
            quality_reward += interest * confidence
        
        # 2. 通信开销惩罚：共享的区域数量
        num_shared = actions.sum().item()
        communication_cost = num_shared * 0.1  # 每个共享区域的固定代价
        
        # 3. 总奖励
        reward = quality_reward - communication_cost
        
        return torch.tensor(reward, dtype=torch.float32)
    
    def _get_state(self) -> Dict:
        """获取当前状态"""
        # 车辆数据
        vehicle_positions = np.array([v['position'] for v in self.vehicles])
        vehicle_velocities = np.array([v['velocity'] for v in self.vehicles])
        vehicle_headings = np.array([v['heading'] for v in self.vehicles])
        
        # 历史感知地图（填充到K个时间步）
        vehicle_perception_maps = []
        for vehicle in self.vehicles:
            history = vehicle['perception_history']
            # 填充到K个时间步
            while len(history) < self.config.K_HISTORY:
                history.insert(0, np.zeros((self.config.GRID_SIZE, self.config.GRID_SIZE)))
            vehicle_perception_maps.append(np.stack(history[-self.config.K_HISTORY:]))
        
        vehicle_perception_maps = np.array(vehicle_perception_maps)
        
        # 区域数据
        region_positions = np.array([r['position'] for r in self.regions])
        region_confidences = np.array([r['confidence'] for r in self.regions])
        region_interests = np.array([r['interest_count'] for r in self.regions])
        
        # 语义地图（简化版：每个语义类型的one-hot编码）
        semantic_map = self._generate_semantic_map()
        
        state = {
            'vehicle_data': {
                'positions': torch.FloatTensor(vehicle_positions),
                'velocities': torch.FloatTensor(vehicle_velocities),
                'headings': torch.FloatTensor(vehicle_headings),
                'perception_maps': torch.FloatTensor(vehicle_perception_maps)
            },
            'region_data': {
                'positions': torch.FloatTensor(region_positions),
                'confidence_maps': torch.FloatTensor(region_confidences),
                'interest_counts': torch.FloatTensor(region_interests)
            },
            'semantic_map': torch.FloatTensor(semantic_map)
        }
        
        return state
    
    def _generate_semantic_map(self) -> np.ndarray:
        """生成语义地图"""
        semantic_types = ['road', 'building', 'tree', 'vehicle']
        num_channels = len(semantic_types)
        
        semantic_map = np.zeros((
            num_channels,
            self.config.GRID_SIZE,
            self.config.GRID_SIZE
        ))
        
        cell_size = self.config.PERCEPTION_RANGE / self.config.GRID_SIZE
        
        for region in self.regions:
            # 找到对应的网格位置
            grid_x = int(region['position'][0] / cell_size)
            grid_y = int(region['position'][1] / cell_size)
            
            # 确保在范围内
            grid_x = min(grid_x, self.config.GRID_SIZE - 1)
            grid_y = min(grid_y, self.config.GRID_SIZE - 1)
            
            # 设置对应的语义通道
            semantic_idx = semantic_types.index(region['semantic_type'])
            semantic_map[semantic_idx, grid_y, grid_x] = 1.0
        
        return semantic_map


class CooperativePerceptionDataset(Dataset):
    """协同感知数据集"""
    
    def __init__(
        self,
        config,
        num_episodes: int = 1000,
        generate_online: bool = False
    ):
        self.config = config
        self.num_episodes = num_episodes
        self.generate_online = generate_online
        
        self.graph_builder = CooperativePerceptionGraph(config)
        
        if not generate_online:
            # 预生成数据
            self.data = self._generate_dataset()
        else:
            self.env = VehicularEnvironment(config)
    
    def _generate_dataset(self) -> List[Dict]:
        """预生成整个数据集"""
        print("Generating dataset...")
        dataset = []
        
        env = VehicularEnvironment(self.config)
        
        for episode in range(self.num_episodes):
            state = env.reset()
            episode_data = []
            
            for step in range(self.config.MAX_EPISODE_LENGTH):
                # 随机策略（用于生成初始数据）
                action = torch.randint(0, 2, (self.config.NUM_REGIONS,)).float()
                
                next_state, reward, done, info = env.step(action)
                
                # 构建图
                graph = self.graph_builder.build_graph(
                    state['vehicle_data'],
                    state['region_data'],
                    state['semantic_map']
                )
                
                next_graph = self.graph_builder.build_graph(
                    next_state['vehicle_data'],
                    next_state['region_data'],
                    next_state['semantic_map']
                )
                
                experience = {
                    'state_graph': graph,
                    'action': action,
                    'reward': reward,
                    'next_state_graph': next_graph,
                    'done': done,
                    'info': {
                        'current_interest_count': state['region_data']['interest_counts'],
                        'next_interest_count': next_state['region_data']['interest_counts'],
                        'transition_prob': torch.ones(self.config.NUM_REGIONS) * 0.8,  # 简化
                        'confidence_change': (
                            next_state['region_data']['confidence_maps'] -
                            state['region_data']['confidence_maps']
                        )
                    }
                }
                
                episode_data.append(experience)
                
                if done:
                    break
                
                state = next_state
            
            dataset.extend(episode_data)
            
            if (episode + 1) % 100 == 0:
                print(f"Generated {episode + 1}/{self.num_episodes} episodes")
        
        print(f"Dataset generation complete. Total samples: {len(dataset)}")
        return dataset
    
    def __len__(self):
        if self.generate_online:
            return self.num_episodes * self.config.MAX_EPISODE_LENGTH
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        if self.generate_online:
            # 在线生成
            state = self.env.reset()
            action = torch.randint(0, 2, (self.config.NUM_REGIONS,)).float()
            next_state, reward, done, info = self.env.step(action)
            
            graph = self.graph_builder.build_graph(
                state['vehicle_data'],
                state['region_data'],
                state['semantic_map']
            )
            
            next_graph = self.graph_builder.build_graph(
                next_state['vehicle_data'],
                next_state['region_data'],
                next_state['semantic_map']
            )
            
            return {
                'state_graph': graph,
                'next_state_graph': next_graph,
                'action': action,
                'reward': reward,
                'done': done
            }
        else:
            return self.data[idx]


def create_dataloaders(config, train_episodes: int = 800, val_episodes: int = 200):
    """创建训练和验证数据加载器"""
    
    train_dataset = CooperativePerceptionDataset(
        config,
        num_episodes=train_episodes,
        generate_online=False
    )
    
    val_dataset = CooperativePerceptionDataset(
        config,
        num_episodes=val_episodes,
        generate_online=False
    )
    
    # 注意：由于PyG的HeteroData不能直接用标准DataLoader，
    # 这里简化处理，实际使用时需要自定义collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # PyG图数据通常单独处理
        shuffle=True,
        num_workers=0  # PyG图数据不支持多进程
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader
