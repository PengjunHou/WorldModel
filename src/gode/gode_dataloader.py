"""
GODE数据加载器 - 完整修复版
1. 正确处理episode边界,避免数据泄漏
2. 考虑真实通信约束(map有延迟,position实时)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from pathlib import Path
import os


class GODEDataset(Dataset):
    """
    GODE训练数据集 - 完整修复版
    
    数据格式 (从gode_dataset.npz加载):
    - local_maps: [T, N_vehicles, H, W] 每个时间步每辆车的local map
    - fused_maps: [T, N_vehicles, H, W] 每个时间步每辆车的fused map
    - positions: [T, N_vehicles, 2] 每个时间步每辆车的位置
    - K: [T] 每个时间步的K值
    
    Episode边界 (从episode_boundaries.npz加载):
    - start_idx: [N_episodes] 每个episode的起始时间步
    - end_idx: [N_episodes] 每个episode的结束时间步
    - lengths: [N_episodes] 每个episode的长度
    
    通信矩阵 (从communication_matrix.npz加载,可选):
    - communication_matrix: [T, N_vehicles, N_vehicles]
      communication_matrix[t, i, j] = 车辆i在t时刻拥有的车辆j的地图时间戳
    
    每个样本返回:
    对某辆车v在时刻t:
    - self_history_maps: [T, H, W] 自己过去T个时刻的local maps (实时)
    - self_history_positions: [T, 2] 自己过去T个时刻的positions (实时)
    - others_maps: [N-1, T, H, W] 其他车辆T个时刻的local maps (考虑通信延迟)
    - others_positions: [N-1, T, 2] 其他车辆T个时刻的positions (实时,GPS)
    - K_value: 标量, 当前K值
    - target_fused_map: [H, W] 目标融合map
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 5,
        split: str = 'train',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        use_perfect_communication: bool = False
    ):
        """
        Args:
            data_path: npz文件路径
            sequence_length: 历史序列长度T
            split: 'train', 'val', 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            use_perfect_communication: 是否假设完美通信(用于对比实验)
        """
        self.sequence_length = sequence_length
        self.split = split
        self.use_perfect_communication = use_perfect_communication
        
        # 加载主数据
        print(f"Loading GODE dataset from {data_path}...")
        data = np.load(data_path)
        
        self.local_maps = data['local_maps']    # [T, N_vehicles, H, W]
        self.fused_maps = data['fused_maps']    # [T, N_vehicles, H, W]
        self.positions = data['positions']      # [T, N_vehicles, 2]
        self.K_values = data['K']               # [T]
        
        total_timesteps, self.n_vehicles, self.H, self.W = self.local_maps.shape
        
        print(f"✓ Loaded data:")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Number of vehicles: {self.n_vehicles}")
        print(f"  Map size: ({self.H}, {self.W})")
        print(f"  Sequence length: {sequence_length}")
        
        # ===== 加载episode边界信息 =====
        boundary_path = os.path.join(
            os.path.dirname(data_path),
            "episode_boundaries.npz"
        )
        
        if os.path.exists(boundary_path):
            print(f"\n✓ Loading episode boundaries from {boundary_path}")
            boundaries = np.load(boundary_path)
            self.episode_starts = boundaries['start_idx']
            self.episode_ends = boundaries['end_idx']
            self.episode_lengths = boundaries['lengths']
            
            print(f"  Number of episodes: {len(self.episode_starts)}")
            print(f"  Episode length: min={self.episode_lengths.min()}, "
                  f"max={self.episode_lengths.max()}, "
                  f"mean={self.episode_lengths.mean():.1f}")
        else:
            print(f"\n⚠️  Episode boundaries not found: {boundary_path}")
            print(f"   Treating all data as single episode (may cause data leakage!)")
            self.episode_starts = np.array([0])
            self.episode_ends = np.array([total_timesteps])
            self.episode_lengths = np.array([total_timesteps])
        
        # ===== 加载或构建通信矩阵 =====
        comm_matrix_path = os.path.join(
            os.path.dirname(data_path),
            "communication_matrix.npz"
        )
        
        if os.path.exists(comm_matrix_path) and not use_perfect_communication:
            print(f"\n✓ Loading communication matrix from {comm_matrix_path}")
            comm_data = np.load(comm_matrix_path)
            self.communication_matrix = comm_data['communication_matrix']
            print(f"  Communication matrix shape: {self.communication_matrix.shape}")
            
            # 统计通信延迟
            delays = []
            for t in range(min(100, total_timesteps)):
                for i in range(self.n_vehicles):
                    for j in range(self.n_vehicles):
                        if i != j:
                            delay = t - self.communication_matrix[t, i, j]
                            delays.append(delay)
            
            if delays:
                delays = np.array(delays)
                print(f"  Communication delay: mean={delays.mean():.2f}, "
                      f"max={delays.max()}, std={delays.std():.2f} timesteps")
        else:
            if use_perfect_communication:
                print(f"\n⚠️  Using PERFECT COMMUNICATION (for comparison)")
            else:
                print(f"\n⚠️  Communication matrix not found: {comm_matrix_path}")
                print(f"   Using PERFECT COMMUNICATION by default")
            
            # 构建完美通信矩阵
            self.communication_matrix = np.zeros(
                (total_timesteps, self.n_vehicles, self.n_vehicles),
                dtype=np.int32
            )
            for t in range(total_timesteps):
                for i in range(self.n_vehicles):
                    for j in range(self.n_vehicles):
                        self.communication_matrix[t, i, j] = t
        
        # ===== 构建有效的时间窗口索引 (考虑episode边界) =====
        valid_indices = []
        
        for ep_idx, (ep_start, ep_end) in enumerate(zip(self.episode_starts, self.episode_ends)):
            # 对于每个episode,只有在episode内部有足够历史的时间步才是有效的
            for t in range(ep_start + sequence_length - 1, ep_end):
                for v in range(self.n_vehicles):
                    valid_indices.append((t, v, ep_idx))
        
        print(f"\n✓ Created valid indices:")
        print(f"  Total valid samples: {len(valid_indices)}")
        
        # 统计每个episode的有效样本数
        if len(self.episode_starts) > 1:
            print(f"  Samples per episode:")
            for ep_idx in range(min(5, len(self.episode_starts))):
                ep_samples = sum(1 for _, _, e in valid_indices if e == ep_idx)
                print(f"    Episode {ep_idx}: {ep_samples} samples "
                      f"(length={self.episode_lengths[ep_idx]})")
            if len(self.episode_starts) > 5:
                print(f"    ... and {len(self.episode_starts) - 5} more episodes")
        
        # 划分数据集
        n_samples = len(valid_indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        if split == 'train':
            self.indices = valid_indices[:n_train]
        elif split == 'val':
            self.indices = valid_indices[n_train:n_train+n_val]
        else:  # test
            self.indices = valid_indices[n_train+n_val:]
        
        print(f"\n✓ {split.upper()} split: {len(self.indices)} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        返回一个训练样本 - 考虑episode边界和通信约束
        """
        t, v, ep_idx = self.indices[idx]  # 时间步、车辆ID、episode索引
        
        # 1. 自己的历史 (从t-sequence_length+1到t, 包含当前时刻)
        t_start = t - self.sequence_length + 1
        
        # ===== 验证不跨episode边界 =====
        ep_start = self.episode_starts[ep_idx]
        ep_end = self.episode_ends[ep_idx]
        assert t_start >= ep_start, \
            f"Sequence starts before episode! t_start={t_start}, ep_start={ep_start}, episode={ep_idx}"
        assert t < ep_end, \
            f"Time step exceeds episode! t={t}, ep_end={ep_end}, episode={ep_idx}"
        
        # ===== 自己的数据 (实时) =====
        self_history_maps = self.local_maps[t_start:t+1, v, :, :]  # [T, H, W]
        self_history_positions = self.positions[t_start:t+1, v, :]  # [T, 2]
        
        # ===== 其他车辆的数据 (考虑通信延迟) =====
        other_indices = [i for i in range(self.n_vehicles) if i != v]
        N_others = len(other_indices)
        
        # 初始化
        others_maps = np.zeros((N_others, self.sequence_length, self.H, self.W), dtype=np.float32)
        others_positions = np.zeros((N_others, self.sequence_length, 2), dtype=np.float32)
        
        # 对每个其他车辆,每个历史时刻
        for idx_other, other_v in enumerate(other_indices):
            for idx_t, t_hist in enumerate(range(t_start, t+1)):
                # 获取车辆v在t_hist时刻从other_v获得的地图时间戳
                map_timestamp = self.communication_matrix[t_hist, v, other_v]
                
                # 确保不超出episode边界
                map_timestamp = max(map_timestamp, ep_start)
                map_timestamp = min(map_timestamp, t_hist)
                
                # 取该时间戳的地图 (可能有延迟)
                others_maps[idx_other, idx_t, :, :] = self.local_maps[map_timestamp, other_v, :, :]
                
                # Position可以实时获取 (GPS广播频率高)
                others_positions[idx_other, idx_t, :] = self.positions[t_hist, other_v, :]
        
        # ===== 当前K值 =====
        K_value = self.K_values[t]
        
        # ===== 目标融合map (当前时刻) =====
        target_fused_map = self.fused_maps[t, v, :, :]  # [H, W]
        
        # 转换为Tensor
        return {
            'self_history_maps': torch.FloatTensor(self_history_maps),  # [T, H, W]
            'self_history_positions': torch.FloatTensor(self_history_positions),  # [T, 2]
            'others_maps': torch.FloatTensor(others_maps),  # [N-1, T, H, W]
            'others_positions': torch.FloatTensor(others_positions),  # [N-1, T, 2]
            'K_value': torch.FloatTensor([K_value]),  # [1]
            'target_fused_map': torch.FloatTensor(target_fused_map),  # [H, W]
            'vehicle_id': v,
            'timestep': t,
            'episode_id': ep_idx
        }


def create_gode_dataloaders(
    data_path: str,
    sequence_length: int = 5,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    use_perfect_communication: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        data_path: gode_dataset.npz路径
        sequence_length: 历史序列长度
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 数据加载线程数
        use_perfect_communication: 是否假设完美通信(调试用)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建数据集
    train_dataset = GODEDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        split='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_perfect_communication=use_perfect_communication
    )
    
    val_dataset = GODEDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_perfect_communication=use_perfect_communication
    )
    
    test_dataset = GODEDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        use_perfect_communication=use_perfect_communication
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_communication_matrix(
    K_values: np.ndarray,
    n_vehicles: int,
    communication_period: int = 5
) -> np.ndarray:
    """
    生成简化的通信矩阵
    
    假设: 每communication_period个时间步,车辆进行一次通信
    
    Args:
        K_values: [T] 每个时刻的K值
        n_vehicles: 车辆数量
        communication_period: 通信周期(时间步)
    
    Returns:
        communication_matrix: [T, N_vehicles, N_vehicles]
    """
    T = len(K_values)
    comm_matrix = np.zeros((T, n_vehicles, n_vehicles), dtype=np.int32)
    
    for v in range(n_vehicles):
        last_comm_time = 0  # 上次通信时间
        
        for t in range(T):
            # 判断是否通信
            if t % communication_period == 0:
                last_comm_time = t
            
            # 所有其他车辆的地图时间戳 = 上次通信时间
            for other_v in range(n_vehicles):
                if other_v == v:
                    comm_matrix[t, v, other_v] = t  # 自己的地图总是最新
                else:
                    comm_matrix[t, v, other_v] = last_comm_time
    
    return comm_matrix


def save_communication_matrix(
    data_path: str,
    communication_period: int = 5
):
    """
    为已有的数据集生成通信矩阵并保存
    
    Args:
        data_path: gode_dataset.npz路径
        communication_period: 通信周期
    """
    # 加载数据
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    K_values = data['K']
    local_maps = data['local_maps']
    T, n_vehicles, H, W = local_maps.shape
    
    print(f"Generating communication matrix...")
    print(f"  Timesteps: {T}")
    print(f"  Vehicles: {n_vehicles}")
    print(f"  Communication period: {communication_period}")
    
    # 生成通信矩阵
    comm_matrix = create_communication_matrix(
        K_values,
        n_vehicles,
        communication_period
    )
    
    # 保存
    output_path = os.path.join(
        os.path.dirname(data_path),
        "communication_matrix.npz"
    )
    
    np.savez_compressed(
        output_path,
        communication_matrix=comm_matrix,
        communication_period=communication_period
    )
    
    print(f"\n✅ Communication matrix saved: {output_path}")
    print(f"   Shape: {comm_matrix.shape}")
    
    # 统计延迟
    delays = []
    for t in range(T):
        for v in range(n_vehicles):
            for other_v in range(n_vehicles):
                if other_v != v:
                    delay = t - comm_matrix[t, v, other_v]
                    delays.append(delay)
    
    delays = np.array(delays)
    print(f"\n   Communication delay statistics:")
    print(f"     Mean: {delays.mean():.2f} timesteps")
    print(f"     Max: {delays.max()} timesteps")
    print(f"     Std: {delays.std():.2f}")


# 测试代码
if __name__ == "__main__":
    # 测试数据加载
    data_path = "/home/peh324/Codes/WorldModel/data/gode_data/gode_dataset.npz"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run evaluate_gode_final.py first to collect data.")
    else:
        # 生成通信矩阵(如果不存在)
        comm_matrix_path = os.path.join(
            os.path.dirname(data_path),
            "communication_matrix.npz"
        )
        
        if not os.path.exists(comm_matrix_path):
            print("Communication matrix not found, generating...")
            save_communication_matrix(
                data_path,
                communication_period=5
            )
        
        print("\n" + "="*60)
        print("Testing DataLoader")
        print("="*60)
        
        # 测试: 真实通信约束
        print("\n[Test 1] With communication constraints:")
        train_loader, val_loader, test_loader = create_gode_dataloaders(
            data_path=data_path,
            sequence_length=5,
            batch_size=8,
            use_perfect_communication=False
        )
        
        # 获取一个batch
        for batch in train_loader:
            print("\nBatch shapes:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
            break
        
        # 测试: 完美通信(对比)
        print("\n" + "="*60)
        print("[Test 2] With perfect communication (for comparison):")
        train_loader_perfect, _, _ = create_gode_dataloaders(
            data_path=data_path,
            sequence_length=5,
            batch_size=8,
            use_perfect_communication=True
        )
        
        print("\n✓ DataLoader test passed!")
        print("\nUsage:")
        print("  - use_perfect_communication=False: Realistic (with delays)")
        print("  - use_perfect_communication=True: Ideal (no delays, for debugging)")