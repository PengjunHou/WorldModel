"""
配置文件：定义所有超参数和模型配置
"""
import torch

class Config:
    """全局配置类"""
    
    # ============ 环境参数 ============
    NUM_VEHICLES = 10  # 最大车辆数
    NUM_REGIONS = 64   # 感知范围划分的区域数 (例如 8x8 grid)
    GRID_SIZE = 8      # 网格大小 (8x8)
    PERCEPTION_RANGE = 100  # 感知范围 (米)
    
    # ============ 时间参数 ============
    K_HISTORY = 5      # 历史时间步数
    MAX_EPISODE_LENGTH = 100  # 每个episode的最大长度
    DISCOUNT_FACTOR = 0.95    # 折扣因子 β
    
    # ============ 输入维度 ============
    VEHICLE_FEATURE_DIM = 128   # 车辆节点特征维度
    REGION_FEATURE_DIM = 64     # 区域节点特征维度
    SEMANTIC_MAP_CHANNELS = 10  # 语义地图通道数
    CONFIDENCE_MAP_CHANNELS = 1 # Confidence map通道数
    
    # ============ GNN架构参数 ============
    GNN_HIDDEN_DIM = 256
    GNN_NUM_LAYERS = 3
    NUM_ATTENTION_HEADS = 4
    DROPOUT_RATE = 0.1
    
    # ============ 训练参数 ============
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 500
    REPLAY_BUFFER_SIZE = 50000
    UPDATE_TARGET_EVERY = 10  # 多少个episode更新target network
    
    # ============ 损失函数权重 ============
    BELLMAN_LOSS_WEIGHT = 1.0
    TEMPORAL_CONSISTENCY_WEIGHT = 0.1
    SPATIAL_SMOOTHNESS_WEIGHT = 0.05
    
    # ============ 设备配置 ============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4  # DataLoader的工作进程数
    
    # ============ 保存路径 ============
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    RESULT_DIR = './results'
    
    @classmethod
    def print_config(cls):
        """打印所有配置"""
        print("=" * 50)
        print("Configuration:")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)
