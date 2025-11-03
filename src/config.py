"""
配置文件：ST-GAT Q网络
"""
import torch


class Config:
    """全局配置"""
    
    # ============ 环境参数 ============
    NUM_VEHICLES = 5          # 车辆数量，与环境相关
    MAP_HEIGHT = 16             # 地图高度，与环境相关
    MAP_WIDTH = 16              # 地图宽度，与环境相关
    # PERCEPTION_RANGE = 100.0    # 感知范围（米）
    COMM_RANGE = 150.0          # 通信范围（米）
    
    # ============ 时间参数 ============
    HISTORY_LENGTH = 5          # 历史时间步数
    MAX_EPISODE_LENGTH = 100    # 每个episode最大长度
    DISCOUNT_FACTOR = 0.95      # 折扣因子 β
    
    # ============ 网络架构参数 ============
    # VAE参数
    VAE_LATENT_DIM = 32
    
    # 兴趣度编码器参数
    INTEREST_ENCODER_DIM = 32
    
    # 位置编码器参数
    POSITION_ENCODER_DIM = 32
    
    # 特征融合后的维度
    FUSED_FEATURE_DIM = 256
    
    # ST-GAT参数
    STGAT_HIDDEN_DIM = 512
    STGAT_NUM_SPATIAL_LAYERS = 3
    STGAT_NUM_TEMPORAL_LAYERS = 3
    STGAT_NUM_HEADS = 4
    STGAT_DROPOUT = 0.1
    STGAT_USEWEIGHTS = True  # 是否使用边权重
    
    # Q值生成器参数
    Q_GENERATOR_TYPE = 'cnn'  # 'cnn' or 'diffusion'
    
    # ============ 训练参数 ============
    # VAE预训练
    VAE_NUM_SAMPLES = 20
    VAE_PRETRAIN_EPOCHS = 10
    VAE_PRETRAIN_BATCH_SIZE = 4
    VAE_PRETRAIN_LR = 0.001
    VAE_CHECKPOINT_DIR = '/home/peh324/Codes/WorldModel/src/extract/checkpoints'
    VAE_CHECKPOINT_PATH = '/home/peh324/Codes/WorldModel/src/extract/checkpoints/perception_vae.pth'
    VAE_VISUALIZE_DIR = '/home/peh324/Codes/WorldModel/src/extract/vae_vis'
    
    # Q网络训练
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    
    # RL参数
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    REPLAY_BUFFER_SIZE = 10000
    TARGET_UPDATE_FREQ = 1
    
    # Top-K选择
    TOP_K = 16  # 选择16个区域共享（25%）
    
    # ============ 损失权重 ============
    BELLMAN_LOSS_WEIGHT = 1.0
    SPATIAL_SMOOTH_WEIGHT = 0.1
    TEMPORAL_SMOOTH_WEIGHT = 0.05
    
    # ============ 设备配置 ============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4
    
    # ============ 保存路径 ============
    CHECKPOINT_DIR = './checkpoints'
    TENSORBOARD_DIR = './tensorboard'
    
    @classmethod
    def print_config(cls):
        """打印配置"""
        print("\n" + "="*60)
        print("Configuration")
        print("="*60)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:30s}: {value}")
        print("="*60 + "\n")
