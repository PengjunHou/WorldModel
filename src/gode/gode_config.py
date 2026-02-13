"""
GODE配置文件
针对你的数据格式和系统需求
"""


class GODEConfig:
    """GODE预测器配置"""
    
    # ==================== 数据路径 ====================
    DATA_PATH = "/home/peh324/Codes/WorldModel/data/gode_data/gode_dataset.npz"
    VAE_CHECKPOINT = "./checkpoints/vae/final_checkpoint.pth"
    VAE_N_CHECKPOINT = "./checkpoints/vae_n/final_checkpoint.pth"
    
    # ==================== 数据参数 ====================
    SEQUENCE_LENGTH = 5         # 历史序列长度T
    BATCH_SIZE = 32            # 批次大小
    NUM_WORKERS = 4            # 数据加载线程数
    TRAIN_RATIO = 0.8          # 训练集比例
    VAL_RATIO = 0.1            # 验证集比例
    
    # ==================== 模型参数 ====================
    # Map尺寸 (需要与你的数据一致)
    MAP_HEIGHT = 64
    MAP_WIDTH = 64
    MAP_SIZE = (MAP_HEIGHT, MAP_WIDTH)
    
    # VAE参数
    VAE_LATENT_DIM = 32        # VAE潜在维度 (需与预训练VAE一致)
    
    # 时序编码器
    TEMPORAL_HIDDEN_DIM = 128  # LSTM隐藏维度
    NUM_TEMPORAL_LAYERS = 2    # LSTM层数
    
    # GNN参数
    GNN_HIDDEN_DIM = 256       # GNN隐藏维度
    NUM_GNN_LAYERS = 3         # GNN层数
    
    # 其他
    K_EMBED_DIM = 16           # K值嵌入维度
    COMM_RANGE = 100.0         # 通信范围(米)
    POSITION_DIM = 2           # 位置维度(x, y)
    
    # ODE求解器
    ODE_METHOD = 'dopri5'      # 'euler', 'rk4', 'dopri5', 'adams'
    ODE_RTOL = 1e-3            # 相对误差
    ODE_ATOL = 1e-4            # 绝对误差
    ODE_ADJOINT = True         # 使用伴随方法
    INTEGRATION_TIME = 1.0     # 积分时间
    
    # ==================== 训练参数 ====================
    NUM_EPOCHS = 100           # 训练轮数
    LEARNING_RATE = 1e-3       # 学习率
    WEIGHT_DECAY = 1e-5        # L2正则化
    
    # 学习率调度
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_FACTOR = 0.5            # 衰减因子
    LR_PATIENCE = 5            # 耐心值
    
    # 损失权重
    MSE_WEIGHT = 1.0           # MSE损失权重
    MAE_WEIGHT = 0.1           # MAE损失权重
    
    # 梯度裁剪
    GRAD_CLIP_NORM = 1.0       # 梯度裁剪阈值
    
    # ==================== 早停 ====================
    EARLY_STOPPING = True      # 是否使用早停
    PATIENCE = 15              # 早停耐心值
    MIN_DELTA = 1e-4           # 最小改进量
    
    # ==================== 保存和日志 ====================
    SAVE_DIR = "./checkpoints/gode"           # 模型保存目录
    TENSORBOARD_DIR = "./runs/gode"           # TensorBoard目录
    SAVE_INTERVAL = 10         # 定期保存间隔
    VISUALIZE_INTERVAL = 5     # 可视化间隔
    
    # ==================== 设备 ====================
    DEVICE = 'cuda'            # 'cuda' or 'cpu'
    
    # ==================== 评估 ====================
    EVAL_METRICS = ['mse', 'mae', 'correlation']
    
    @classmethod
    def print_config(cls):
        """打印配置"""
        print("\n" + "="*60)
        print("GODE Configuration")
        print("="*60)
        
        sections = {
            'Data': ['DATA_PATH', 'SEQUENCE_LENGTH', 'BATCH_SIZE', 'TRAIN_RATIO', 'VAL_RATIO'],
            'Model': ['MAP_SIZE', 'VAE_LATENT_DIM', 'TEMPORAL_HIDDEN_DIM', 'GNN_HIDDEN_DIM', 
                     'NUM_GNN_LAYERS', 'K_EMBED_DIM', 'COMM_RANGE'],
            'Training': ['NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'EARLY_STOPPING', 'PATIENCE'],
            'Paths': ['SAVE_DIR', 'VAE_CHECKPOINT']
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            for key in keys:
                value = getattr(cls, key, 'N/A')
                print(f"  {key}: {value}")
        
        print("\n" + "="*60 + "\n")
    
    @classmethod
    def to_dict(cls):
        """转换为字典"""
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and k.isupper()
        }


# 快速训练配置(用于测试)
class FastConfig(GODEConfig):
    NUM_EPOCHS = 10
    BATCH_SIZE = 16
    SAVE_INTERVAL = 2
    VISUALIZE_INTERVAL = 2


# 高质量配置(用于最终训练)
class HighQualityConfig(GODEConfig):
    NUM_EPOCHS = 200
    BATCH_SIZE = 64
    TEMPORAL_HIDDEN_DIM = 256
    GNN_HIDDEN_DIM = 512
    NUM_GNN_LAYERS = 5
    LEARNING_RATE = 5e-4


if __name__ == "__main__":
    # 打印默认配置
    GODEConfig.print_config()
    
    # 保存配置
    import json
    config_dict = GODEConfig.to_dict()
    
    with open('gode_config.json', 'w') as f:
        # 过滤掉不能序列化的对象
        serializable_dict = {}
        for k, v in config_dict.items():
            if isinstance(v, (int, float, str, bool, list, dict, tuple)):
                serializable_dict[k] = v
        json.dump(serializable_dict, f, indent=2)
    
    print("✓ Config saved to gode_config.json")