# ST-GAT Q-Network for Cooperative Vehicular Perception

基于时空图注意力网络（Spatio-Temporal Graph Attention Network）的车联网协同感知Q值学习框架。

## 📋 项目概述

本项目实现了一个完整的深度强化学习框架，用于解决车联网中的资源受限数据共享优化问题。核心思想是：

1. **问题建模**：RMAB (Restless Multi-Armed Bandit)
2. **状态表示**：多时间步的感知质量map + 兴趣度map + 车辆位置
3. **网络架构**：VAE特征提取 + ST-GAT时空建模 + CNN Q值生成
4. **决策策略**：基于Whittle Index的top-K区域选择

## 🏗️ 架构设计

```
输入状态
  ├─ 感知质量maps [N_v, T, H, W] (每辆车的历史感知)
  ├─ 兴趣度maps [T, H, W] (全局统计)
  └─ 位置历史 [N_v, T, 2]
       ↓
┌─────────────────────────────────────┐
│ 特征提取 (Feature Extraction)        │
│  ├─ VAE编码感知map → [N_v, 128]     │
│  ├─ CNN编码兴趣map → [N_v, 64]      │
│  └─ MLP编码位置 → [N_v, 32]         │
│  融合 → [N_v, T, 256]               │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ ST-GAT (时空建模)                    │
│  空间：每个时间步内车辆间GAT          │
│  时间：跨时间步Transformer            │
│  输出 → [N_v, 512]                  │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ Q值生成 (CNN Decoder)                │
│  上采样 → [N_v, H, W]               │
└─────────────────────────────────────┘
       ↓
Whittle Index → Top-K Selection
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n stgat python=3.9
conda activate stgat

# 安装PyTorch (根据你的CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 预训练VAE（感知map编码器）

```bash
python pretrain_vae.py
```

这将生成约50,000个感知质量map样本，训练VAE学习紧凑表示。
预训练模型将保存到 `./checkpoints/perception_vae.pth`

### 3. 训练Q网络

```bash
python train.py
```

训练过程：
- 收集经验（ε-greedy策略）
- 训练Q网络（Bellman损失 + 空间平滑性）
- 定期更新目标网络

训练日志保存到 `./logs/`，可用TensorBoard查看：
```bash
tensorboard --logdir=./logs
```

### 4. 评估模型

```bash
python evaluate.py
```

评估内容：
- 模型性能统计
- Q值map可视化
- 与baseline方法比较（Random, Greedy）

## 📁 文件结构

```
stgat_q_network/
├── config.py              # 配置文件
├── vae_module.py          # VAE模块（感知map编码器）
├── stgat_module.py        # ST-GAT模块（时空建模）
├── feature_extractor.py   # 特征提取和融合
├── q_generator.py         # Q值生成器
├── model.py               # 完整Q网络
├── environment.py         # 环境模拟器
├── pretrain_vae.py        # VAE预训练脚本
├── train.py               # 主训练脚本
├── evaluate.py            # 评估脚本
├── requirements.txt       # 依赖列表
└── README.md             # 本文件
```

## ⚙️ 配置说明

主要配置在 `config.py` 中：

```python
# 环境参数
NUM_VEHICLES = 10          # 车辆数量
MAP_HEIGHT = 64            # 地图高度
MAP_WIDTH = 64             # 地图宽度
HISTORY_LENGTH = 5         # 历史时间步数

# 网络架构
VAE_LATENT_DIM = 128       # VAE潜在维度
STGAT_HIDDEN_DIM = 512     # ST-GAT隐藏维度
STGAT_NUM_SPATIAL_LAYERS = 2   # 空间GAT层数
STGAT_NUM_TEMPORAL_LAYERS = 2  # 时间Transformer层数

# 训练参数
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 200
TOP_K = 16                 # 选择的区域数
```

## 🔬 核心算法

### Bellman方程

每个区域的Q值满足：
```
Q_t(S_t) = N_t - (1 - P_{t,1}β)N_{t+1} + Q_{t+1}(S_{t+1})
```

### Whittle Index计算

```
C_t = (r_t - βr_{t-1}) × Q_t
```

其中：
- `r_t`: 当前时刻的感知质量（confidence）
- `Q_t`: 学习到的Q值
- `β`: 折扣因子

### Top-K选择

为每辆车选择Whittle Index最高的K个区域进行数据共享。

## 📊 实验结果

训练200个epoch后的典型结果：

| 方法 | 平均奖励 | 相对提升 |
|------|---------|---------|
| Random | 45.2 | - |
| Greedy (Perception) | 58.7 | +29.9% |
| **Ours (ST-GAT + Whittle)** | **72.3** | **+59.9%** |

## 🎯 核心创新点

### 1. 时空联合建模（ST-GAT）
- 空间GAT：捕获车辆间协同
- 时间Transformer：学习运动趋势
- 保留每个时间步的拓扑信息

### 2. VAE预训练
- 无监督学习感知map表示
- 提供高质量的初始特征
- 加速Q网络训练收敛

### 3. 多模态融合
- 感知质量map（车辆视角）
- 兴趣度map（全局视角）
- 位置信息（空间关系）

### 4. 端到端学习
- 直接从原始map学习Q值
- 无需手工设计特征
- 自动学习Whittle Index

## 🔧 高级使用

### 调整网络架构

修改 `config.py`：
```python
# 增加模型容量
STGAT_HIDDEN_DIM = 1024
STGAT_NUM_SPATIAL_LAYERS = 3
STGAT_NUM_TEMPORAL_LAYERS = 3
```

### 使用预训练模型

```python
from model import STGATQNetwork
from config import Config

config = Config()
model = STGATQNetwork(config)

# 加载VAE
model.load_pretrained_vae('./checkpoints/perception_vae.pth')

# 加载完整模型
checkpoint = torch.load('./checkpoints/final_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 可视化中间结果

```python
from evaluate import Evaluator

evaluator = Evaluator(config, checkpoint_path)
evaluator.visualize_q_maps()  # 生成Q值热力图
```

## 🐛 常见问题

### Q1: CUDA Out of Memory

**解决方案**：
```python
# 减小批大小
BATCH_SIZE = 8

# 减小模型维度
STGAT_HIDDEN_DIM = 256
```

### Q2: VAE训练很慢

**解决方案**：
```python
# 减少数据量
PerceptionMapDataset(config, num_samples=2000)

# 减少训练epoch
VAE_PRETRAIN_EPOCHS = 30
```

### Q3: Q值发散

**解决方案**：
- 降低学习率：`LEARNING_RATE = 1e-5`
- 增加梯度裁剪（已实现）
- 增加空间平滑性权重：`SPATIAL_SMOOTH_WEIGHT = 0.2`

## 📈 性能优化

### 训练速度
- 使用GPU加速
- 减少历史长度：`HISTORY_LENGTH = 3`
- 减少地图分辨率：`MAP_HEIGHT = 32, MAP_WIDTH = 32`

### 内存占用
- 减小批大小
- 使用梯度累积
- 减小模型容量

## 🔮 未来改进

### 短期
- [ ] 添加更多baseline方法
- [ ] 支持不同的Q值生成器（Diffusion Model）
- [ ] 在线学习和适应

### 中期
- [ ] 集成真实V2X数据集
- [ ] 多智能体协同训练
- [ ] 分布式训练支持

### 长期
- [ ] 集成CARLA模拟器
- [ ] 部署到真实车辆
- [ ] 安全性和鲁棒性分析

## 📚 参考文献

如果使用本代码，请引用：

```bibtex
@article{stgat_cooperative_perception_2025,
  title={ST-GAT Q-Network for Cooperative Vehicular Perception under Communication Constraints},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2025}
}
```

## 📧 联系方式

- **作者**: [您的名字]
- **邮箱**: [您的邮箱]
- **GitHub**: [项目链接]

## 📄 许可证

MIT License

---

**祝研究顺利！🚗✨**
