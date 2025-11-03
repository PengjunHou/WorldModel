# GNN-based Q-Network for Cooperative Vehicular Perception

## 项目概述

这是一个基于图神经网络（GNN）的Q网络学习框架，用于解决车联网协同感知中的资源受限数据共享优化问题。该项目将问题建模为Restless Multi-Armed Bandit (RMAB)，通过学习Whittle Index来确定最优的top-K区域共享策略。

### 核心特性

- **异构图神经网络**：建模车辆-车辆、区域-区域、车辆-区域之间的复杂交互关系
- **Whittle Index学习**：直接学习Q值函数，计算Whittle Index指导区域选择
- **时空一致性约束**：通过多任务损失函数保证Q值的时间和空间平滑性
- **Double DQN**：使用目标网络和Double DQN技术提高学习稳定性
- **灵活的模型架构**：支持标准、Dueling、Ensemble等多种模型变体

## 目录结构

```
cooperative_perception_gnn/
├── config.py              # 配置文件
├── graph_builder.py       # 图构建模块
├── models.py              # GNN模型定义
├── trainer.py             # 训练器
├── data_generator.py      # 环境模拟和数据生成
├── train.py               # 主训练脚本
├── visualization.py       # 可视化工具
├── requirements.txt       # 依赖包
└── README.md             # 本文件
```

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (推荐，用于GPU加速)

### 安装步骤

1. 创建虚拟环境（推荐）：
```bash
conda create -n coop_perception python=3.9
conda activate coop_perception
```

2. 安装PyTorch和PyTorch Geometric：
```bash
# CPU版本
pip install torch torchvision torchaudio

# GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

3. 安装其他依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型

使用默认配置训练标准Q网络：

```bash
python train.py --mode train --num_epochs 100
```

训练Dueling Q网络：

```bash
python train.py --mode train --model_type dueling --use_double_dqn --num_epochs 100
```

### 2. 评估模型

```bash
python train.py --mode eval --checkpoint_path ./checkpoints/final_model.pth --num_eval_episodes 50
```

### 3. 可视化结果

```python
from visualization import Visualizer
from config import Config

config = Config()
visualizer = Visualizer(config)

# 可视化Q值热力图
visualizer.visualize_q_values(q_values, save_name='q_values.png')

# 可视化Whittle Index
visualizer.visualize_whittle_index(whittle_indices, selected_regions)

# 可视化训练曲线
visualizer.plot_training_curves('./logs')
```

## 核心模块说明

### 1. 配置模块 (config.py)

包含所有超参数和模型配置：
- 环境参数：车辆数、区域数、感知范围等
- GNN架构参数：隐藏层维度、层数、注意力头数等
- 训练参数：学习率、批大小、折扣因子等

### 2. 图构建模块 (graph_builder.py)

`CooperativePerceptionGraph`类负责构建异构图：

**输入特征**：
- 车辆：位置、速度、朝向、历史感知地图
- 区域：confidence值、感兴趣车辆数、位置、语义特征

**边类型**：
- `vehicle-communicates-vehicle`: 车辆间通信
- `region-adjacent-region`: 区域空间邻接
- `vehicle-perceives-region`: 车辆感知区域
- `region-interests-vehicle`: 区域被车辆感兴趣

### 3. 模型模块 (models.py)

#### QNetworkGNN
标准的异构图神经网络Q值函数：
- 输入：异构图 (HeteroData)
- 输出：每个区域的Q值 [N_r]

架构：
```
VehicleEncoder → 
RegionEncoder → 
Multi-layer HeteroGNN (GAT) → 
Q Predictor → 
Q values
```

#### DuelingQNetwork
将Q值分解为状态值V和优势函数A：
```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

#### EnsembleQNetwork
集成多个Q网络减少估计方差。

### 4. 训练器 (trainer.py)

`QNetworkTrainer`类实现完整的训练流程：

**核心方法**：
- `compute_bellman_target()`: 根据RMAB的Bellman方程计算目标Q值
- `train_step()`: 单步训练，计算损失并更新参数
- `update_target_network()`: 更新目标网络

**损失函数**：
```python
Total Loss = w1 * Bellman Loss 
           + w2 * Temporal Consistency Loss 
           + w3 * Spatial Smoothness Loss
```

其中：
- **Bellman Loss**: Q值与目标的MSE/Huber损失
- **Temporal Consistency Loss**: 相邻时间步Q值的平滑性
- **Spatial Smoothness Loss**: 相邻区域Q值的平滑性

### 5. 数据生成模块 (data_generator.py)

#### VehicularEnvironment
模拟车联网环境：
- 车辆移动（恒速运动 + 随机转向）
- 感知地图生成（基于距离衰减）
- 区域状态更新（confidence演化）
- 奖励计算（感知质量 vs 通信开销）

#### CooperativePerceptionDataset
生成训练数据集：
- 支持预生成和在线生成两种模式
- 自动构建图结构
- 包含完整的经验元组 (s, a, r, s', done)

## 算法原理

### 问题建模

车联网协同感知优化问题被建模为RMAB：

**状态空间**：
- 车辆状态：位置、速度、历史感知
- 区域状态：confidence、感兴趣车辆数

**动作空间**：
- 选择top-K个区域进行共享

**奖励函数**：
```
R = Σ(interest_i × confidence_i) - λ × num_shared_regions
```

### Bellman方程

根据RMAB理论，每个区域的Q值满足：

```
Q_t(S_t) = N_t - (1 - P_{t,1}β)N_{t+1} + Q_{t+1}(S_{t+1})
```

其中：
- N_t: 时刻t感兴趣的车辆数
- P_{t,j}: 转移概率
- β: 折扣因子

### Whittle Index

Whittle Index计算：
```
C_t = (r_t - βr_{t-1}) × Q_t
```

其中r_t是区域的confidence值。

**决策规则**：选择Whittle Index最高的K个区域进行共享。

### GNN架构优势

1. **捕获空间关系**：通过图卷积聚合邻域信息
2. **处理变长输入**：自然支持不同数量的车辆和区域
3. **学习交互模式**：注意力机制学习重要的交互
4. **泛化能力**：学到的策略可以迁移到不同场景

## 实验结果

（这里可以添加您的实验结果）

### 基准方法比较

- **Random**: 随机选择K个区域
- **Greedy Confidence**: 选择confidence最高的K个
- **Fixed Priority**: 基于固定优先级
- **Ours (GNN-Whittle)**: 本方法

### 性能指标

- 平均累积奖励
- 通信开销
- 感知质量（mAP）
- 计算时间

## 进阶使用

### 自定义环境

修改`data_generator.py`中的`VehicularEnvironment`类：

```python
class CustomEnvironment(VehicularEnvironment):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定义初始化
    
    def _compute_reward(self, actions):
        # 自定义奖励函数
        pass
```

### 调整模型架构

修改`config.py`中的参数：

```python
class Config:
    GNN_HIDDEN_DIM = 512  # 增加隐藏层维度
    GNN_NUM_LAYERS = 5    # 增加GNN层数
    NUM_ATTENTION_HEADS = 8  # 更多注意力头
```

### 添加新的损失函数

在`trainer.py`中添加：

```python
class CustomLoss(nn.Module):
    def forward(self, q_pred, additional_info):
        # 实现自定义损失
        pass

# 在训练器中使用
self.custom_loss_fn = CustomLoss()
```

## 常见问题

### Q1: CUDA Out of Memory

**解决方案**：
- 减小批大小：`config.BATCH_SIZE = 16`
- 减小模型维度：`config.GNN_HIDDEN_DIM = 128`
- 使用梯度累积

### Q2: 训练不稳定/发散

**解决方案**：
- 降低学习率：`config.LEARNING_RATE = 1e-5`
- 增加目标网络更新频率：`config.UPDATE_TARGET_EVERY = 5`
- 使用梯度裁剪（已实现）
- 增加经验回放缓冲区大小

### Q3: Q值估计不准确

**解决方案**：
- 收集更多数据
- 使用Ensemble方法
- 调整时间一致性和空间平滑性权重
- 检查转移概率P_{t,j}的估计是否准确

## 未来改进方向

1. **更复杂的车辆动态**：考虑加速度、道路约束等
2. **真实数据集**：使用真实的V2X通信数据
3. **在线学习**：支持在线更新和适应
4. **多智能体强化学习**：考虑车辆间的策略交互
5. **安全性约束**：保证关键信息的可靠传输

## 引用

如果您使用本代码，请引用：

```bibtex
@article{your_paper,
  title={GNN-based Whittle Index Learning for Cooperative Vehicular Perception},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: your.email@example.com
- GitHub Issues: [项目链接]

## 致谢

感谢所有为本项目做出贡献的人员和机构。
