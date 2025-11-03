# 快速入门指南

## 5分钟快速开始

### 步骤1: 安装依赖

```bash
# 克隆或下载项目后，进入项目目录
cd cooperative_perception_gnn

# 安装PyTorch (选择合适的版本)
# CPU版本:
pip install torch torchvision torchaudio

# GPU版本 (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 安装其他依赖
pip install -r requirements.txt
```

### 步骤2: 运行示例

```bash
# 运行所有示例（推荐第一次使用）
python example.py
```

这将演示：
- ✅ 基本使用流程
- ✅ 完整episode运行
- ✅ 结果可视化
- ✅ 不同模型架构比较
- ✅ 超参数影响分析

### 步骤3: 训练模型

```bash
# 使用默认配置训练（标准Q网络，100个epoch）
python train.py --mode train --num_epochs 100

# 使用Dueling架构 + Double DQN
python train.py --mode train --model_type dueling --use_double_dqn --num_epochs 100
```

训练过程中会自动保存：
- 检查点: `./checkpoints/checkpoint_epoch_*.pth`
- TensorBoard日志: `./logs/`
- 可视化结果: `./results/visualizations/`

### 步骤4: 监控训练

```bash
# 在另一个终端启动TensorBoard
tensorboard --logdir=./logs

# 在浏览器打开 http://localhost:6006
```

### 步骤5: 评估模型

```bash
# 评估训练好的模型
python train.py --mode eval --checkpoint_path ./checkpoints/final_model.pth --num_eval_episodes 50
```

## 常用命令速查

### 训练相关

```bash
# 从检查点恢复训练
python train.py --mode train --resume --checkpoint_path ./checkpoints/checkpoint_epoch_50.pth

# 每5个epoch保存一次
python train.py --mode train --save_every 5

# 使用Ensemble模型
python train.py --mode train --model_type ensemble
```

### 自定义配置

编辑 `config.py` 修改超参数：

```python
class Config:
    # 环境配置
    NUM_VEHICLES = 10      # 车辆数量
    NUM_REGIONS = 64       # 区域数量 (8x8 grid)
    PERCEPTION_RANGE = 100 # 感知范围 (米)
    
    # GNN配置
    GNN_HIDDEN_DIM = 256   # 隐藏层维度
    GNN_NUM_LAYERS = 3     # GNN层数
    NUM_ATTENTION_HEADS = 4 # 注意力头数
    
    # 训练配置
    LEARNING_RATE = 1e-4   # 学习率
    BATCH_SIZE = 32        # 批大小
    DISCOUNT_FACTOR = 0.95 # 折扣因子 β
```

## 核心概念

### 1. 状态 (State)

包括：
- **车辆数据**: 位置、速度、朝向、历史感知地图
- **区域数据**: confidence值、感兴趣车辆数
- **语义地图**: 环境的语义信息

### 2. 动作 (Action)

选择K个区域进行数据共享（二值向量）

### 3. Q值 (Q-Value)

每个区域的价值函数，满足Bellman方程：
```
Q_t = N_t - (1 - P_{t,1}β)N_{t+1} + Q_{t+1}
```

### 4. Whittle Index

决策指标：
```
C_t = (r_t - βr_{t-1}) × Q_t
```
选择Whittle Index最高的K个区域

## 项目结构

```
cooperative_perception_gnn/
├── config.py              # 配置文件 ⚙️
├── graph_builder.py       # 图构建 🔗
├── models.py              # GNN模型 🧠
├── trainer.py             # 训练器 🎓
├── data_generator.py      # 数据生成 📊
├── train.py               # 主训练脚本 🚀
├── example.py             # 使用示例 📖
├── visualization.py       # 可视化工具 📈
├── requirements.txt       # 依赖列表 📦
├── README.md             # 详细文档 📚
└── ARCHITECTURE.md       # 架构文档 🏗️
```

## 实验流程建议

### Phase 1: 验证框架 (1-2天)

1. 运行 `example.py` 确保一切正常
2. 训练一个小模型（10 epochs）
3. 检查可视化结果

### Phase 2: 初步实验 (3-5天)

1. 尝试不同的模型架构（standard, dueling, ensemble）
2. 调整超参数（hidden_dim, num_layers）
3. 比较不同K值的影响

### Phase 3: 完整实验 (1-2周)

1. 长时间训练（100+ epochs）
2. 与baseline方法比较：
   - Random selection
   - Greedy (highest confidence)
   - Fixed priority
3. 消融实验：
   - 不同损失函数的影响
   - Double DQN vs 标准DQN
   - 不同GNN层数

### Phase 4: 分析与优化 (1周)

1. 性能分析（推理时间、内存使用）
2. 可视化分析（注意力权重、Q值分布）
3. 错误案例分析
4. 撰写实验报告

## 常见问题快速解决

### Q: 报错 "CUDA out of memory"

```python
# 在config.py中减小这些参数
BATCH_SIZE = 16
GNN_HIDDEN_DIM = 128
```

### Q: 训练不收敛

```python
# 降低学习率
LEARNING_RATE = 1e-5

# 增加目标网络更新频率
UPDATE_TARGET_EVERY = 5
```

### Q: 想要更快的训练

```python
# 减少环境复杂度
NUM_VEHICLES = 5
NUM_REGIONS = 16  # 4x4 grid
MAX_EPISODE_LENGTH = 50
```

### Q: 需要更好的性能

```python
# 增加模型容量
GNN_HIDDEN_DIM = 512
GNN_NUM_LAYERS = 5
NUM_ATTENTION_HEADS = 8

# 使用Ensemble
model_type = 'ensemble'
```

## 下一步

1. 📖 阅读 `README.md` 了解详细功能
2. 🏗️ 阅读 `ARCHITECTURE.md` 了解架构设计
3. 🧪 运行自己的实验
4. 📊 分析实验结果
5. 📝 撰写研究论文

## 获取帮助

- 查看 `example.py` 中的使用示例
- 查看 `ARCHITECTURE.md` 了解实现细节
- 查看代码注释获取函数级文档

## 小贴士

💡 **Tip 1**: 先用小规模参数快速验证想法，再进行大规模实验

💡 **Tip 2**: 定期保存检查点，避免训练中断导致进度丢失

💡 **Tip 3**: 使用TensorBoard实时监控训练，及时发现问题

💡 **Tip 4**: 可视化中间结果（Q值、Whittle Index）帮助理解模型行为

💡 **Tip 5**: 与简单baseline比较，证明方法的有效性

Good luck with your research! 🚀
