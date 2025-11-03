# 项目架构总结

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户接口层                                 │
├─────────────────────────────────────────────────────────────────┤
│  train.py (训练脚本)  │  example.py (示例)  │  visualization.py  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        核心训练层                                 │
├─────────────────────────────────────────────────────────────────┤
│              trainer.py (QNetworkTrainer)                        │
│  • 经验回放 (ReplayBuffer)                                       │
│  • 损失函数 (Bellman, Temporal, Spatial)                        │
│  • 训练循环 (train_step, train_epoch)                           │
│  • 模型更新 (target network, optimizer)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        模型层                                     │
├─────────────────────────────────────────────────────────────────┤
│                    models.py                                     │
│  ┌──────────────────┬──────────────────┬──────────────────┐    │
│  │  QNetworkGNN     │  DuelingQNetwork │ EnsembleQNetwork │    │
│  │  (标准架构)       │  (Dueling架构)    │  (集成多个网络)   │    │
│  └──────────────────┴──────────────────┴──────────────────┘    │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────┐      │
│  │          异构图神经网络层                              │      │
│  │  • VehicleEncoder (车辆特征编码)                      │      │
│  │  • RegionEncoder (区域特征编码)                       │      │
│  │  • HeteroGNNLayer (多关系图卷积)                      │      │
│  │    - vehicle-vehicle (通信)                          │      │
│  │    - region-region (邻接)                            │      │
│  │    - vehicle-region (感知)                           │      │
│  │    - region-vehicle (兴趣)                           │      │
│  │  • Q Predictor (Q值预测头)                            │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        图构建层                                   │
├─────────────────────────────────────────────────────────────────┤
│            graph_builder.py (CooperativePerceptionGraph)         │
│  • 节点特征编码                                                  │
│    - 车辆：位置、速度、朝向、历史感知地图                         │
│    - 区域：confidence、兴趣数、位置、语义                        │
│  • 边构建                                                        │
│    - 基于距离的车辆通信边                                        │
│    - 网格邻接的区域边                                            │
│    - 感知范围内的车辆-区域边                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        数据生成层                                 │
├─────────────────────────────────────────────────────────────────┤
│                  data_generator.py                               │
│  ┌────────────────────────────────────────────────────┐         │
│  │          VehicularEnvironment (环境模拟器)          │         │
│  │  • 车辆初始化与更新                                 │         │
│  │  • 区域状态演化                                     │         │
│  │  • 感知地图生成                                     │         │
│  │  • 奖励计算                                         │         │
│  └────────────────────────────────────────────────────┘         │
│  ┌────────────────────────────────────────────────────┐         │
│  │    CooperativePerceptionDataset (数据集)           │         │
│  │  • 预生成模式 / 在线生成模式                        │         │
│  │  • 经验元组构建 (s, a, r, s', done)                │         │
│  └────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        配置层                                     │
├─────────────────────────────────────────────────────────────────┤
│                      config.py (Config)                          │
│  • 环境参数 (车辆数、区域数、感知范围)                           │
│  • 模型参数 (隐藏层、层数、注意力头)                             │
│  • 训练参数 (学习率、批大小、折扣因子)                           │
│  • 损失权重 (Bellman、时间一致性、空间平滑性)                    │
└─────────────────────────────────────────────────────────────────┘
```

## 数据流程图

```
开始
  ↓
环境初始化 (VehicularEnvironment.reset)
  ↓
获取状态 s_t = {vehicle_data, region_data, semantic_map}
  ↓
构建异构图 (CooperativePerceptionGraph.build_graph)
  ↓
前向传播 (QNetworkGNN.forward)
  ├→ Vehicle Encoder → vehicle features
  ├→ Region Encoder → region features
  ├→ Multi-layer HeteroGNN → updated embeddings
  └→ Q Predictor → Q values [N_r]
  ↓
计算Whittle Index: C_t = (r_t - β*r_{t-1}) * Q_t
  ↓
选择Top-K区域 (基于Whittle Index排序)
  ↓
执行动作 a_t (Environment.step)
  ├→ 更新车辆位置
  ├→ 更新区域confidence
  └→ 计算奖励 r_t
  ↓
获取下一状态 s_{t+1}
  ↓
存储经验 (state, action, reward, next_state, done)
到 ReplayBuffer
  ↓
从ReplayBuffer采样批次
  ↓
计算目标Q值 (使用Target Network)
Q_target = N_t - (1-P_{t,1}β)N_{t+1} + β*Q_{t+1}
  ↓
计算损失
  ├→ Bellman Loss: ||Q_pred - Q_target||²
  ├→ Temporal Consistency Loss
  └→ Spatial Smoothness Loss
  ↓
反向传播 + 优化器更新
  ↓
每N个episode更新Target Network
  ↓
是否结束？
  ├→ 否: 返回"获取状态"
  └→ 是: 保存模型 → 结束
```

## 关键算法流程

### 1. Bellman方程求解

```
输入: 当前状态图 G_t
输出: 每个区域的Q值

1. 编码节点特征
   vehicle_emb = VehicleEncoder(vehicle_features)
   region_emb = RegionEncoder(region_features)

2. 多层图卷积
   For layer in GNN_layers:
       vehicle_emb, region_emb = HeteroConv(
           vehicle_emb, region_emb, edge_index_dict
       )

3. 预测Q值
   Q_t = MLP(region_emb)

4. 计算目标 (训练时)
   Q_target = N_t - (1 - P_{t,1}*β)*N_{t+1} + Q_{t+1}(s_{t+1})
```

### 2. Whittle Index计算与Top-K选择

```
输入: Q值 [N_r], confidence值 [N_r]
输出: 选择的K个区域

1. 计算Whittle Index
   For each region i:
       C_i = (r_i^t - β*r_i^{t-1}) * Q_i

2. 排序并选择Top-K
   indices = argsort(C, descending=True)
   selected = indices[:K]

3. 构建动作向量
   action = zeros(N_r)
   action[selected] = 1
```

### 3. 训练循环

```
For epoch in range(num_epochs):
    # 收集经验
    For episode in range(num_episodes_per_epoch):
        state = env.reset()
        For step in range(max_steps):
            # 构建图
            graph = build_graph(state)
            
            # ε-greedy策略
            if random() < epsilon:
                action = random_action()
            else:
                q = model(graph)
                whittle = compute_whittle_index(q)
                action = select_top_k(whittle)
            
            # 执行并存储
            next_state, reward, done = env.step(action)
            replay_buffer.push(graph, action, reward, next_graph, done)
    
    # 训练网络
    For _ in range(training_steps):
        batch = replay_buffer.sample(batch_size)
        loss = compute_loss(batch)
        optimizer.step()
    
    # 更新目标网络
    if epoch % update_freq == 0:
        target_network.load(model)
```

## 核心技术点

### 1. 异构图神经网络

**优势**:
- 自然建模多类型实体（车辆、区域）
- 捕获不同类型的关系（通信、邻接、感知）
- 处理变长输入（车辆数可变）

**实现**:
- 使用PyTorch Geometric的HeteroConv
- 每种边类型使用独立的GAT层
- 残差连接 + LayerNorm提高训练稳定性

### 2. Multi-task损失函数

**组成**:
1. **Bellman Loss**: 确保Q值满足Bellman方程
2. **Temporal Consistency**: 鼓励时间上的平滑变化
3. **Spatial Smoothness**: 鼓励空间上的平滑变化

**平衡**:
```python
Total = w1*Bellman + w2*Temporal + w3*Spatial
```

### 3. Double DQN

**目的**: 减少Q值过估计

**实现**:
- 主网络选择动作
- 目标网络评估Q值
- 定期更新目标网络

### 4. 经验回放

**作用**:
- 打破数据相关性
- 提高样本利用效率
- 稳定训练过程

## 可扩展性设计

### 1. 模块化架构
- 每个模块独立，易于替换和扩展
- 清晰的接口定义

### 2. 配置驱动
- 所有超参数集中在config.py
- 易于调参和实验

### 3. 工厂模式
```python
model = create_model(config, model_type='standard')
```

### 4. 插件式损失函数
```python
class CustomLoss(nn.Module):
    def forward(self, ...):
        pass

trainer.add_loss('custom', CustomLoss(), weight=0.1)
```

## 性能优化

### 1. 计算优化
- GPU加速（CUDA）
- 批处理（尽管图数据受限）
- 梯度累积

### 2. 内存优化
- 经验回放缓冲区限制大小
- 定期清理不需要的计算图

### 3. 并行化
- 多进程数据生成（DataLoader）
- 可能的分布式训练

## 实验建议

### 1. 基准对比
- Random
- Greedy (最高confidence)
- Round-robin
- 本方法

### 2. 消融实验
- 不同GNN层数的影响
- 不同损失项的影响
- Double DQN vs 标准DQN

### 3. 泛化性测试
- 不同车辆数
- 不同感知范围
- 不同K值

### 4. 实时性测试
- 推理时间
- 训练收敛速度

## 未来扩展方向

1. **更真实的环境建模**
   - 真实道路网络
   - V2X通信延迟
   - 数据包丢失

2. **更复杂的决策**
   - 连续动作空间（共享比例）
   - 多目标优化
   - 安全性约束

3. **在线学习**
   - 持续适应
   - 增量学习
   - 元学习

4. **分布式训练**
   - 多GPU训练
   - 联邦学习

5. **与真实数据集集成**
   - CARLA模拟器
   - nuScenes数据集
   - Waymo数据集
