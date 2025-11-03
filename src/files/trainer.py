"""
训练器：实现Q网络的训练逻辑
包括经验回放、目标网络、损失函数等
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

from models import QNetworkGNN, create_model
from graph_builder import CooperativePerceptionGraph


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state_graph,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state_graph,
        done: bool,
        additional_info: Optional[Dict] = None
    ):
        """
        存储一条经验
        
        Args:
            state_graph: 当前状态的图
            action: 选择的动作（哪些区域被选择）
            reward: 获得的奖励
            next_state_graph: 下一状态的图
            done: 是否结束
            additional_info: 额外信息（如confidence值等）
        """
        experience = {
            'state': state_graph,
            'action': action,
            'reward': reward,
            'next_state': next_state_graph,
            'done': done,
            'info': additional_info or {}
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """随机采样一批经验"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class BellmanLoss(nn.Module):
    """Bellman方程损失函数"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(
        self,
        q_pred: torch.Tensor,
        q_target: torch.Tensor,
        importance_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算Bellman损失
        
        Args:
            q_pred: [N_r] 预测的Q值
            q_target: [N_r] 目标Q值
            importance_weights: [N_r] 重要性采样权重（可选）
        Returns:
            loss: 标量损失
        """
        # 使用Huber损失（对异常值更鲁棒）
        loss = self.huber_loss(q_pred, q_target)
        
        # 如果提供了重要性权重
        if importance_weights is not None:
            loss = (loss * importance_weights).mean()
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """时间一致性损失（鼓励相邻时间步的Q值平滑变化）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        q_current: torch.Tensor,
        q_prev: torch.Tensor,
        confidence_change: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q_current: [N_r] 当前Q值
            q_prev: [N_r] 上一时刻Q值
            confidence_change: [N_r] confidence变化量
        """
        # Q值的变化应该与confidence的变化相关
        q_change = q_current - q_prev
        
        # 如果confidence大幅变化，允许Q值大幅变化；否则惩罚Q值的大幅变化
        scale = torch.abs(confidence_change) + 1e-6
        normalized_q_change = q_change / scale
        
        loss = torch.mean(normalized_q_change ** 2)
        
        return loss


class SpatialSmoothnessLoss(nn.Module):
    """空间平滑性损失（鼓励相邻区域的Q值平滑）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        q_values: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q_values: [N_r] Q值
            edge_index: [2, E] 区域邻接边
        """
        # 计算相邻区域Q值的差异
        source_q = q_values[edge_index[0]]
        target_q = q_values[edge_index[1]]
        
        # L2平滑损失
        loss = torch.mean((source_q - target_q) ** 2)
        
        return loss


class QNetworkTrainer:
    """Q网络训练器"""
    
    def __init__(
        self,
        config,
        model_type: str = 'standard',
        use_double_dqn: bool = True
    ):
        self.config = config
        self.device = config.DEVICE
        self.use_double_dqn = use_double_dqn
        
        # ============ 创建模型 ============
        self.q_network = create_model(config, model_type).to(self.device)
        self.target_network = create_model(config, model_type).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # ============ 优化器 ============
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2
        )
        
        # ============ 损失函数 ============
        self.bellman_loss_fn = BellmanLoss(config)
        self.temporal_loss_fn = TemporalConsistencyLoss()
        self.spatial_loss_fn = SpatialSmoothnessLoss()
        
        # ============ 经验回放 ============
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        # ============ 日志记录 ============
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # ============ 统计信息 ============
        self.episode = 0
        self.total_steps = 0
        self.best_performance = float('-inf')
        
        # 图构建器
        self.graph_builder = CooperativePerceptionGraph(config)
    
    def compute_target_q(
        self,
        batch: List[Dict]
    ) -> torch.Tensor:
        """
        计算目标Q值
        使用Bellman方程: Q(s,a) = r + γ * max_a' Q(s',a')
        
        对于RMAB问题，我们需要计算：
        Q_t = N_t + E[β^n * N_{t+n}] + β * Q_{t+1}
        """
        target_q_list = []
        
        with torch.no_grad():
            for experience in batch:
                next_graph = experience['next_state'].to(self.device)
                reward = experience['reward'].to(self.device)
                done = experience['done']
                
                if done:
                    # 终端状态
                    target_q = reward
                else:
                    # 计算下一状态的Q值
                    if self.use_double_dqn:
                        # Double DQN: 使用当前网络选择动作，目标网络评估
                        next_q_pred = self.q_network(next_graph)
                        next_q_target = self.target_network(next_graph)
                        target_q = reward + self.config.DISCOUNT_FACTOR * next_q_target
                    else:
                        # 标准DQN
                        next_q = self.target_network(next_graph)
                        target_q = reward + self.config.DISCOUNT_FACTOR * next_q
                
                target_q_list.append(target_q)
        
        return torch.stack(target_q_list)
    
    def compute_bellman_target(
        self,
        experience: Dict
    ) -> torch.Tensor:
        """
        根据RMAB的Bellman方程计算目标
        Q_t(S_t) = N_t - (1 - P_{t,1}β^n)N_{t+1} + Q_{t+1}(S_{t+1})
        """
        info = experience['info']
        N_t = info['current_interest_count']
        N_t1 = info['next_interest_count']
        P_t1 = info['transition_prob']
        beta = self.config.DISCOUNT_FACTOR
        
        with torch.no_grad():
            next_graph = experience['next_state'].to(self.device)
            Q_t1 = self.target_network(next_graph)
            
            # 根据公式计算
            target = N_t - (1 - P_t1 * beta) * N_t1 + Q_t1
        
        return target
    
    def train_step(self, batch_size: int) -> Dict[str, float]:
        """
        执行一步训练
        
        Returns:
            losses: 各项损失的字典
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # 采样batch
        batch = self.replay_buffer.sample(batch_size)
        
        # 准备数据
        state_graphs = [exp['state'].to(self.device) for exp in batch]
        
        # ============ 前向传播 ============
        q_pred_list = []
        for graph in state_graphs:
            q_pred = self.q_network(graph)
            q_pred_list.append(q_pred)
        
        # ============ 计算目标Q值 ============
        target_q_list = []
        for experience in batch:
            target_q = self.compute_bellman_target(experience)
            target_q_list.append(target_q)
        
        # ============ 计算损失 ============
        losses = {}
        
        # 1. Bellman损失
        total_bellman_loss = 0
        for q_pred, q_target in zip(q_pred_list, target_q_list):
            bellman_loss = self.bellman_loss_fn(q_pred, q_target)
            total_bellman_loss += bellman_loss
        
        total_bellman_loss /= len(batch)
        losses['bellman_loss'] = total_bellman_loss.item()
        
        # 2. 时间一致性损失
        if self.config.TEMPORAL_CONSISTENCY_WEIGHT > 0:
            temporal_loss = 0
            count = 0
            for i, exp in enumerate(batch):
                if 'prev_q' in exp['info']:
                    prev_q = exp['info']['prev_q'].to(self.device)
                    confidence_change = exp['info']['confidence_change'].to(self.device)
                    temporal_loss += self.temporal_loss_fn(
                        q_pred_list[i],
                        prev_q,
                        confidence_change
                    )
                    count += 1
            
            if count > 0:
                temporal_loss /= count
                losses['temporal_loss'] = temporal_loss.item()
            else:
                temporal_loss = torch.tensor(0.0, device=self.device)
        else:
            temporal_loss = torch.tensor(0.0, device=self.device)
        
        # 3. 空间平滑性损失
        if self.config.SPATIAL_SMOOTHNESS_WEIGHT > 0:
            spatial_loss = 0
            for i, graph in enumerate(state_graphs):
                edge_index = graph['region', 'adjacent', 'region'].edge_index
                spatial_loss += self.spatial_loss_fn(q_pred_list[i], edge_index)
            
            spatial_loss /= len(batch)
            losses['spatial_loss'] = spatial_loss.item()
        else:
            spatial_loss = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = (
            self.config.BELLMAN_LOSS_WEIGHT * total_bellman_loss +
            self.config.TEMPORAL_CONSISTENCY_WEIGHT * temporal_loss +
            self.config.SPATIAL_SMOOTHNESS_WEIGHT * spatial_loss
        )
        
        losses['total_loss'] = total_loss.item()
        
        # ============ 反向传播 ============
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.total_steps += 1
        
        return losses
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_epoch(
        self,
        num_episodes: int,
        batch_size: int,
        update_target_every: int = 10
    ):
        """
        训练一个epoch
        
        Args:
            num_episodes: episode数量
            batch_size: 批大小
            update_target_every: 多少个episode更新目标网络
        """
        self.q_network.train()
        
        epoch_losses = {
            'bellman_loss': [],
            'temporal_loss': [],
            'spatial_loss': [],
            'total_loss': []
        }
        
        pbar = tqdm(range(num_episodes), desc="Training")
        
        for episode in pbar:
            # 训练多步
            for _ in range(10):  # 每个episode训练10步
                losses = self.train_step(batch_size)
                
                if losses:
                    for key, value in losses.items():
                        epoch_losses[key].append(value)
            
            # 更新目标网络
            if episode % update_target_every == 0:
                self.update_target_network()
            
            # 更新进度条
            if epoch_losses['total_loss']:
                avg_loss = np.mean(epoch_losses['total_loss'][-10:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # 记录到TensorBoard
            if losses:
                for key, value in losses.items():
                    self.writer.add_scalar(f'train/{key}', value, self.total_steps)
            
            self.episode += 1
        
        # 返回平均损失
        return {key: np.mean(values) if values else 0 
                for key, values in epoch_losses.items()}
    
    def evaluate(self, eval_data_loader) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            eval_data_loader: 评估数据加载器
        Returns:
            metrics: 评估指标字典
        """
        self.q_network.eval()
        
        total_mse = 0
        total_mae = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_data_loader:
                graphs = batch['graphs']
                true_q = batch['true_q'].to(self.device)
                
                for graph in graphs:
                    graph = graph.to(self.device)
                    pred_q = self.q_network(graph)
                    
                    mse = F.mse_loss(pred_q, true_q)
                    mae = F.l1_loss(pred_q, true_q)
                    
                    total_mse += mse.item()
                    total_mae += mae.item()
                    num_samples += 1
        
        metrics = {
            'mse': total_mse / num_samples if num_samples > 0 else 0,
            'mae': total_mae / num_samples if num_samples > 0 else 0
        }
        
        # 记录到TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'eval/{key}', value, self.episode)
        
        return metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_performance': self.best_performance
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.episode = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.best_performance = checkpoint['best_performance']
        
        print(f"Loaded checkpoint from episode {self.episode}")
