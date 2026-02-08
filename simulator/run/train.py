import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm
from collections import deque
import random
from env import make_env
from utils import PARSER, init_log
from ...agent import STGATQNetwork
from ...agent import ReplayBuffer

class Trainer:
    """Q网络训练器"""
    def __init__(self, env, config):
        self.config = config,
        self.model_config = config.stgat
        self.env_config = config.env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型
        self.q_network = STGATQNetwork(env_config=self.env_config, model_config=self.model_config).to(self.device)
        self.target_network = STGATQNetwork(env_config=self.env_config, model_config=self.model_config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 加载预训练的VAE
        if os.path.exists(self.model_config.vae_pretrain.conf_map.checkpoint_path):
            self.q_network.load_pretrained_vae(self.model_config.vae_pretrain.conf_map.checkpoint_path, self.model_config.vae_pretrain.interested_map.checkpoint_path)
        else:
            print("⚠ VAE checkpoint not found. Please run pretrain_vae.py first!")
        
        # 优化器
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.q_network.parameters()),
            lr=self.model_config.learning_rate
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(self.model_config.replay_buffer_size)
        
        # 环境
        self.env = env # make_env(args=env_config, dream_env=False, render_mode=False)

        # TensorBoard
        self.writer = SummaryWriter(self.model_config.logging.tensorboard_dir)
        
        # 统计
        self.episode = 0
        self.total_steps = 0
        self.epsilon = self.model_config.rl.epsilon.start
    
    def compute_bellman_loss(self, batch):
        """计算Bellman损失"""
        losses = []
        
        for exp in batch:
            # 当前Q值
            q_pred = self.q_network(exp['state'])  # ✅ 重新 forward，可求导
            
            # 目标Q值
            if exp['done']:
                q_target = torch.zeros_like(q_pred)
            else:
                with torch.no_grad():
                    next_state = exp['next_state']
                    q_next = self.target_network(next_state)
                    
                    # Q_target = reward + γ * Q_next
                    q_target = exp['reward'] + self.model_config.DISCOUNT_FACTOR * q_next
            
            # Huber损失
            loss = F.smooth_l1_loss(q_pred, q_target)
            losses.append(loss)
        
        return torch.stack(losses).mean()
    
    def compute_spatial_smoothness_loss(self, q_maps):
        """空间平滑性损失"""
        # 计算梯度
        grad_x = q_maps[:, :, 1:] - q_maps[:, :, :-1]
        grad_y = q_maps[:, 1:, :] - q_maps[:, :-1, :]
        
        loss = torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2)
        return loss
    
    def train_step(self):
        """单步训练"""
        if len(self.replay_buffer) < self.model_config.batch_size:
            return {}
        
        # 采样
        batch = self.replay_buffer.sample(self.model_config.batch_size)
        
        # 计算损失
        bellman_loss = self.compute_bellman_loss(batch)
        
        # 空间平滑性
        q_maps_batch = torch.stack([exp['q_maps'] for exp in batch])
        spatial_loss = self.compute_spatial_smoothness_loss(q_maps_batch)
        
        # 总损失
        total_loss = (
            self.model_config.loss.bellman * bellman_loss +
            self.model_config.loss.spatial_smooth * spatial_loss
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        self.total_steps += 1
        
        return {
            'bellman_loss': bellman_loss.item(),
            'spatial_loss': spatial_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def collect_episode(self):
        """收集一个episode的经验"""
        obs, info = self.env.reset()
        states = self.env.wrapper_state(obs)
        # 移动到设备
        states = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                        for k, v in states.items()}
        episode_reward = 0
        step = 0
        
        done = False
        while not done:
            
            # 预测Q值
            with torch.no_grad():
                q_maps = self.q_network(states)
                
                # 计算Whittle Index, TODO: 搞清楚计算whittle index时用的perception map是current还是previous
                current_perception = states['local_maps'][:, -1, :, :]
                prev_perception = states['fused_maps'][:, -1, :, :]
                
                whittle_indices = self.q_network.compute_whittle_index(
                    q_maps,
                    current_perception,
                    prev_perception
                )
            
            # ε-greedy选择动作
            if np.random.rand() < self.epsilon:
                # 探索：随机选择
                actions = torch.rand_like(whittle_indices)
                actions = (actions > (1 - self.env_config.TOP_K / (self.model_config.MAP_HEIGHT * self.model_config.MAP_WIDTH))).float()
            else:
                # 利用：选择top-K
                actions = self.q_network.select_top_k_regions(
                    whittle_indices,
                    k=self.env_config.TOP_K
                )
            
            # 执行动作
            next_obs, reward, terminated, truncated, _ = self.env.step(actions.cpu())
            next_state = self.env.wrapper_state(next_obs)
            # 移动到设备
            next_state = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                         for k, v in next_state.items()}
            done = terminated
            
            # 存储经验
            self.replay_buffer.push(
                states,
                q_maps,
                actions,
                reward,
                next_state,
                done
            )
            
            episode_reward += reward
            step += 1
            
            if done:
                break
            
            states = next_state
            #obs = next_obs
        
        # 衰减epsilon
        self.epsilon = max(
            self.model_config.EPSILON_END,
            self.epsilon * self.model_config.EPSILON_DECAY
        )
        
        self.episode += 1
        
        return episode_reward, step # + 1
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*60)
        print("Training ST-GAT Q-Network")
        print("="*60)
        
        for epoch in range(self.model_config.train.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.model_config.train.num_epochs}")
            print("-" * 60)
            
            # 收集经验
            episode_rewards = []
            episode_lengths = []
            
            print("Collecting experiences...")
            for _ in tqdm(range(2), desc="Episodes"):  # 每个epoch收集10个episode
                reward, length = self.collect_episode()
                episode_rewards.append(reward)
                episode_lengths.append(length)
            
            # 训练网络
            print("Training network...")
            train_losses = {
                'bellman_loss': [],
                'spatial_loss': [],
                'total_loss': []
            }
            
            for _ in tqdm(range(20), desc="Training steps"):  # 每个epoch训练50步
                losses = self.train_step()
                if losses:
                    for key, value in losses.items():
                        train_losses[key].append(value)
            
            # 更新目标网络
            if (epoch + 1) % self.model_config.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                print("✓ Target network updated")
            
            # 记录
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            
            self.writer.add_scalar('episode/reward', avg_reward, epoch)
            self.writer.add_scalar('episode/length', avg_length, epoch)
            self.writer.add_scalar('episode/epsilon', self.epsilon, epoch)
            
            if train_losses['total_loss']:
                for key, values in train_losses.items():
                    self.writer.add_scalar(f'train/{key}', np.mean(values), epoch)
            
            # 打印统计
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Epsilon: {self.epsilon:.3f}")
            if train_losses['total_loss']:
                print(f"  Total Loss: {np.mean(train_losses['total_loss']):.4f}")
            
            # 保存检查点
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 保存最终模型
        self.save_checkpoint('final_model.pth')
        print("\n✓ Training complete!")
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        os.makedirs(self.model_config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.model_config.checkpoint_dir, filename)
        
        torch.save({
            'epoch': self.episode,
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        
        print(f"✓ Saved checkpoint: {path}")