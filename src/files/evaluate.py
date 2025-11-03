"""
评估脚本：评估训练好的模型
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import Config
from model import STGATQNetwork
from environment import VehicularEnvironment


class Evaluator:
    """模型评估器"""
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config.DEVICE
        
        # 加载模型
        self.model = STGATQNetwork(config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Loaded model from {checkpoint_path}")
        
        # 环境
        self.env = VehicularEnvironment(config)
    
    def evaluate(self, num_episodes=50):
        """评估模型性能"""
        print("\n" + "="*60)
        print("Evaluating Model")
        print("="*60)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluation"):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config.MAX_EPISODE_LENGTH):
                # 移动到设备
                state_gpu = {k: v.to(self.device) for k, v in state.items()}
                
                # 预测Q值
                with torch.no_grad():
                    q_maps = self.model(state_gpu)
                    
                    # 计算Whittle Index
                    current_perception = state['perception_maps_history'][:, -1, :, :].to(self.device)
                    prev_perception = state['perception_maps_history'][:, -2, :, :].to(self.device)
                    
                    whittle_indices = self.model.compute_whittle_index(
                        q_maps,
                        current_perception,
                        prev_perception
                    )
                    
                    # 贪心选择top-K
                    actions = self.model.select_top_k_regions(
                        whittle_indices,
                        k=self.config.TOP_K
                    )
                
                # 执行动作
                next_state, reward, done, info = self.env.step(actions.cpu())
                
                episode_reward += reward
                
                if done:
                    break
                
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
        
        # 统计
        results = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
        
        # 打印结果
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Max Reward: {results['max_reward']:.2f}")
        print(f"Min Reward: {results['min_reward']:.2f}")
        print(f"Average Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}")
        print("="*60 + "\n")
        
        return results, episode_rewards
    
    def visualize_q_maps(self, save_dir='./results/visualizations'):
        """可视化Q值map"""
        os.makedirs(save_dir, exist_ok=True)
        
        state = self.env.reset()
        state_gpu = {k: v.to(self.device) for k, v in state.items()}
        
        with torch.no_grad():
            q_maps = self.model(state_gpu).cpu()
            
            current_perception = state['perception_maps_history'][:, -1, :, :]
            prev_perception = state['perception_maps_history'][:, -2, :, :]
            
            whittle_indices = self.model.compute_whittle_index(
                q_maps,
                current_perception,
                prev_perception
            ).cpu()
            
            actions = self.model.select_top_k_regions(
                whittle_indices,
                k=self.config.TOP_K
            ).cpu()
        
        # 可视化第一辆车
        vehicle_id = 0
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Q值map
        im1 = axes[0, 0].imshow(q_maps[vehicle_id].numpy(), cmap='YlOrRd')
        axes[0, 0].set_title(f'Q Value Map - Vehicle {vehicle_id}', fontsize=14)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Whittle Index
        im2 = axes[0, 1].imshow(whittle_indices[vehicle_id].numpy(), cmap='RdYlGn', center=0)
        axes[0, 1].set_title(f'Whittle Index - Vehicle {vehicle_id}', fontsize=14)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 感知质量map
        im3 = axes[1, 0].imshow(current_perception[vehicle_id].numpy(), cmap='Blues')
        axes[1, 0].set_title(f'Perception Quality Map - Vehicle {vehicle_id}', fontsize=14)
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 选择的区域
        im4 = axes[1, 1].imshow(actions[vehicle_id].numpy(), cmap='Greys', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Selected Regions (Top-{self.config.TOP_K}) - Vehicle {vehicle_id}', fontsize=14)
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'q_maps_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {save_path}")
    
    def compare_with_baselines(self, num_episodes=50):
        """与baseline方法比较"""
        print("\n" + "="*60)
        print("Baseline Comparison")
        print("="*60)
        
        results = {}
        
        # 1. 我们的方法
        print("\n1. Our Method (ST-GAT + Whittle Index)")
        our_results, _ = self.evaluate(num_episodes)
        results['Ours'] = our_results['avg_reward']
        
        # 2. Random baseline
        print("\n2. Random Selection")
        random_rewards = []
        for _ in tqdm(range(num_episodes), desc="Random"):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.MAX_EPISODE_LENGTH):
                # 随机选择
                actions = torch.rand(
                    self.config.NUM_VEHICLES,
                    self.config.MAP_HEIGHT,
                    self.config.MAP_WIDTH
                )
                threshold = 1 - self.config.TOP_K / (self.config.MAP_HEIGHT * self.config.MAP_WIDTH)
                actions = (actions > threshold).float()
                
                state, reward, done, _ = self.env.step(actions)
                episode_reward += reward
                
                if done:
                    break
            
            random_rewards.append(episode_reward)
        
        results['Random'] = np.mean(random_rewards)
        
        # 3. Greedy baseline (最高感知质量)
        print("\n3. Greedy (Highest Perception Quality)")
        greedy_rewards = []
        for _ in tqdm(range(num_episodes), desc="Greedy"):
            state = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.MAX_EPISODE_LENGTH):
                # 选择感知质量最高的区域
                perception = state['perception_maps_history'][:, -1, :, :]
                
                # 展平并找top-K
                N_v, H, W = perception.shape
                perception_flat = perception.view(N_v, -1)
                _, top_k_indices = torch.topk(perception_flat, self.config.TOP_K, dim=1)
                
                actions = torch.zeros_like(perception_flat)
                actions.scatter_(1, top_k_indices, 1)
                actions = actions.view(N_v, H, W)
                
                state, reward, done, _ = self.env.step(actions)
                episode_reward += reward
                
                if done:
                    break
            
            greedy_rewards.append(episode_reward)
        
        results['Greedy'] = np.mean(greedy_rewards)
        
        # 打印比较结果
        print("\n" + "="*60)
        print("Comparison Results")
        print("="*60)
        for method, reward in results.items():
            improvement = (reward - results['Random']) / abs(results['Random']) * 100
            print(f"{method:20s}: {reward:8.2f}  ({improvement:+.1f}% vs Random)")
        print("="*60 + "\n")
        
        # 可视化比较
        self._plot_comparison(results)
        
        return results
    
    def _plot_comparison(self, results):
        """绘制baseline比较图"""
        os.makedirs('./results', exist_ok=True)
        
        methods = list(results.keys())
        rewards = list(results.values())
        
        colors = ['red' if m == 'Ours' else 'gray' for m in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, rewards, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=12)
        
        plt.ylabel('Average Reward', fontsize=14)
        plt.title('Method Comparison', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        save_path = './results/baseline_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plot saved to {save_path}")


def main():
    config = Config()
    
    # 检查点路径
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python train.py")
        return
    
    # 创建评估器
    evaluator = Evaluator(config, checkpoint_path)
    
    # 评估
    evaluator.evaluate(num_episodes=50)
    
    # 可视化
    evaluator.visualize_q_maps()
    
    # 与baseline比较
    evaluator.compare_with_baselines(num_episodes=30)


if __name__ == '__main__':
    main()
