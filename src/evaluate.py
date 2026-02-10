"""
评估脚本：评估训练好的模型
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

from env import make_env
from utils import PARSER, init_log
from src.config import Config
from src.model import STGATQNetwork


class Evaluator:
    """模型评估器"""
    def __init__(self, env_config, model_config : Config , checkpoint_path):
        self.model_config = model_config
        self.env_config = env_config
        self.device = model_config.DEVICE
        
        # 加载模型
        self.model = STGATQNetwork(env_config, model_config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Loaded model from {checkpoint_path}")
        
        # 环境
        self.env = make_env(args=env_config, dream_env=False, render_mode=False)
    
    def evaluate(self, num_episodes=50):
        """评估模型性能"""
        print("\n" + "="*60)
        print("Evaluating Model")
        print("="*60)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluation"):
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
                    q_maps = self.model(states)
                    
                    # 计算Whittle Index, TODO: 搞清楚计算whittle index时用的perception map是current还是previous
                    current_perception = states['local_maps'][:, -1, :, :]
                    prev_perception = states['fused_maps'][:, -1, :, :]
                    
                    whittle_indices = self.model.compute_whittle_index(
                        q_maps,
                        current_perception,
                        prev_perception
                    )
                    
                    # 贪心选择top-K
                    actions = self.model.select_top_k_regions(
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
                
                episode_reward += reward
                step += 1
                
                if done:
                    break
                
                states = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
        
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
        
        obs, info = self.env.reset()
        states = self.env.wrapper_state(obs)
        # 移动到设备
        states = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                    for k, v in states.items()}
        
        with torch.no_grad():
            q_maps = self.model(states)
            
            current_perception = states['local_maps'][:, -1, :, :]
            prev_perception = states['fused_maps'][:, -1, :, :]
            
            whittle_indices = self.model.compute_whittle_index(
                q_maps,
                current_perception,
                prev_perception
            ).cpu()
            
            actions = self.model.select_top_k_regions(
                whittle_indices,
                k=self.env_config.TOP_K
            ).cpu()
        
        # 可视化第一辆车
        vehicle_index = 0
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Q值map
        im1 = axes[0, 0].imshow(q_maps[vehicle_index].cpu().numpy(), cmap='YlOrRd')
        axes[0, 0].set_title(f'Q Value Map - Vehicle {vehicle_index+1}', fontsize=14)
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Whittle Index
        im2 = axes[0, 1].imshow(whittle_indices[vehicle_index].numpy(), cmap='RdYlGn')
        axes[0, 1].set_title(f'Whittle Index - Vehicle {vehicle_index+1}', fontsize=14)
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 感知质量map
        im3 = axes[1, 0].imshow(current_perception[vehicle_index].cpu().numpy(), cmap='Blues')
        axes[1, 0].set_title(f'Perception Quality Map - Vehicle {vehicle_index+1}', fontsize=14)
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 选择的区域
        im4 = axes[1, 1].imshow(actions[vehicle_index].numpy(), cmap='Greys', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Selected Regions (Top-{self.env_config.TOP_K}) - Vehicle {vehicle_index+1}', fontsize=14)
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'q_maps_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to {save_path}")


    def visualize_q_maps_vehicle(self, vehicle_id, save_dir='./results/qmaps_visualizations'):
        """可视化Q值map"""
        save_dir = os.path.join(save_dir, f'vehicle_{vehicle_id}')
        os.makedirs(save_dir, exist_ok=True)
        
        obs, info = self.env.reset()
        states = self.env.wrapper_state(obs)
        # 移动到设备
        states = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                    for k, v in states.items()}
        step = 0
        done = False

        while not done:
            
            # 预测Q值
            with torch.no_grad():
                q_maps = self.model(states)
                
                # 计算Whittle Index, TODO: 搞清楚计算whittle index时用的perception map是current还是previous
                current_perception = states['local_maps'][:, -1, :, :]
                prev_perception = states['fused_maps'][:, -1, :, :]
                
                whittle_indices = self.model.compute_whittle_index(
                    q_maps,
                    current_perception,
                    prev_perception
                )
                
                # 贪心选择top-K
                actions = self.model.select_top_k_regions(
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
            
            step += 1
            


            # 可视化
            vehicle_index = vehicle_id - 1
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Q值map
            # im1 = axes[0, 0].imshow(q_maps[vehicle_index].cpu().numpy(), cmap='YlOrRd')
            # axes[0, 0].set_title(f'Q Value Map - Vehicle {vehicle_index+1}', fontsize=14)
            # plt.colorbar(im1, ax=axes[0, 0])
            fused_q_map = next_state['fused_maps'][:, -1, :, :]
            im1 = axes[0, 0].imshow(fused_q_map[vehicle_index].cpu().numpy(), cmap='YlOrRd')
            axes[0, 0].set_title(f'Perception Quality Map - Vehicle {vehicle_index+1}', fontsize=14)
            plt.colorbar(im1, ax=axes[0, 0])
            
            # Whittle Index
            im2 = axes[0, 1].imshow(whittle_indices[vehicle_index].cpu().numpy(), cmap='RdYlGn')
            axes[0, 1].set_title(f'Whittle Index - Vehicle {vehicle_index+1}', fontsize=14)
            plt.colorbar(im2, ax=axes[0, 1])
            
            # 感知质量map
            im3 = axes[1, 0].imshow(current_perception[vehicle_index].cpu().numpy(), cmap='Blues')
            axes[1, 0].set_title(f'Perception Confidence Map - Vehicle {vehicle_index+1}', fontsize=14)
            plt.colorbar(im3, ax=axes[1, 0])
            
            # 选择的区域
            im4 = axes[1, 1].imshow(actions[vehicle_index].cpu().numpy(), cmap='Greys', vmin=0, vmax=1)
            axes[1, 1].set_title(f'Selected Regions (Top-{self.env_config.TOP_K}) - Vehicle {vehicle_index+1}', fontsize=14)
            plt.colorbar(im4, ax=axes[1, 1])
            
            plt.tight_layout()

            save_path = os.path.join(save_dir, str(step)+'_q_maps.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Visualization saved to {save_path}")

            if done:
                break
            
            states = next_state

    def visualize_q_maps_vehicle_compare(self, vehicle_id, save_dir='./results/qmaps_visualizations'):
        """可视化两个热力图（统一颜色、统一刻度）"""
        save_dir = os.path.join(save_dir, f"vehicle_{vehicle_id}")
        os.makedirs(save_dir, exist_ok=True)

        obs, info = self.env.reset()
        states = self.env.wrapper_state(obs)

        # Move to device
        states = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                    for k, v in states.items()}

        step = 0
        done = False

        while not done:
            # 预测Q值
            with torch.no_grad():
                q_maps = self.model(states)
                
                # 计算Whittle Index, TODO: 搞清楚计算whittle index时用的perception map是current还是previous
                current_perception = states['local_maps'][:, -1, :, :]
                prev_perception = states['fused_maps'][:, -1, :, :]
                
                whittle_indices = self.model.compute_whittle_index(
                    q_maps,
                    current_perception,
                    prev_perception
                )
                
                # 贪心选择top-K
                actions = self.model.select_top_k_regions(
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
            
            step += 1

            # ----------- Extract target vehicle -----------
            veh_idx = vehicle_id - 1

            confidence_map = current_perception[veh_idx]
            quality_map = obs["cur_fused_maps"][veh_idx]  # (H,W)

            confidence_map_np = confidence_map.cpu().numpy()[0:10, 1:15]  # 截取部分区域以便显示
            quality_map_np = quality_map[0:10, 1:15] 

            # ----------- Compute unified scaling (VERY IMPORTANT) -----------
            vmin = min(quality_map_np.min(), confidence_map_np.min())
            vmax = max(quality_map_np.max(), confidence_map_np.max())

            # ----------- Visualization -----------

            # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 1. Perception Confidence
            # im1 = axes[0].imshow(
            #     confidence_map_np,
            #     cmap="YlOrRd",
            #     vmin=vmin,
            #     vmax=vmax
            # )
            # axes[0].set_title(f"Perception Confidence Map - Vehicle {vehicle_id}", fontsize=16)
            # plt.colorbar(im1, ax=axes[0])

            # # 2. Perception Quality
            # im2 = axes[1].imshow(
            #     quality_map_np,
            #     cmap="YlOrRd",
            #     vmin=vmin,
            #     vmax=vmax
            # )
            # axes[1].set_title(f"Perception Quality Map - Vehicle {vehicle_id}", fontsize=16)
            # plt.colorbar(im2, ax=axes[1])


            # plt.tight_layout()

            # save_path = os.path.join(save_dir, f"{step}_perception_maps.png")
            # plt.savefig(save_path, dpi=300, bbox_inches="tight")
            # plt.close()

            # print(f"✓ Saved: {save_path}")

            # ----------- Visualization -----------

            # ==== 第一张图：Perception Confidence Map ====
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im_conf = ax1.imshow(
                confidence_map_np,
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax
            )
            ax1.set_xlabel("X axis", fontsize=18)
            ax1.set_ylabel("Y axis", fontsize=18)
            # ax1.set_title(f"Perception Confidence Map - Vehicle {vehicle_id}", fontsize=16)
            plt.colorbar(im_conf, ax=ax1, shrink=0.58)


            conf_path = os.path.join(save_dir, f"{step}_confidence_map.png")
            plt.savefig(conf_path, dpi=400, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {conf_path}")

            # ==== 第二张图：Perception Quality Map ====
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            im_qual = ax2.imshow(
                quality_map_np,
                cmap="YlOrRd",
                vmin=vmin,
                vmax=vmax
            )
            ax2.set_xlabel("X axis", fontsize=18)
            ax2.set_ylabel("Y axis", fontsize=18)
            # ax2.set_title(f"Perception Quality Map - Vehicle {vehicle_id}", fontsize=16)
            plt.colorbar(im_qual, ax=ax2, shrink=0.58)

            qual_path = os.path.join(save_dir, f"{step}_quality_map.png")
            plt.savefig(qual_path, dpi=400, bbox_inches="tight")
            plt.close()
            print(f"✓ Saved: {qual_path}")


            if done:
                break

            states = next_state
            obs = next_obs


    def compare_with_baselines(self, num_episodes=10):
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
            ons, info = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # 随机选择
                actions = torch.rand(
                    self.model_config.NUM_VEHICLES,
                    self.model_config.MAP_HEIGHT,
                    self.model_config.MAP_WIDTH
                )
                threshold = 1 - self.env_config.TOP_K / (self.model_config.MAP_HEIGHT * self.model_config.MAP_WIDTH)
                actions = (actions > threshold).float()
                
                next_obs, reward, terminated, truncated, _ = self.env.step(actions)
                episode_reward += reward
                done = terminated
                
                if done:
                    break
            
            random_rewards.append(episode_reward)
        
        results['Random'] = np.mean(random_rewards)
        
        # 3. Greedy baseline (最高感知质量)
        print("\n3. Greedy (Highest Perception Quality)")
        greedy_rewards = []
        for _ in tqdm(range(num_episodes), desc="Greedy"):
            obs, info = self.env.reset()
            states = self.env.wrapper_state(obs)
            # 移动到设备
            states = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                        for k, v in states.items()}
            episode_reward = 0
            done = False

            while not done:
                # 选择感知质量最高的区域
                perception = states['local_maps'][:, -1, :, :]
                
                # 展平并找top-K
                N_v, H, W = perception.shape
                perception_flat = perception.view(N_v, -1)
                _, top_k_indices = torch.topk(perception_flat, self.env_config.TOP_K, dim=1)
                
                actions = torch.zeros_like(perception_flat)
                actions.scatter_(1, top_k_indices, 1)
                actions = actions.view(N_v, H, W)
                
                next_obs, reward, terminated, truncated, _ = self.env.step(actions.cpu())
                next_state = self.env.wrapper_state(next_obs)
                # 移动到设备
                next_state = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) \
                            for k, v in next_state.items()}
                done = terminated
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

    def compare_different_topk(self, topk_list=[5, 10, 20, 30, 40, 50], num_episodes=10):
        """
        比较相同模型在不同 TOP-K 配置下的性能
        ----------------------------------------------------
        Args:
            topk_list (list): 不同的TOP-K设置列表
            num_episodes (int): 每个配置下的评估轮数
        ----------------------------------------------------
        输出：
            - 每个TOP-K下的平均回报、方差
            - 折线/柱状可视化结果
        """
        print("\n" + "="*60)
        print("TOP-K Sensitivity Evaluation")
        print("="*60)
        
        results = {}
        
        for k in topk_list:
            print(f"\n▶ Evaluating with TOP-K = {k}")
            self.env_config.TOP_K = k  # 动态修改配置
            avg_reward_list = []
            
            for episode in tqdm(range(num_episodes), desc=f"TOP-K={k}"):
                obs, info = self.env.reset()
                states = self.env.wrapper_state(obs)
                states = {k_: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                          for k_, v in states.items()}
                
                episode_reward = 0
                done = False
                
                while not done:
                    with torch.no_grad():
                        q_maps = self.model(states)
                        current_perception = states['local_maps'][:, -1, :, :]
                        prev_perception = states['fused_maps'][:, -1, :, :]
                        
                        whittle_indices = self.model.compute_whittle_index(
                            q_maps, current_perception, prev_perception
                        )
                        actions = self.model.select_top_k_regions(
                            whittle_indices, k=self.env_config.TOP_K
                        )
                    
                    next_obs, reward, terminated, truncated, _ = self.env.step(actions.cpu())
                    next_state = self.env.wrapper_state(next_obs)
                    next_state = {k_: torch.as_tensor(v, dtype=torch.float32, device=self.device)
                                  for k_, v in next_state.items()}
                    done = terminated
                    episode_reward += reward
                    states = next_state
                
                avg_reward_list.append(episode_reward)
            
            results[k] = {
                'mean': np.mean(avg_reward_list),
                'std': np.std(avg_reward_list)
            }
            print(f"TOP-K={k} → Avg Reward: {results[k]['mean']:.2f} ± {results[k]['std']:.2f}")
        
        # 绘图
        os.makedirs('./results', exist_ok=True)
        plt.figure(figsize=(8, 5))
        x = list(results.keys())
        y = [results[k]['mean'] for k in x]
        yerr = [results[k]['std'] for k in x]
        
        plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, linewidth=2)
        plt.xlabel('TOP-K', fontsize=14)
        plt.ylabel('Average Reward', fontsize=14)
        plt.title('Performance under Different TOP-K Configurations', fontsize=15, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = './results/topk_sensitivity.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"\n✓ TOP-K comparison plot saved to {save_path}")
        
        return results


    # [5, 10, 20, 30, 50, 75, 100]
    def compare_all_methods_under_topk(self, topk_list=[5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250], num_episodes=10):
        """
        比较在不同 TOP-K 设置下三种方法（Ours, Random, Greedy）的性能。
        ----------------------------------------------------
        Args:
            topk_list (list): 不同TOP-K配置
            num_episodes (int): 每个配置下评估轮数
        ----------------------------------------------------
        输出：
            - 每个TOP-K下三种方法的平均reward和标准差
            - 保存CSV文件 results/topk_method_comparison.csv
            - 绘制折线对比图 results/topk_method_comparison.png
        """
        print("\n" + "="*60)
        print("Comparing Baselines under Different TOP-K Settings")
        print("="*60)

        results = []  # 存储每个配置的结果

        for k in topk_list:
            print(f"\n▶ Evaluating TOP-K = {k}")
            self.env_config.TOP_K = k

            # ========================= 1. Ours =========================
            print("→ Ours (ST-GAT + Whittle Index)")
            our_rewards = []
            for _ in tqdm(range(num_episodes), desc=f"Ours@K={k}"):
                ep_res, _ = self.evaluate(1)  # 这里内部已运行一次完整episode
                our_rewards.append(ep_res['avg_reward'])
            our_mean, our_std = np.mean(our_rewards), np.std(our_rewards)

            # ========================= 2. Random =========================
            print("→ Random Selection")
            random_rewards = []
            for _ in tqdm(range(num_episodes), desc=f"Random@K={k}"):
                obs, info = self.env.reset()
                episode_reward, done = 0, False
                while not done:
                    actions = torch.rand(
                        self.model_config.NUM_VEHICLES,
                        self.model_config.MAP_HEIGHT,
                        self.model_config.MAP_WIDTH
                    )
                    threshold = 1 - self.env_config.TOP_K / (self.model_config.MAP_HEIGHT * self.model_config.MAP_WIDTH)
                    actions = (actions > threshold).float()
                    obs, reward, terminated, truncated, _ = self.env.step(actions)
                    episode_reward += reward
                    done = terminated
                random_rewards.append(episode_reward)
            random_mean, random_std = np.mean(random_rewards), np.std(random_rewards)

            # ========================= 3. Greedy =========================
            print("→ Greedy (Highest Perception Quality)")
            greedy_rewards = []
            for _ in tqdm(range(num_episodes), desc=f"Greedy@K={k}"):
                obs, info = self.env.reset()
                states = self.env.wrapper_state(obs)
                states = {kk: torch.as_tensor(v, dtype=torch.float32, device=self.device) for kk, v in states.items()}
                episode_reward, done = 0, False
                while not done:
                    perception = states['local_maps'][:, -1, :, :]
                    N_v, H, W = perception.shape
                    perception_flat = perception.view(N_v, -1)
                    _, top_k_indices = torch.topk(perception_flat, k, dim=1)
                    actions = torch.zeros_like(perception_flat)
                    actions.scatter_(1, top_k_indices, 1)
                    actions = actions.view(N_v, H, W)
                    next_obs, reward, terminated, truncated, _ = self.env.step(actions.cpu())
                    next_state = self.env.wrapper_state(next_obs)
                    next_state = {kk: torch.as_tensor(v, dtype=torch.float32, device=self.device) for kk, v in next_state.items()}
                    states = next_state
                    done = terminated
                    episode_reward += reward
                greedy_rewards.append(episode_reward)
            greedy_mean, greedy_std = np.mean(greedy_rewards), np.std(greedy_rewards)

            # ========================= 记录结果 =========================
            results.append({
                'TOP_K': k,
                'Ours_mean': our_mean,
                'Ours_std': our_std,
                'Random_mean': random_mean,
                'Random_std': random_std,
                'Greedy_mean': greedy_mean,
                'Greedy_std': greedy_std
            })

        # ========================= 保存为CSV =========================
        os.makedirs('./results', exist_ok=True)
        df = pd.DataFrame(results)
        csv_path = './results/topk_method_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")

        # ========================= 绘制折线图 =========================
        plt.figure(figsize=(8, 5))
        x = df['TOP_K']
        # plt.errorbar(x, df['Ours_mean'], yerr=df['Ours_std'], fmt='-o', label='Ours', capsize=4, linewidth=2)
        plt.errorbar(x, df['Greedy_mean'], yerr=df['Greedy_std'], fmt='-s', label='Greedy', capsize=4, linewidth=2)
        # plt.errorbar(x, df['Random_mean'], yerr=df['Random_std'], fmt='-^', label='Random', capsize=4, linewidth=2)

        plt.xlabel(r"Number of selected subregions $K_t$", fontsize=13)
        plt.ylabel(r"Total perception quality over time", fontsize=13)
        # plt.title("Baseline Comparison under Different TOP-K Settings", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_path = './results/topk_method_comparison.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✓ Comparison plot saved to {save_path}\n")

        return df


def plot_topk_comparison_from_csv(csv_path='./results/topk_method_comparison.csv',
                                  save_path='./results/topk_method_comparison_from_csv.png'):
    """
    读取 topk_method_comparison.csv 并绘制三种方法的 TOP-K 对比图
    -------------------------------------------------------
    Args:
        csv_path (str): 存储结果的CSV路径
        save_path (str): 保存绘图的路径
    -------------------------------------------------------
    输出:
        折线 + 误差带图，横轴为 TOP-K，纵轴为平均回报
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

    # 1️⃣ 读取 CSV 文件
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded results from {csv_path}")
    print(df.head())

    # 2️⃣ 提取数据
    x = df['TOP_K']
    ours_mean, ours_std = df['Ours_mean'], df['Ours_std']
    greedy_mean, greedy_std = df['Greedy_mean'], df['Greedy_std']
    random_mean, random_std = df['Random_mean'], df['Random_std']

    # 3️⃣ 绘制折线图
    plt.figure(figsize=(6, 4))

    plt.errorbar(x, ours_mean, yerr=ours_std, fmt='-o', capsize=4,
                 linewidth=2.5, markersize=6, label='Proposed')
    plt.errorbar(x, greedy_mean, yerr=greedy_std, fmt='-s', capsize=4,
                 linewidth=2.5, markersize=6, label='Greedy')
    plt.errorbar(x, random_mean, yerr=random_std, fmt='-^', capsize=4,
                 linewidth=2.5, markersize=6, label='Random')

    plt.xlabel("Number of selected subregions $K_t$", fontsize=18)
    plt.ylabel("Total perception quality", fontsize=18)
    # plt.title('Performance Comparison under Different TOP-K Settings',
    #           fontsize=15, fontweight='bold')

    plt.legend(
        fontsize=16,
        loc='center right', 
        bbox_to_anchor=(0.99, 0.50)   # ← 右侧偏内、垂直居于两条线中间
    )
    plt.grid(True)
    plt.tight_layout()

    # 4️⃣ 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.close()

    print(f"✓ Comparison plot saved to {save_path}")


def main():
    model_config = Config()
    env_config = PARSER.parse_args()
    model_config.print_config()
    
    # 检查点路径
    # checkpoint_epoch_260.pth
    checkpoint_path = os.path.join(model_config.CHECKPOINT_DIR, 'final_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first: python train.py")
        return
    
    # 创建评估器
    evaluator = Evaluator(env_config, model_config, checkpoint_path)
    
    # 评估
    evaluator.evaluate(num_episodes=1)
    
    # # 可视化
    # evaluator.visualize_q_maps()
    evaluator.visualize_q_maps_vehicle_compare(vehicle_id=1)
    
    # # 与baseline比较
    evaluator.compare_with_baselines(num_episodes=1)

    evaluator.compare_different_topk(num_episodes=1)

    # evaluator.compare_all_methods_under_topk(num_episodes=1)

    # plot_topk_comparison_from_csv()




if __name__ == '__main__':
    main()
