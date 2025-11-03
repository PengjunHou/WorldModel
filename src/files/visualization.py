"""
可视化工具：用于分析和展示结果
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import networkx as nx
from matplotlib.patches import Rectangle
import os


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, config, save_dir: str = './results/visualizations'):
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def visualize_q_values(
        self,
        q_values: torch.Tensor,
        save_name: str = 'q_values_heatmap.png'
    ):
        """
        可视化Q值热力图
        
        Args:
            q_values: [N_r] Q值
            save_name: 保存文件名
        """
        grid_size = self.config.GRID_SIZE
        q_grid = q_values.reshape(grid_size, grid_size).cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            q_grid,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Q Value'}
        )
        plt.title('Q Values Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Q values heatmap saved to {save_path}")
    
    def visualize_whittle_index(
        self,
        whittle_indices: torch.Tensor,
        selected_regions: Optional[torch.Tensor] = None,
        save_name: str = 'whittle_index.png'
    ):
        """
        可视化Whittle Index和选择的区域
        
        Args:
            whittle_indices: [N_r] Whittle Index值
            selected_regions: [N_r] 二值向量，表示哪些区域被选择
            save_name: 保存文件名
        """
        grid_size = self.config.GRID_SIZE
        wi_grid = whittle_indices.reshape(grid_size, grid_size).cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：Whittle Index热力图
        sns.heatmap(
            wi_grid,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Whittle Index'},
            ax=axes[0]
        )
        axes[0].set_title('Whittle Index Heatmap', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Grid X')
        axes[0].set_ylabel('Grid Y')
        
        # 右图：选择的区域
        if selected_regions is not None:
            selected_grid = selected_regions.reshape(grid_size, grid_size).cpu().numpy()
            sns.heatmap(
                selected_grid,
                annot=True,
                fmt='.0f',
                cmap='Blues',
                cbar_kws={'label': 'Selected (1) / Not Selected (0)'},
                ax=axes[1]
            )
            axes[1].set_title('Selected Regions', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Grid X')
            axes[1].set_ylabel('Grid Y')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Whittle Index visualization saved to {save_path}")
    
    def visualize_vehicle_region_graph(
        self,
        data,
        q_values: torch.Tensor,
        save_name: str = 'vehicle_region_graph.png'
    ):
        """
        可视化车辆-区域图结构
        
        Args:
            data: HeteroData图对象
            q_values: [N_r] Q值
            save_name: 保存文件名
        """
        plt.figure(figsize=(14, 10))
        
        # 获取位置
        vehicle_pos = data['vehicle'].pos.cpu().numpy()
        region_pos = data['region'].pos.cpu().numpy()
        
        # 绘制区域（作为网格背景）
        grid_size = self.config.GRID_SIZE
        cell_size = self.config.PERCEPTION_RANGE / grid_size
        
        # Q值归一化用于颜色映射
        q_normalized = (q_values.cpu().numpy() - q_values.min().item()) / (
            q_values.max().item() - q_values.min().item() + 1e-6
        )
        
        for idx, pos in enumerate(region_pos):
            grid_x = int(pos[0] / cell_size)
            grid_y = int(pos[1] / cell_size)
            
            rect = Rectangle(
                (grid_x * cell_size, grid_y * cell_size),
                cell_size,
                cell_size,
                facecolor=plt.cm.YlOrRd(q_normalized[idx]),
                edgecolor='gray',
                alpha=0.6
            )
            plt.gca().add_patch(rect)
        
        # 绘制车辆-区域感知边
        if ('vehicle', 'perceives', 'region') in data.edge_index_dict:
            edge_index = data[('vehicle', 'perceives', 'region')].edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                v_idx, r_idx = edge_index[:, i]
                v_pos = vehicle_pos[v_idx]
                r_pos = region_pos[r_idx]
                plt.plot(
                    [v_pos[0], r_pos[0]],
                    [v_pos[1], r_pos[1]],
                    'b-',
                    alpha=0.1,
                    linewidth=0.5
                )
        
        # 绘制车辆
        plt.scatter(
            vehicle_pos[:, 0],
            vehicle_pos[:, 1],
            c='blue',
            s=200,
            marker='o',
            edgecolors='black',
            linewidths=2,
            label='Vehicles',
            zorder=3
        )
        
        # 绘制区域中心
        plt.scatter(
            region_pos[:, 0],
            region_pos[:, 1],
            c='red',
            s=50,
            marker='s',
            alpha=0.5,
            label='Region Centers',
            zorder=2
        )
        
        plt.xlim(0, self.config.PERCEPTION_RANGE)
        plt.ylim(0, self.config.PERCEPTION_RANGE)
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.title('Vehicle-Region Cooperative Perception Graph', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.YlOrRd,
            norm=plt.Normalize(vmin=q_values.min().item(), vmax=q_values.max().item())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Q Value')
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Vehicle-region graph saved to {save_path}")
    
    def plot_training_curves(
        self,
        log_dir: str,
        save_name: str = 'training_curves.png'
    ):
        """
        从TensorBoard日志绘制训练曲线
        
        Args:
            log_dir: TensorBoard日志目录
            save_name: 保存文件名
        """
        from tensorboard.backend.event_processing import event_accumulator
        
        # 读取TensorBoard日志
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        # 获取可用的标量
        tags = ea.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 绘制关键指标
        metrics_to_plot = [
            'train/total_loss',
            'train/bellman_loss',
            'episode/reward',
            'episode/length'
        ]
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in tags:
                events = ea.Scalars(metric)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                axes[idx].plot(steps, values, linewidth=2)
                axes[idx].set_xlabel('Step', fontsize=12)
                axes[idx].set_ylabel(metric.split('/')[-1].replace('_', ' ').title(), fontsize=12)
                axes[idx].set_title(metric, fontsize=14, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
    
    def visualize_attention_weights(
        self,
        data,
        layer_idx: int = 0,
        save_name: str = 'attention_weights.png'
    ):
        """
        可视化注意力权重（如果模型使用GAT）
        
        Args:
            data: 带有注意力权重的图数据
            layer_idx: 要可视化的层索引
            save_name: 保存文件名
        """
        # 这里需要修改模型以返回注意力权重
        # 暂时跳过实现，可以后续添加
        pass
    
    def create_comparison_plot(
        self,
        baseline_results: Dict[str, List[float]],
        our_results: Dict[str, List[float]],
        save_name: str = 'comparison.png'
    ):
        """
        创建与baseline方法的比较图
        
        Args:
            baseline_results: baseline方法的结果
            our_results: 我们方法的结果
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        methods = list(baseline_results.keys())
        
        # 奖励比较
        baseline_rewards = [np.mean(baseline_results[m]) for m in methods]
        our_reward = np.mean(our_results['reward'])
        
        x_pos = np.arange(len(methods) + 1)
        rewards = baseline_rewards + [our_reward]
        colors = ['gray'] * len(methods) + ['red']
        labels = methods + ['Ours (GNN-Whittle)']
        
        axes[0].bar(x_pos, rewards, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0].set_ylabel('Average Reward', fontsize=12)
        axes[0].set_title('Reward Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, axis='y', alpha=0.3)
        
        # 通信开销比较
        baseline_comm = [np.mean(baseline_results[m]) for m in methods]
        our_comm = np.mean(our_results['communication_cost'])
        
        comm_costs = baseline_comm + [our_comm]
        
        axes[1].bar(x_pos, comm_costs, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_ylabel('Communication Cost', fontsize=12)
        axes[1].set_title('Communication Cost Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {save_path}")


def analyze_model_performance(
    model,
    test_loader,
    config,
    visualizer: Visualizer
):
    """
    全面分析模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        config: 配置对象
        visualizer: 可视化器
    """
    model.eval()
    
    all_q_values = []
    all_whittle_indices = []
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch['state_graph'].to(config.DEVICE)
            
            # 预测Q值
            q_values = model(data)
            all_q_values.append(q_values.cpu())
            
            # 计算Whittle Index
            confidence = batch['state_graph']['region'].x[:, 0]  # 假设第一维是confidence
            whittle_idx = model.compute_whittle_index(
                q_values,
                confidence.to(config.DEVICE),
                torch.zeros_like(confidence).to(config.DEVICE)
            )
            all_whittle_indices.append(whittle_idx.cpu())
    
    # 统计分析
    all_q_values = torch.cat(all_q_values)
    all_whittle_indices = torch.cat(all_whittle_indices)
    
    print("\n" + "="*50)
    print("Model Performance Analysis")
    print("="*50)
    print(f"Q Value Statistics:")
    print(f"  Mean: {all_q_values.mean():.4f}")
    print(f"  Std: {all_q_values.std():.4f}")
    print(f"  Min: {all_q_values.min():.4f}")
    print(f"  Max: {all_q_values.max():.4f}")
    print(f"\nWhittle Index Statistics:")
    print(f"  Mean: {all_whittle_indices.mean():.4f}")
    print(f"  Std: {all_whittle_indices.std():.4f}")
    print(f"  Min: {all_whittle_indices.min():.4f}")
    print(f"  Max: {all_whittle_indices.max():.4f}")
    print("="*50 + "\n")
    
    # 绘制分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(all_q_values.numpy(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Q Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Q Value Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(all_whittle_indices.numpy(), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Whittle Index')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Whittle Index Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(visualizer.save_dir, 'distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distribution plots saved to {save_path}")
