"""
示例脚本：展示如何使用GNN Q-Network框架
"""
import torch
from config import Config
from models import QNetworkGNN
from graph_builder import CooperativePerceptionGraph
from data_generator import VehicularEnvironment
from visualization import Visualizer
import numpy as np


def example_basic_usage():
    """示例1：基本使用流程"""
    print("="*60)
    print("Example 1: Basic Usage")
    print("="*60 + "\n")
    
    # 1. 创建配置
    config = Config()
    print("✓ Configuration loaded")
    
    # 2. 创建模型
    model = QNetworkGNN(config).to(config.DEVICE)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 3. 创建环境
    env = VehicularEnvironment(config)
    graph_builder = CooperativePerceptionGraph(config)
    print("✓ Environment initialized")
    
    # 4. 重置环境获取初始状态
    state = env.reset()
    print(f"✓ Initial state: {state['vehicle_data']['positions'].shape[0]} vehicles, "
          f"{state['region_data']['positions'].shape[0]} regions")
    
    # 5. 构建图
    graph = graph_builder.build_graph(
        state['vehicle_data'],
        state['region_data'],
        state['semantic_map']
    )
    print(f"✓ Graph built: {graph.num_vehicles} vehicles, {graph.num_regions} regions")
    print(f"  Edge types: {list(graph.edge_index_dict.keys())}")
    
    # 6. 前向传播
    model.eval()
    with torch.no_grad():
        graph = graph.to(config.DEVICE)
        q_values = model(graph)
        print(f"✓ Q values computed: shape={q_values.shape}, mean={q_values.mean():.4f}")
        
        # 7. 计算Whittle Index
        whittle_indices = model.compute_whittle_index(
            q_values,
            state['region_data']['confidence_maps'].to(config.DEVICE),
            torch.zeros_like(state['region_data']['confidence_maps']).to(config.DEVICE)
        )
        print(f"✓ Whittle Index computed: shape={whittle_indices.shape}")
        
        # 8. 选择top-K区域
        K = config.NUM_REGIONS // 4
        _, top_k_indices = torch.topk(whittle_indices, K)
        print(f"✓ Selected top-{K} regions: {top_k_indices.cpu().numpy()[:10]}...")
    
    print("\n" + "="*60 + "\n")


def example_single_episode():
    """示例2：运行一个完整的episode"""
    print("="*60)
    print("Example 2: Run a Complete Episode")
    print("="*60 + "\n")
    
    config = Config()
    model = QNetworkGNN(config).to(config.DEVICE)
    env = VehicularEnvironment(config)
    graph_builder = CooperativePerceptionGraph(config)
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    
    print("Running episode...")
    
    for step in range(config.MAX_EPISODE_LENGTH):
        # 构建图
        graph = graph_builder.build_graph(
            state['vehicle_data'],
            state['region_data'],
            state['semantic_map']
        )
        
        # 预测并选择动作
        with torch.no_grad():
            graph = graph.to(config.DEVICE)
            q_values = model(graph)
            whittle_indices = model.compute_whittle_index(
                q_values,
                state['region_data']['confidence_maps'].to(config.DEVICE),
                torch.zeros_like(state['region_data']['confidence_maps']).to(config.DEVICE)
            )
            
            K = config.NUM_REGIONS // 4
            _, top_k_indices = torch.topk(whittle_indices, K)
            action = torch.zeros(config.NUM_REGIONS)
            action[top_k_indices.cpu()] = 1
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        episode_reward += reward.item()
        episode_length += 1
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: reward={reward.item():.2f}, "
                  f"avg_confidence={info['avg_confidence']:.3f}")
        
        if done:
            break
        
        state = next_state
    
    print(f"\n✓ Episode finished:")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Episode length: {episode_length}")
    print(f"  Average reward per step: {episode_reward/episode_length:.4f}")
    
    print("\n" + "="*60 + "\n")


def example_visualization():
    """示例3：可视化结果"""
    print("="*60)
    print("Example 3: Visualization")
    print("="*60 + "\n")
    
    config = Config()
    model = QNetworkGNN(config).to(config.DEVICE)
    env = VehicularEnvironment(config)
    graph_builder = CooperativePerceptionGraph(config)
    visualizer = Visualizer(config)
    
    # 获取状态
    state = env.reset()
    graph = graph_builder.build_graph(
        state['vehicle_data'],
        state['region_data'],
        state['semantic_map']
    )
    
    # 预测
    model.eval()
    with torch.no_grad():
        graph = graph.to(config.DEVICE)
        q_values = model(graph)
        whittle_indices = model.compute_whittle_index(
            q_values,
            state['region_data']['confidence_maps'].to(config.DEVICE),
            torch.zeros_like(state['region_data']['confidence_maps']).to(config.DEVICE)
        )
        
        # 选择区域
        K = config.NUM_REGIONS // 4
        _, top_k_indices = torch.topk(whittle_indices, K)
        selected = torch.zeros(config.NUM_REGIONS)
        selected[top_k_indices.cpu()] = 1
    
    # 可视化
    print("Generating visualizations...")
    
    # Q值热力图
    visualizer.visualize_q_values(q_values.cpu(), 'example_q_values.png')
    print("✓ Q values heatmap saved")
    
    # Whittle Index
    visualizer.visualize_whittle_index(
        whittle_indices.cpu(),
        selected,
        'example_whittle_index.png'
    )
    print("✓ Whittle Index visualization saved")
    
    # 车辆-区域图
    visualizer.visualize_vehicle_region_graph(
        graph.cpu(),
        q_values.cpu(),
        'example_graph.png'
    )
    print("✓ Vehicle-region graph saved")
    
    print(f"\nAll visualizations saved to: {visualizer.save_dir}")
    print("\n" + "="*60 + "\n")


def example_model_comparison():
    """示例4：比较不同模型架构"""
    print("="*60)
    print("Example 4: Model Architecture Comparison")
    print("="*60 + "\n")
    
    config = Config()
    
    from models import create_model
    
    model_types = ['standard', 'dueling', 'ensemble']
    
    for model_type in model_types:
        model = create_model(config, model_type).to(config.DEVICE)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"{model_type.capitalize()} Model:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Memory (approx): {num_params * 4 / 1024 / 1024:.2f} MB")
        
        # 测试前向传播速度
        env = VehicularEnvironment(config)
        graph_builder = CooperativePerceptionGraph(config)
        state = env.reset()
        graph = graph_builder.build_graph(
            state['vehicle_data'],
            state['region_data'],
            state['semantic_map']
        ).to(config.DEVICE)
        
        import time
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(graph)
        
        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(graph)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start_time
        print(f"  Inference time: {elapsed/100*1000:.2f} ms per sample")
        print()
    
    print("="*60 + "\n")


def example_hyperparameter_tuning():
    """示例5：超参数影响分析"""
    print("="*60)
    print("Example 5: Hyperparameter Impact Analysis")
    print("="*60 + "\n")
    
    config = Config()
    
    # 测试不同的隐藏层维度
    hidden_dims = [64, 128, 256, 512]
    
    print("Testing different hidden dimensions:")
    for dim in hidden_dims:
        config.GNN_HIDDEN_DIM = dim
        model = QNetworkGNN(config).to(config.DEVICE)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  Hidden dim {dim}: {num_params:,} parameters")
    
    print()
    
    # 测试不同的GNN层数
    config.GNN_HIDDEN_DIM = 256  # 重置
    num_layers_list = [1, 2, 3, 4, 5]
    
    print("Testing different number of GNN layers:")
    for num_layers in num_layers_list:
        config.GNN_NUM_LAYERS = num_layers
        model = QNetworkGNN(config).to(config.DEVICE)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  {num_layers} layers: {num_params:,} parameters")
    
    print("\n" + "="*60 + "\n")


def main():
    """运行所有示例"""
    print("\n" + "🚗 " * 20)
    print("GNN-based Q-Network for Cooperative Vehicular Perception")
    print("Example Usage Demonstrations")
    print("🚗 " * 20 + "\n")
    
    try:
        # 示例1：基本使用
        example_basic_usage()
        
        # 示例2：完整episode
        example_single_episode()
        
        # 示例3：可视化
        example_visualization()
        
        # 示例4：模型比较
        example_model_comparison()
        
        # 示例5：超参数分析
        example_hyperparameter_tuning()
        
        print("\n" + "="*60)
        print("All examples completed successfully! ✓")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
