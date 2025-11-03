"""
主训练脚本
"""
import torch
import os
import argparse
from config import Config
from trainer import QNetworkTrainer
from data_generator import VehicularEnvironment, create_dataloaders
from graph_builder import CooperativePerceptionGraph
import numpy as np
from tqdm import tqdm


def train(args):
    """主训练函数"""
    
    # ============ 设置 ============
    config = Config()
    config.print_config()
    
    # 创建保存目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    
    # ============ 创建训练器 ============
    trainer = QNetworkTrainer(
        config,
        model_type=args.model_type,
        use_double_dqn=args.use_double_dqn
    )
    
    # 如果有检查点，加载
    if args.resume and os.path.exists(args.checkpoint_path):
        trainer.load_checkpoint(args.checkpoint_path)
        print(f"Resumed from {args.checkpoint_path}")
    
    # ============ 创建环境 ============
    env = VehicularEnvironment(config)
    graph_builder = CooperativePerceptionGraph(config)
    
    # ============ 训练循环 ============
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)
        
        # ============ 收集经验 ============
        print("Collecting experiences...")
        num_episodes = 50  # 每个epoch收集50个episode
        
        for episode in tqdm(range(num_episodes), desc="Episodes"):
            state = env.reset()
            episode_reward = 0
            
            for step in range(config.MAX_EPISODE_LENGTH):
                # 构建当前状态图
                state_graph = graph_builder.build_graph(
                    state['vehicle_data'],
                    state['region_data'],
                    state['semantic_map']
                )
                
                # ε-greedy策略选择动作
                epsilon = max(0.01, 1.0 - epoch / args.num_epochs)
                
                with torch.no_grad():
                    state_graph = state_graph.to(config.DEVICE)
                    q_values = trainer.q_network(state_graph)
                    whittle_indices = trainer.q_network.compute_whittle_index(
                        q_values,
                        state['region_data']['confidence_maps'].to(config.DEVICE),
                        torch.zeros_like(state['region_data']['confidence_maps']).to(config.DEVICE)
                    )
                
                # 选择top-K区域
                K = config.NUM_REGIONS // 4  # 共享25%的区域
                
                if np.random.rand() < epsilon:
                    # 探索：随机选择
                    action = torch.zeros(config.NUM_REGIONS)
                    selected_indices = np.random.choice(
                        config.NUM_REGIONS,
                        K,
                        replace=False
                    )
                    action[selected_indices] = 1
                else:
                    # 利用：选择Whittle Index最高的K个
                    _, top_k_indices = torch.topk(whittle_indices, K)
                    action = torch.zeros(config.NUM_REGIONS)
                    action[top_k_indices.cpu()] = 1
                
                # 执行动作
                next_state, reward, done, info = env.step(action)
                episode_reward += reward.item()
                
                # 构建下一状态图
                next_state_graph = graph_builder.build_graph(
                    next_state['vehicle_data'],
                    next_state['region_data'],
                    next_state['semantic_map']
                )
                
                # 存储经验
                experience_info = {
                    'current_interest_count': state['region_data']['interest_counts'],
                    'next_interest_count': next_state['region_data']['interest_counts'],
                    'transition_prob': torch.ones(config.NUM_REGIONS) * 0.8,
                    'confidence_change': (
                        next_state['region_data']['confidence_maps'] -
                        state['region_data']['confidence_maps']
                    )
                }
                
                trainer.replay_buffer.push(
                    state_graph.cpu(),
                    action,
                    reward,
                    next_state_graph.cpu(),
                    done,
                    experience_info
                )
                
                if done:
                    break
                
                state = next_state
            
            # 记录episode奖励
            trainer.writer.add_scalar('episode/reward', episode_reward, trainer.episode)
            trainer.writer.add_scalar('episode/length', step + 1, trainer.episode)
        
        # ============ 训练网络 ============
        print("\nTraining network...")
        epoch_losses = trainer.train_epoch(
            num_episodes=100,  # 训练100次
            batch_size=config.BATCH_SIZE,
            update_target_every=config.UPDATE_TARGET_EVERY
        )
        
        # 打印损失
        print(f"\nEpoch {epoch + 1} Summary:")
        for key, value in epoch_losses.items():
            print(f"  {key}: {value:.4f}")
        
        # ============ 保存检查点 ============
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nSaved checkpoint to {checkpoint_path}")
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # 保存最终模型
    final_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    trainer.save_checkpoint(final_path)
    print(f"\nFinal model saved to {final_path}")


def evaluate(args):
    """评估模型"""
    config = Config()
    
    # 创建训练器（只用于加载模型）
    trainer = QNetworkTrainer(
        config,
        model_type=args.model_type,
        use_double_dqn=False  # 评估时不需要
    )
    
    # 加载模型
    trainer.load_checkpoint(args.checkpoint_path)
    trainer.q_network.eval()
    
    # 创建环境
    env = VehicularEnvironment(config)
    graph_builder = CooperativePerceptionGraph(config)
    
    # 运行评估episodes
    print("\n" + "="*50)
    print("Evaluating Model")
    print("="*50 + "\n")
    
    total_rewards = []
    total_lengths = []
    
    for episode in tqdm(range(args.num_eval_episodes), desc="Evaluation"):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.MAX_EPISODE_LENGTH):
            # 构建图
            state_graph = graph_builder.build_graph(
                state['vehicle_data'],
                state['region_data'],
                state['semantic_map']
            )
            
            # 预测Q值和Whittle Index
            with torch.no_grad():
                state_graph = state_graph.to(config.DEVICE)
                q_values = trainer.q_network(state_graph)
                whittle_indices = trainer.q_network.compute_whittle_index(
                    q_values,
                    state['region_data']['confidence_maps'].to(config.DEVICE),
                    torch.zeros_like(state['region_data']['confidence_maps']).to(config.DEVICE)
                )
            
            # 选择top-K（贪心）
            K = config.NUM_REGIONS // 4
            _, top_k_indices = torch.topk(whittle_indices, K)
            action = torch.zeros(config.NUM_REGIONS)
            action[top_k_indices.cpu()] = 1
            
            # 执行
            next_state, reward, done, info = env.step(action)
            episode_reward += reward.item()
            
            if done:
                break
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_lengths.append(step + 1)
    
    # 打印统计
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Length: {np.mean(total_lengths):.2f} ± {np.std(total_lengths):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description='GNN-based Q-Network for Cooperative Perception')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval'],
                       help='运行模式')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'dueling', 'ensemble'],
                       help='模型类型')
    parser.add_argument('--use_double_dqn', action='store_true',
                       help='是否使用Double DQN')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--save_every', type=int, default=10,
                       help='每隔多少epoch保存一次')
    
    # 评估参数
    parser.add_argument('--num_eval_episodes', type=int, default=50,
                       help='评估的episode数量')
    
    # 检查点
    parser.add_argument('--resume', action='store_true',
                       help='是否从检查点恢复')
    parser.add_argument('--checkpoint_path', type=str,
                       default='./checkpoints/checkpoint_latest.pth',
                       help='检查点路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)


if __name__ == '__main__':
    main()
