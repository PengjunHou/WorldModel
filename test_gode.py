"""
Integration Test Script for GODE Cooperative Perception
GODE协同感知系统集成测试
"""

import torch
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# # 确保能导入所有模块
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_gode import (
    get_default_config, 
    get_training_config, 
    get_eval_config,
    get_fast_config
)
from gode_model import GODEVehicleModel, ActionDecisionModule, build_vehicle_graph
from gode_wrapper import GODECoordinationManager
from train_gode import GODETrainer


class GODEIntegrationTest:
    """GODE系统集成测试"""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Test] Using device: {self.device}")
    
    def test_model_forward(self):
        """测试模型前向传播"""
        print("\n" + "="*50)
        print("Test 1: Model Forward Pass")
        print("="*50)
        
        # 创建模型
        model = GODEVehicleModel(
            map_size=tuple(self.config.conf_map.local_bev_config['grid_size']),
            state_dim=6,
            hidden_dim=self.config.gode.hidden_dim,
            num_gnn_layers=self.config.gode.num_gnn_layers,
            use_controlled_gde=self.config.gode.use_controlled_gde,
            ode_method=self.config.gode.ode_method
        ).to(self.device)
        
        # 创建虚拟输入
        num_vehicles = 5
        H, W = self.config.conf_map.local_bev_config['grid_size']
        
        local_maps = torch.randn(num_vehicles, H, W).to(self.device)
        vehicle_states = torch.randn(num_vehicles, 6).to(self.device)
        
        # 创建图
        positions = np.random.rand(num_vehicles, 2) * 100 - 50
        graph = build_vehicle_graph(positions, max_comm_range=50.0).to(self.device)
        
        # 前向传播
        print(f"Input shapes:")
        print(f"  local_maps: {local_maps.shape}")
        print(f"  vehicle_states: {vehicle_states.shape}")
        print(f"  graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        
        with torch.no_grad():
            pred_maps = model(
                local_maps=local_maps,
                vehicle_states=vehicle_states,
                graph=graph,
                prediction_horizon=1.0
            )
        
        print(f"Output shape: {pred_maps.shape}")
        print(f"NFE (Number of Function Evaluations): {model.get_nfe()}")
        print(f"Output range: [{pred_maps.min():.3f}, {pred_maps.max():.3f}]")
        print("✓ Model forward pass successful!")
        
        return True
    
    def test_action_module(self):
        """测试决策模块"""
        print("\n" + "="*50)
        print("Test 2: Action Decision Module")
        print("="*50)
        
        action_module = ActionDecisionModule(
            map_size=tuple(self.config.conf_map.local_bev_config['grid_size']),
            hidden_dim=64
        ).to(self.device)
        
        # 创建虚拟地图
        H, W = self.config.conf_map.local_bev_config['grid_size']
        pred_map = torch.rand(1, H, W).to(self.device)
        current_map = torch.rand(1, H, W).to(self.device)
        
        # 决策
        with torch.no_grad():
            action = action_module(pred_map, current_map)
        
        print(f"Predicted map shape: {pred_map.shape}")
        print(f"Current map shape: {current_map.shape}")
        print(f"Action: {action.item():.3f}")
        print(f"Action range: [-1, 1]")
        assert -1 <= action.item() <= 1, "Action out of range!"
        print("✓ Action module test successful!")
        
        return True
    
    def test_gode_manager(self):
        """测试GODE管理器"""
        print("\n" + "="*50)
        print("Test 3: GODE Coordination Manager")
        print("="*50)
        
        manager = GODECoordinationManager(self.config, self.device)
        
        # 注册车辆
        vehicle_ids = [1, 2, 3, 4, 5]
        for vid in vehicle_ids:
            manager.register_vehicle(vid)
        
        print(f"Registered {len(manager.agents)} vehicles")
        
        # 更新观测
        H, W = self.config.conf_map.local_bev_config['grid_size']
        for vid in vehicle_ids:
            local_map = np.random.rand(H, W).astype(np.float32)
            vehicle_state = np.array([
                np.random.rand() * 100,  # x
                np.random.rand() * 100,  # y
                np.random.randn() * 5,   # vx
                np.random.randn() * 5,   # vy
                np.random.rand() * 2 * np.pi,  # heading
                np.random.rand() * 30    # speed
            ])
            manager.agents[vid].update_local_observation(local_map, vehicle_state)
        
        print("Updated observations for all vehicles")
        
        # 测试通信
        vehicle_groups = {0: vehicle_ids}
        manager.perform_communication(vehicle_groups, current_time=10)
        print("✓ Communication test successful!")
        
        # 测试预测和决策
        positions = {vid: np.random.rand(2) * 100 for vid in vehicle_ids}
        adj_matrix = np.eye(len(vehicle_ids))
        
        actions = manager.predict_and_decide(
            vehicle_groups=vehicle_groups,
            vehicle_positions=positions,
            adjacency_matrix=adj_matrix,
            current_time=11
        )
        
        print(f"Generated actions for {len(actions)} vehicles:")
        for vid, action in actions.items():
            print(f"  Vehicle {vid}: {action:.3f}")
        
        print("✓ GODE manager test successful!")
        
        return True
    
    def test_training_pipeline(self):
        """测试训练流程"""
        print("\n" + "="*50)
        print("Test 4: Training Pipeline")
        print("="*50)
        
        # 创建训练器
        trainer = GODETrainer(self.config, self.device)
        
        # 生成少量数据
        train_data = trainer.generate_synthetic_data(
            num_samples=10,
            num_vehicles=3
        )
        
        print(f"Generated {len(train_data)} training samples")
        
        # 训练几步
        for epoch in range(3):
            loss = trainer.train_epoch(train_data, epoch)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        print("✓ Training pipeline test successful!")
        
        return True
    
    def test_save_load(self):
        """测试模型保存和加载"""
        print("\n" + "="*50)
        print("Test 5: Model Save and Load")
        print("="*50)
        
        manager = GODECoordinationManager(self.config, self.device)
        
        # 保存
        save_dir = './test_checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        manager.save_model(save_dir, epoch=0)
        print(f"Saved model to {save_dir}")
        
        # 修改模型参数
        original_params = {
            name: param.clone() 
            for name, param in manager.gode_model.named_parameters()
        }
        
        with torch.no_grad():
            for param in manager.gode_model.parameters():
                param.fill_(0.0)
        
        # 加载
        checkpoint_path = os.path.join(save_dir, 'gode_checkpoint_epoch_0.pth')
        manager.load_model(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")
        
        # 验证参数恢复
        for name, param in manager.gode_model.named_parameters():
            assert torch.allclose(param, original_params[name]), \
                f"Parameter {name} not restored correctly!"
        
        print("✓ Save/load test successful!")
        
        # 清理
        import shutil
        shutil.rmtree(save_dir)
        
        return True
    
    def test_graph_construction(self):
        """测试图构建"""
        print("\n" + "="*50)
        print("Test 6: Graph Construction")
        print("="*50)
        
        # 测试不同的位置配置
        positions = np.array([
            [0, 0],
            [10, 0],
            [20, 0],
            [100, 100],  # 远离的车辆
        ])
        
        graph = build_vehicle_graph(positions, max_comm_range=15.0)
        
        print(f"Positions:\n{positions}")
        print(f"Communication range: 15.0")
        print(f"Graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        
        # 检查边
        edges = graph.edges()
        src, dst = edges[0].cpu().numpy(), edges[1].cpu().numpy()
        print(f"Edges: {list(zip(src, dst))}")
        
        # 验证距离约束
        for i, j in zip(src, dst):
            if i != j:  # 排除自环
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist <= 15.0, f"Edge ({i},{j}) violates distance constraint!"
        
        print("✓ Graph construction test successful!")
        
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("GODE Integration Test Suite")
        print("="*60)
        
        tests = [
            ("Model Forward Pass", self.test_model_forward),
            ("Action Module", self.test_action_module),
            ("GODE Manager", self.test_gode_manager),
            ("Training Pipeline", self.test_training_pipeline),
            ("Save/Load", self.test_save_load),
            ("Graph Construction", self.test_graph_construction),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"\n✗ Test '{test_name}' failed with error:")
                print(f"  {type(e).__name__}: {e}")
                results.append((test_name, False))
        
        # 总结
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        for test_name, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status:8} | {test_name}")
        
        total = len(results)
        passed = sum(1 for _, s in results if s)
        
        print("="*60)
        print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
        
        return all(s for _, s in results)


def main():
    parser = argparse.ArgumentParser(description='GODE Integration Test')
    parser.add_argument('--config', type=str, choices=['default', 'training', 'eval', 'fast'],
                       default='fast', help='Config type to use')
    parser.add_argument('--test', type=str, default='all',
                       help='Specific test to run (or "all")')
    args = parser.parse_args()
    
    # 加载配置
    if args.config == 'default':
        config = get_default_config()
    elif args.config == 'training':
        config = get_training_config()
    elif args.config == 'eval':
        config = get_eval_config()
    else:
        config = get_fast_config()
    
    print(f"Using config: {args.config}")
    print(f"GODE T_comm: {config.gode.T_comm}")
    print(f"GODE T_action: {config.gode.T_action}")
    
    # 创建测试套件
    test_suite = GODEIntegrationTest(config)
    
    # 运行测试
    if args.test == 'all':
        success = test_suite.run_all_tests()
    else:
        # 运行单个测试
        test_map = {
            'forward': test_suite.test_model_forward,
            'action': test_suite.test_action_module,
            'manager': test_suite.test_gode_manager,
            'training': test_suite.test_training_pipeline,
            'save_load': test_suite.test_save_load,
            'graph': test_suite.test_graph_construction,
        }
        
        if args.test in test_map:
            success = test_map[args.test]()
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {list(test_map.keys())}")
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
