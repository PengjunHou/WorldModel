"""
快速测试脚本：验证所有模块可以正常运行
"""
import torch
import sys

print("="*60)
print("ST-GAT Q-Network Quick Test")
print("="*60)

# 测试导入
print("\n1. Testing imports...")
try:
    from config import Config
    from vae_module import PerceptionMapVAE, InterestMapEncoder
    from stgat_module import SpatioTemporalGAT, EdgeBuilder
    from feature_extractor import ComprehensiveFeatureExtractor
    from q_generator import CNNQGenerator
    from model import STGATQNetwork
    from environment import VehicularEnvironment
    print("✓ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# 测试配置
print("\n2. Testing configuration...")
config = Config()
print(f"✓ Device: {config.DEVICE}")
print(f"✓ Num vehicles: {config.NUM_VEHICLES}")
print(f"✓ Map size: {config.MAP_HEIGHT}x{config.MAP_WIDTH}")

# 测试VAE
print("\n3. Testing VAE module...")
try:
    vae = PerceptionMapVAE(
        map_size=(config.MAP_HEIGHT, config.MAP_WIDTH),
        latent_dim=config.VAE_LATENT_DIM
    )
    test_input = torch.randn(2, 1, config.MAP_HEIGHT, config.MAP_WIDTH)
    recon, mu, logvar = vae(test_input)
    latent = vae.get_latent(test_input)
    print(f"✓ VAE test passed")
    print(f"  Input: {test_input.shape}")
    print(f"  Latent: {latent.shape}")
    print(f"  Recon: {recon.shape}")
except Exception as e:
    print(f"❌ VAE test failed: {e}")
    sys.exit(1)

# 测试ST-GAT
print("\n4. Testing ST-GAT module...")
try:
    stgat = SpatioTemporalGAT(
        feature_dim=256,
        hidden_dim=512,
        num_spatial_layers=2,
        num_temporal_layers=2
    )
    
    # 模拟输入
    N_v, T = 5, 3
    features = torch.randn(N_v, T, 256)
    
    # 模拟边
    edge_indices = []
    for _ in range(T):
        edge_index = torch.randint(0, N_v, (2, 10))  # 10条随机边
        edge_indices.append(edge_index)
    
    output = stgat(features, edge_indices)
    print(f"✓ ST-GAT test passed")
    print(f"  Input: {features.shape}")
    print(f"  Output: {output.shape}")
except Exception as e:
    print(f"❌ ST-GAT test failed: {e}")
    sys.exit(1)

# 测试完整模型
print("\n5. Testing complete Q-Network...")
try:
    model = STGATQNetwork(config)
    
    # 模拟输入
    batch_data = {
        'perception_maps_history': torch.randn(
            config.NUM_VEHICLES,
            config.HISTORY_LENGTH,
            config.MAP_HEIGHT,
            config.MAP_WIDTH
        ),
        'interest_maps_history': torch.randn(
            config.HISTORY_LENGTH,
            config.MAP_HEIGHT,
            config.MAP_WIDTH
        ),
        'positions_history': torch.randn(
            config.NUM_VEHICLES,
            config.HISTORY_LENGTH,
            2
        )
    }
    
    q_maps = model(batch_data)
    print(f"✓ Q-Network test passed")
    print(f"  Output Q-maps: {q_maps.shape}")
    
    # 测试Whittle Index计算
    confidence_current = torch.randn_like(q_maps)
    confidence_prev = torch.randn_like(q_maps)
    whittle_indices = model.compute_whittle_index(
        q_maps,
        confidence_current,
        confidence_prev
    )
    print(f"  Whittle indices: {whittle_indices.shape}")
    
    # 测试top-K选择
    actions = model.select_top_k_regions(whittle_indices, k=config.TOP_K)
    print(f"  Actions (top-{config.TOP_K}): {actions.shape}, sum={actions.sum().item()}")
    
except Exception as e:
    print(f"❌ Q-Network test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试环境
print("\n6. Testing environment...")
try:
    env = VehicularEnvironment(config)
    state = env.reset()
    
    print(f"✓ Environment test passed")
    print(f"  Perception maps history: {state['perception_maps_history'].shape}")
    print(f"  Interest maps history: {state['interest_maps_history'].shape}")
    print(f"  Positions history: {state['positions_history'].shape}")
    
    # 测试一步
    actions = torch.rand(
        config.NUM_VEHICLES,
        config.MAP_HEIGHT,
        config.MAP_WIDTH
    )
    actions = (actions > 0.75).float()
    
    next_state, reward, done, info = env.step(actions)
    print(f"  Step reward: {reward:.2f}")
    print(f"  Done: {done}")
    
except Exception as e:
    print(f"❌ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试完整前向传播流程
print("\n7. Testing end-to-end forward pass...")
try:
    model = STGATQNetwork(config)
    env = VehicularEnvironment(config)
    
    state = env.reset()
    q_maps = model(state)
    
    # 计算Whittle Index
    current_perception = state['perception_maps_history'][:, -1, :, :]
    prev_perception = state['perception_maps_history'][:, -2, :, :]
    
    whittle_indices = model.compute_whittle_index(
        q_maps,
        current_perception,
        prev_perception
    )
    
    # 选择动作
    actions = model.select_top_k_regions(whittle_indices, k=config.TOP_K)
    
    # 执行
    next_state, reward, done, info = env.step(actions)
    
    print(f"✓ End-to-end test passed")
    print(f"  Q-maps: {q_maps.shape}")
    print(f"  Whittle indices: {whittle_indices.shape}")
    print(f"  Actions: {actions.shape}, num_selected={actions.sum().item()}")
    print(f"  Reward: {reward:.2f}")
    
except Exception as e:
    print(f"❌ End-to-end test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 统计模型参数
print("\n8. Model statistics...")
try:
    model = STGATQNetwork(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
except Exception as e:
    print(f"❌ Model statistics failed: {e}")

print("\n" + "="*60)
print("✅ All tests passed successfully!")
print("="*60)
print("\nYou can now:")
print("1. Train VAE: python pretrain_vae.py")
print("2. Train Q-Network: python train.py")
print("3. Evaluate model: python evaluate.py")
print("="*60 + "\n")
