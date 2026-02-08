"""
VAE预训练脚本
用于预训练N map编码器
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os,sys
from tqdm import tqdm
from env import make_env
from src.config import Config
import matplotlib.pyplot as plt
import matplotlib
from .vae_module import PerceptionMapVAE, vae_loss, InterestMapEncoder

matplotlib.use('Agg')  

class PerceptionMapDataset(Dataset):
    def __init__(self, env_config, model_config: Config):
        self.env_config = env_config
        self.model_config = model_config
        self.samples = []
        self.env = make_env(args=env_config, dream_env=False, render_mode=False)
        
        os.makedirs(model_config.VAE_PRETRAIN_DATA_DIR, exist_ok=True)
        self.cache_path = os.path.join(model_config.VAE_PRETRAIN_DATA_DIR, "vae_interest_dataset.npy")
        if os.path.exists(self.cache_path):
            print(f"✓ Loaded cached dataset: {self.cache_path}")
            self.samples = np.load(self.cache_path)
        else:
            print("Generating interest map dataset...")
            # self.env = make_env(args=env_config, dream_env=False, render_mode=False)
            num_samples = 0
            for _ in range(100):
                obs, info = self.env.reset()
                done = False
                while not done:
                    states = self.env.wrapper_state(obs)   # 获取状态特征和邻接矩阵
                    # states_tuple = (local_maps, fused_maps, position, interested_maps, adj
                    
                    actions = np.random.rand(*states['local_maps'][:, -1].shape)
                    threshold = 1 - self.env_config.TOP_K / (self.model_config.MAP_HEIGHT * self.model_config.MAP_WIDTH)
                    actions = (actions > threshold).astype(float)

                    obs, reward, terminated, truncated, _ = self.env.step(actions)

                    done = terminated 

                    self.samples.append(np.expand_dims(states['interest_maps'][-1], axis = 0))  # [1, H, W]
                    num_samples +=1 
                    if num_samples >= model_config.VAE_NUM_SAMPLES:
                        np.save(self.cache_path, np.array(self.samples))
                        print(f"✓ Cached interest dataset saved: {self.cache_path}")
                        print(f"interest Dataset size: {len(self.samples)}")
                        return
                               
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def train_vae(env_config, model_config: Config):
    """训练或继续训练VAE"""
    print("\n" + "="*60)
    print("VAE Pretraining (Resume supported)")
    print("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== 1. 加载数据集 ==========
    dataset = PerceptionMapDataset(env_config, model_config)
    dataloader = DataLoader(
        dataset,
        batch_size=model_config.VAE_PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # ========== 2. 创建模型 ==========
    MAP_HEIGHT, MAP_WIDTH = dataset.env.dataset.global_bev_config['grid_size']
    vae = InterestMapEncoder(
        map_size=(MAP_HEIGHT, MAP_WIDTH),
        latent_dim=model_config.INTEREST_ENCODER_DIM
    ).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=model_config.VAE_PRETRAIN_LR)
    start_epoch = 0
    loss_log = []

    # ========== 3. 检查是否有 checkpoint ==========
    if os.path.exists(model_config.VAE_N_CHECKPOINT_PATH):
        print(f"✓ Found checkpoint: {model_config.VAE_N_CHECKPOINT_PATH}")
        checkpoint = torch.load(model_config.VAE_N_CHECKPOINT_PATH, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming training from epoch {start_epoch} ...")

    # ========== 4. 训练循环 ==========
    vae.train()
    for epoch in range(start_epoch, model_config.VAE_PRETRAIN_EPOCHS):
        total_loss = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{model_config.VAE_PRETRAIN_EPOCHS}")

        for batch in pbar:
            batch = batch.to(device, dtype=torch.float32)
            recon, mu, logvar = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar) / batch.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        loss_log.append(avg_loss)

        # ========== 5. 保存 checkpoint ==========
        os.makedirs(model_config.VAE_N_CHECKPOINT_DIR, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, model_config.VAE_N_CHECKPOINT_PATH)
        print(f"✓ Saved checkpoint: {model_config.VAE_N_CHECKPOINT_PATH}")

    print("\n✓ interested map VAE training complete!")

    # ========== 6. 可视化 loss ==========
    vae.eval()
    plt.figure(figsize=(10, 4))
    plt.plot(loss_log)
    plt.title("Training Loss Curve (Resume Supported)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()

    save_dir = model_config.VAE_VISUALIZE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "N_vae_loss_resume.png")
    plt.savefig(save_path, dpi=200)
    print(f"✓ Loss curve saved to {save_path}")

    return vae


def load_trained_vae(env_config, model_config:Config, checkpoint_path, device="cuda"):
    # 从环境获取 map 尺寸
    env = make_env(args=env_config, dream_env=False, render_mode=False)
    MAP_HEIGHT, MAP_WIDTH = env.dataset.global_bev_config['grid_size']

    # 初始化模型结构（参数必须和训练时一致）
    vae = InterestMapEncoder(
        map_size=(MAP_HEIGHT, MAP_WIDTH),
        latent_dim=model_config.INTEREST_ENCODER_DIM
    ).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()   # 切换为推理模式（关闭dropout、BN等）
    
    print(f"✓ Loaded pretrained Interested map VAE from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return vae

def visualize_vae_reconstruction(vae, dataloader, device, save_dir, num_samples=4):
    """
    可视化 VAE 重构效果
    ----------------------------------------------------
    参数:
        vae: 已训练好的 VAE 模型
        dataloader: DataLoader（和训练时相同）
        device: torch.device
        save_dir: 可选，保存可视化图的目录
        num_samples: 展示样本数量
    """
    vae.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # 从 dataloader 取一个 batch
        for batch in dataloader:
            batch = batch.to(device, dtype=torch.float32)
            recon, mu, logvar = vae(batch)
            break   # 只取一个 batch 展示

        # 限制展示样本数量
        n = min(num_samples, batch.size(0))

        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(6, 3 * n))
        if n == 1:
            axes = [axes]  # 兼容单张情况

        for i in range(n):
            original = batch[i, 0].cpu().numpy()
            reconstructed = recon[i, 0].cpu().numpy()

            # 原图
            axes[i][0].imshow(original, cmap='viridis')
            axes[i][0].set_title("Original")
            axes[i][0].axis('off')

            # 重构图
            axes[i][1].imshow(reconstructed, cmap='viridis')
            axes[i][1].set_title("Reconstruction")
            axes[i][1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "interested_map_vae_reconstruction.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"✓ 可视化已保存到: {save_path}")

def test_single_sample():
    """
    测试 VAE 是否能过拟合真实数据集中某一张图。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ====== 1. 载入配置与数据集 ======
    env_config = PARSER.parse_args()
    model_config = Config()
    dataset = PerceptionMapDataset(env_config, model_config)

    # 取第一张样本作为测试图
    sample = torch.tensor(dataset[0], dtype=torch.float32).unsqueeze(0).to(device)  # [1,1,16,16]
    print(f"Loaded one sample from dataset, shape: {sample.shape}")

    # ====== 2. 初始化模型与优化器 ======
    vae = InterestMapEncoder(map_size=(128, 128), latent_dim=128, beta=0.1).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)

    # ====== 3. 单图过拟合训练 ======
    loss_log = []
    for step in range(20000):
        recon, mu, logvar = vae(sample)
        loss = vae_loss(recon, sample, mu, logvar, beta=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"[{step+1}/2000] loss={loss.item():.4f}")

    # ====== 4. 可视化结果 ======
    vae.eval()
    with torch.no_grad():
        recon, _, _ = vae(sample)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(sample[0, 0].cpu(), cmap='viridis')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(recon[0, 0].cpu(), cmap='viridis')
    plt.title('Reconstruction')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.plot(loss_log)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.tight_layout()
    save_dir=model_config.VAE_VISUALIZE_DIR
    save_path = os.path.join(save_dir, "test_N_vae_reconstruction.png")
    plt.savefig(save_path, dpi=200)  

if __name__ == '__main__':
    # test_single_sample()
    env_config = PARSER.parse_args()
    model_config = Config()
    
    # train_vae(env_config, model_config)

    vae = load_trained_vae(
        env_config, model_config,
        checkpoint_path=model_config.VAE_N_CHECKPOINT_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # model_config.VAE_NUM_SAMPLES = 20
    dataset = PerceptionMapDataset(env_config, model_config)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    visualize_vae_reconstruction(
        vae,
        dataloader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=model_config.VAE_VISUALIZE_DIR,
        num_samples=4
    )