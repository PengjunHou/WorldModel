import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Total Variation Loss ----------
def total_variation_loss(x):
    """
    x: [B, 1, H, W]  (reconstructed confidence map)
    Returns scalar TV regularization term.
    """
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return (dh.mean() + dw.mean())

class ConfidenceMapAutoEncoder(nn.Module):     
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # ---------- Encoder ----------
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, H/4, W/4]
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, H/8, W/8]
            nn.ReLU(inplace=True)
        )
        
        # ---------- Latent ----------
        self.latent_dim = latent_dim
        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)   # 默认假设输入尺寸为 64×64，可自动调整
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [B, 1, H, W]
        h = self.encoder(x)
        h_flat = h.flatten(start_dim=1)
        z = self.fc_enc(h_flat)
        h_dec = self.fc_dec(z)
        h_dec = h_dec.view(x.size(0), 64, 8, 8)
        out = self.decoder(h_dec)
        return out, z


# ---------- Example Training Loop ----------
if __name__ == "__main__":
    model = ConfidenceMapAutoEncoder(latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    lambda_tv = 1e-4   # 控制平滑强度，可根据实验调整

    dummy_input = torch.rand(4, 1, 64, 64)

    for step in range(1000):
        recon, z = model(dummy_input)
        loss_recon = mse_loss(recon, dummy_input)
        loss_tv = total_variation_loss(recon)
        loss = loss_recon + lambda_tv * loss_tv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: MSE={loss_recon.item():.6f}, TV={loss_tv.item():.6f}, Total={loss.item():.6f}")
