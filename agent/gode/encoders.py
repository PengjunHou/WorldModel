
import torch.nn as nn
import torch

class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (B*L*N, 3, H, W)
        h = self.net(x).flatten(1)     # (B*L*N, 128)
        return self.proj(h)            # (B*L*N, out_dim)

class CameraEncoder(nn.Module):
    def __init__(self, feat_dim=128, hidden_dim=128):
        super().__init__()
        self.cnn = SmallCNN(out_dim=feat_dim)
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, images_past):
        """
        images_past: (B, L, N, 3, H, W)
        return z_cam: (B, N, hidden_dim)
        """
        B, L, N, C, H, W = images_past.shape
        x = images_past.reshape(B*L*N, C, H, W)
        f = self.cnn(x).reshape(B, L, N, -1)      # (B,L,N,feat)
        f = f.permute(0, 2, 1, 3).contiguous()    # (B,N,L,feat)
        f = f.reshape(B*N, L, -1)                 # (B*N,L,feat)
        _, h = self.gru(f)                        # h: (1, B*N, hidden)
        z = h.squeeze(0).reshape(B, N, -1)        # (B,N,hidden)
        return z

class StateEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, states_past):
        # states_past: (B, L, N, 6)
        B, L, N, D = states_past.shape
        x = states_past.permute(0, 2, 1, 3).contiguous().reshape(B*N, L, D)
        _, h = self.gru(x)
        z = h.squeeze(0).reshape(B, N, -1)
        return z
