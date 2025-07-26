
import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pythae.data.datasets import DatasetOutput

class MemmapDataset(Dataset):
    def __init__(self, mem):
        self.mem   = mem              # numpy.memmap (N,C,H,W)
        self.dtype = mem.dtype

    def __len__(self):
        return len(self.mem)

    def __getitem__(self, idx):
        img = self.mem[idx]

        # → float32 0‑1
        if self.dtype == np.uint8:
            img = img.astype(np.float32) / 255.
        else:
            img = img.astype(np.float32)

        img = torch.from_numpy(img)    # CHW, float32
        return DatasetOutput(data=img, labels=None)   # <-- 关键



def imshow_tensor(ax, tensor):
    img = tensor.detach().cpu()
    if img.shape[0] == 1:
        img = img.squeeze(0)
        ax.imshow(img, cmap='gray')
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0)
        ax.imshow(img)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    ax.axis('off')