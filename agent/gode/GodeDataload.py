import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class TrajDatasetPaths(Dataset):
    def __init__(self, npz_path, img_size=(112, 112)):
        self.data = np.load(npz_path, allow_pickle=True)
        self.img_paths = self.data["image_paths_past"]      # (B,L,N)
        self.x_past = self.data["states_past"]              # (B,L,N,6)
        self.y = self.data["states_future"]                 # (B,H,N,6)
        self.group_id = self.data["group_id"]               # (B,N)
        self.mask = self.data["node_mask"]                  # (B,N)
        self.img_size = img_size

    def __len__(self):
        return self.x_past.shape[0]

    def _load_rgb(self, path: str):
        # path 可能为空（mask=0的节点），返回全0图
        if (path is None) or (len(path) == 0) or (not isinstance(path, str)):
            return torch.zeros(3, self.img_size[0], self.img_size[1], dtype=torch.float32)

        img = Image.open(path).convert("RGB").resize(self.img_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0     # (H,W,3)
        ten = torch.from_numpy(arr).permute(2, 0, 1)        # (3,H,W)
        return ten

    def __getitem__(self, idx):
        # numpy -> torch
        x = torch.from_numpy(self.x_past[idx]).float()         # (L,N,6)
        y = torch.from_numpy(self.y[idx]).float()              # (H,N,6)
        gid = torch.from_numpy(self.group_id[idx]).long()      # (N,)
        mask = torch.from_numpy(self.mask[idx]).bool()         # (N,)

        # load images: (L,N,3,H,W)
        paths = self.img_paths[idx]
        L, N = paths.shape
        imgs = torch.zeros(L, N, 3, self.img_size[0], self.img_size[1], dtype=torch.float32)
        for t in range(L):
            for i in range(N):
                if mask[i]:
                    imgs[t, i] = self._load_rgb(str(paths[t, i]))
        return {
            "images_past": imgs,   # (L,N,3,H,W)
            "states_past": x,      # (L,N,6)
            "states_future": y,    # (H,N,6)
            "group_id": gid,       # (N,)
            "node_mask": mask      # (N,)
        }
