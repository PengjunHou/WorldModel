# data_logger.py
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image

@dataclass
class Record:
    frame: int
    timestep: int
    actor_id: int
    group_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw: float
    image_path: Optional[str]  # 如果没有camera就为None


class CarlaDataLogger:
    def __init__(self, root_dir: str, run_name: str, flush_every: int = 200):
        self.run_dir = os.path.join(root_dir, run_name)
        self.img_dir = os.path.join(self.run_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)

        self.flush_every = flush_every
        self.records: List[Record] = []

        # 最终会写到这个文件
        self.meta_path = os.path.join(self.run_dir, "records.jsonl")

    def _save_image(self, actor_id: int, frame: int, img: np.ndarray) -> str:
        """
        img: (H, W, 3) uint8 or (3, H, W) uint8
        """
        actor_dir = os.path.join(self.img_dir, f"actor_{actor_id}")
        os.makedirs(actor_dir, exist_ok=True)

        if img.ndim == 3 and img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)

        path = os.path.join(actor_dir, f"frame_{frame:08d}.png")
        Image.fromarray(img).save(path)
        return path

    def log_step(
        self,
        frame: int,
        timestep: int,
        actor_id: int,
        group_id: int,
        transform,   # carla.Transform
        velocity,    # carla.Vector3D
        yaw: float,
        obs: Optional[Dict[str, Any]] = None
    ) -> None:
        img_path = None

        # 这里假设 observer 给的 obs 里有 camera/rgb 等字段
        # 你需要根据你 Observer 的真实 key 调一下（我后面会给你定位方法）
        if obs is not None:
            # 常见key候选："rgb", "camera", "front_camera", "image"
            for k in ["camera"]: # "lidar", "collision"
                if k in obs:
                    img = obs[k]
                    if img is None:
                        print(f"Warning: obs[{k}] is None")
                        return
                    # 允许 obs[k] 是 dict { "data": ndarray } 这种
                    if isinstance(img, dict) and "data" in img:
                        img = img["data"]
                    img_path = self._save_image(actor_id, frame, img)
                    break

        loc = transform.location
        rec = Record(
            frame=int(frame),
            timestep=int(timestep),
            actor_id=int(actor_id),
            group_id=int(group_id),
            x=float(loc.x), y=float(loc.y), z=float(loc.z),
            vx=float(velocity.x), vy=float(velocity.y), vz=float(velocity.z),
            yaw=float(yaw),
            image_path=img_path
        )
        self.records.append(rec)

        if len(self.records) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        # jsonl 追加写，安全、可中断恢复
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.__dict__) + "\n")
        self.records = []

    def close(self) -> None:
        self.flush()
