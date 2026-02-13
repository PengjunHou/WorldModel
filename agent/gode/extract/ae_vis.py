import os
import numpy as np
from PIL import Image

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .ae_model import AutoEncoder


# -------------------------
# Utils: load image -> tensor
# -------------------------
def load_img_tensor(path: str, img_size=(256, 144)):
    img_w, img_h = img_size
    img = Image.open(path).convert("RGB").resize((img_w, img_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return x


def tensor_to_uint8_img(x: torch.Tensor):
    """
    x: (3,H,W) float in [0,1]
    return HxWx3 uint8
    """
    x = x.detach().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return arr


# -------------------------
# Detector wrappers (optional)
# -------------------------
class BaseDetector:
    def detect(self, img_uint8: np.ndarray):
        """
        img_uint8: HxWx3 uint8 (RGB)
        return boxes, scores, labels
          boxes: (M,4) in xyxy
          scores: (M,)
          labels: list[str] or list[int]
        """
        raise NotImplementedError


class YOLOv8Detector(BaseDetector):
    def __init__(self, weight="yolov8n.pt", device=None, conf=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weight)
        self.device = device
        self.conf = conf

    def detect(self, img_uint8: np.ndarray):
        # ultralytics expects numpy RGB ok
        res = self.model.predict(img_uint8, conf=self.conf, verbose=False, device=self.device)
        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), []

        boxes = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls = r0.boxes.cls.detach().cpu().numpy().astype(np.int32)

        # names mapping
        names = r0.names if hasattr(r0, "names") else None
        labels = [names[int(c)] if names is not None else int(c) for c in cls]
        return boxes, scores, labels


class TorchvisionFRCNNDetector(BaseDetector):
    def __init__(self, device="cuda", score_thr=0.5):
        import torchvision
        self.device = device
        self.score_thr = score_thr
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(device).eval()
        # COCO label map (简版)
        self.coco_names = ["__bg__"] + [str(i) for i in range(1, 91)]

    @torch.no_grad()
    def detect(self, img_uint8: np.ndarray):
        # torchvision expects float tensor 0..1 in (C,H,W)
        x = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0
        x = x.to(self.device)
        out = self.model([x])[0]
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        keep = scores >= self.score_thr
        boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)[keep]
        scores = scores[keep]
        labels_id = out["labels"].detach().cpu().numpy().astype(np.int32)[keep]
        labels = [self.coco_names[i] if i < len(self.coco_names) else int(i) for i in labels_id]
        return boxes, scores, labels


def build_detector(prefer="yolo", device="cuda"):
    """
    自动选择 detector：
      - prefer="yolo": 优先 YOLOv8（需要 ultralytics）
      - 否则 fallback 到 torchvision FasterRCNN（需要 torchvision + 可能下载权重）
    """
    if prefer == "yolo":
        try:
            return YOLOv8Detector(weight="yolov8n.pt", device=device, conf=0.25)
        except Exception as e:
            print("[warn] YOLOv8Detector not available:", e)

    try:
        return TorchvisionFRCNNDetector(device=device, score_thr=0.5)
    except Exception as e:
        print("[warn] Torchvision detector not available:", e)

    return None


# -------------------------
# Draw boxes
# -------------------------
def draw_boxes(ax, boxes, scores, labels, max_det=20):
    for i in range(min(len(boxes), max_det)):
        x1, y1, x2, y2 = boxes[i]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        ax.add_patch(rect)
        txt = f"{labels[i]} {scores[i]:.2f}" if len(scores) > 0 else f"{labels[i]}"
        ax.text(x1, y1, txt, fontsize=9, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))


# -------------------------
# Simple detection comparison: IoU matching
# -------------------------
def iou_xyxy(a, b):
    # a,b: (4,)
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def match_detections(boxes_a, boxes_b, iou_thr=0.5):
    """
    返回简单匹配数：greedy
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return 0
    used = set()
    matched = 0
    for i in range(len(boxes_a)):
        best_j, best_iou = -1, 0.0
        for j in range(len(boxes_b)):
            if j in used:
                continue
            v = iou_xyxy(boxes_a[i], boxes_b[j])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_thr and best_j >= 0:
            used.add(best_j)
            matched += 1
    return matched


# -------------------------
# Main visualization function
# -------------------------
@torch.no_grad()
def visualize_recon_and_detection(
    ae_ckpt: str,
    image_path: str,
    out_dir: str = "ae_vis",
    img_size=(256, 144),
    z_dim=128,
    device="cuda",
    detector_prefer="yolo",
    max_det=20,
):
    os.makedirs(out_dir, exist_ok=True)

    # load AE
    ae = AutoEncoder(z_dim=z_dim, out_hw=(img_size[1], img_size[0])).to(device).eval()
    ck = torch.load(ae_ckpt, map_location=device)
    ae.load_state_dict(ck["model"])

    # load image
    x = load_img_tensor(image_path, img_size=img_size).unsqueeze(0).to(device)  # (1,3,H,W)

    # recon
    z, x_hat = ae(x)
    x0 = x[0]
    xh = x_hat[0]

    img0 = tensor_to_uint8_img(x0)
    imgh = tensor_to_uint8_img(xh)

    # detector
    det = build_detector(prefer=detector_prefer, device=device)
    det0 = det.detect(img0) if det is not None else (np.zeros((0,4)), np.zeros((0,)), [])
    deth = det.detect(imgh) if det is not None else (np.zeros((0,4)), np.zeros((0,)), [])

    boxes0, scores0, labels0 = det0
    boxesh, scoresh, labelsh = deth

    # compare
    matched = match_detections(boxes0, boxesh, iou_thr=0.5)
    summary = {
        "n_det_orig": int(len(boxes0)),
        "n_det_recon": int(len(boxesh)),
        "matched_iou>=0.5": int(matched),
    }
    print("[detect compare]", summary)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(img0)
    axs[0].set_title(f"Original | det={len(boxes0)}")
    axs[0].axis("off")
    draw_boxes(axs[0], boxes0, scores0, labels0, max_det=max_det)

    axs[1].imshow(imgh)
    axs[1].set_title(f"Reconstruction | det={len(boxesh)} | matched={matched}")
    axs[1].axis("off")
    draw_boxes(axs[1], boxesh, scoresh, labelsh, max_det=max_det)

    out_png = os.path.join(out_dir, "orig_vs_recon_det.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()

    # also save raw recon image for quick look
    Image.fromarray(imgh).save(os.path.join(out_dir, "recon.png"))
    Image.fromarray(img0).save(os.path.join(out_dir, "orig.png"))

    return summary, out_png


@torch.no_grad()
def load_ae_and_detector(
    ae_ckpt: str,
    z_dim: int = 128,
    img_size=(256, 144),
    device="cuda",
    detector_prefer="yolo",
):
    """
    Load once. Return (ae, detector).
    """
    ae = AutoEncoder(z_dim=z_dim, out_hw=(img_size[1], img_size[0])).to(device).eval()
    ck = torch.load(ae_ckpt, map_location=device)
    ae.load_state_dict(ck["model"])
    det = build_detector(prefer=detector_prefer, device=device)
    return ae, det

@torch.no_grad()
def visualize_predz_vs_gtimage_detection(
    ae: AutoEncoder,
    detector,
    gt_image_path: str,
    pred_z: torch.Tensor,     # (Z,) on device
    out_png: str,
    img_size=(256,144),
    max_det=20,
):
    # 1) load GT raw image
    x_gt = load_img_tensor(gt_image_path, img_size=img_size).unsqueeze(0).to(pred_z.device)  # (1,3,H,W)

    # 2) decode pred_z -> recon image
    # 你 AE 模型如果没有单独 decode 接口，就用 ae.dec(...)
    x_pred = ae.dec(pred_z.unsqueeze(0))  # (1,3,H,W)  ⚠️按你的 AE 实现可能是 ae.decoder(...)
    x_pred = x_pred.clamp(0, 1)

    img0 = tensor_to_uint8_img(x_gt[0])
    imgh = tensor_to_uint8_img(x_pred[0])

    # 3) detection compare
    if detector is None:
        boxes0, scores0, labels0 = np.zeros((0,4)), np.zeros((0,)), []
        boxesh, scoresh, labelsh = np.zeros((0,4)), np.zeros((0,)), []
        matched = 0
    else:
        boxes0, scores0, labels0 = detector.detect(img0)
        boxesh, scoresh, labelsh = detector.detect(imgh)
        matched = match_detections(boxes0, boxesh, iou_thr=0.5)

    # 4) plot
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(img0); axs[0].axis("off")
    axs[0].set_title(f"GT Raw | det={len(boxes0)}")
    draw_boxes(axs[0], boxes0, scores0, labels0, max_det=max_det)

    axs[1].imshow(imgh); axs[1].axis("off")
    axs[1].set_title(f"Pred(z)->Recon | det={len(boxesh)} | matched={matched}")
    draw_boxes(axs[1], boxesh, scoresh, labelsh, max_det=max_det)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"[visualize_predz_vs_gtimage_detection] saved visualization to: {out_png}")
    plt.close()

    return {
        "n_det_gt": int(len(boxes0)),
        "n_det_pred_recon": int(len(boxesh)),
        "matched_iou>=0.5": int(matched),
    }

import os
import numpy as np
import torch

@torch.no_grad()
def encode_img_paths_to_z(
    ae,
    paths_2d,
    img_size=(256, 144),
    device="cuda",
    batch_imgs: int = 128,
    return_device: str = "cpu",
):
    """
    Encode a 2D array of image paths into AE latent z.

    Args
    ----
    ae: AutoEncoder (must have ae.enc(x) -> (B,Z))
    paths_2d: np.ndarray or list-like, shape (L,N) or (T,N), dtype=str
    img_size: (W,H)
    device: where AE runs (cuda/cpu)
    batch_imgs: batch size for encoding
    return_device: 'cpu' or 'cuda' for returned tensor

    Returns
    -------
    z: torch.Tensor, shape (L,N,Z), float32
    """
    from PIL import Image  # keep local to avoid extra import needs

    img_w, img_h = img_size

    # ---- normalize input to numpy array ----
    paths = np.asarray(paths_2d)
    assert paths.ndim == 2, f"paths_2d must be 2D (L,N). Got shape={paths.shape}"
    L, N = paths.shape

    # infer z_dim robustly
    z_dim = None
    if hasattr(ae, "enc") and hasattr(ae.enc, "fc") and hasattr(ae.enc.fc, "out_features"):
        z_dim = int(ae.enc.fc.out_features)
    else:
        # fallback: run a tiny dummy forward
        dummy = torch.zeros(1, 3, img_h, img_w, dtype=torch.float32, device=device)
        z_dim = int(ae.enc(dummy).shape[-1])

    def _load_img(path_str: str):
        if (path_str is None) or (len(str(path_str)) == 0) or (not os.path.exists(str(path_str))):
            return torch.zeros(3, img_h, img_w, dtype=torch.float32)
        img = Image.open(str(path_str)).convert("RGB").resize((img_w, img_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    # flatten
    flat = paths.reshape(-1)
    total = flat.shape[0]

    # output buffer (CPU numpy first, then tensor)
    out = np.zeros((total, z_dim), dtype=np.float32)

    ae = ae.to(device).eval()

    idx = 0
    while idx < total:
        j = min(total, idx + batch_imgs)
        imgs = [_load_img(flat[k]) for k in range(idx, j)]
        x = torch.stack(imgs, dim=0).to(device)  # (bs,3,H,W)
        z = ae.enc(x).detach().float().cpu().numpy()  # (bs,Z)
        out[idx:j] = z
        idx = j

    z = torch.from_numpy(out.reshape(L, N, z_dim)).float()
    if return_device == "cuda":
        z = z.to(device)
    return z


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/home/peh324/Codes/WorldModel/result/checkpoints/ae/ae_best.pt"
    
    image_path = "/home/peh324/Codes/WorldModel/data/GODE/images/actor_941/frame_00019143.png"  # 替换成实际图片路径
    out_dir = "/home/peh324/Codes/WorldModel/result/ae/"
    summary, out_png = visualize_recon_and_detection(
        ae_ckpt=checkpoint_path,
        image_path=image_path,
        out_dir=out_dir,
        img_size=(256,144),
        z_dim=128,
        device=device,
        detector_prefer="yolo",
    )
    print("saved:", out_png)
