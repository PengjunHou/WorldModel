
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # 或者 'Qt5Agg'
import torch
import torch.nn as nn
from coperception.models.det import FaFNet, DiscoNet, When2com, V2VNet
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from functools import reduce

import logging
LOG = logging.getLogger(__name__)

# 初始化检测模型 (只做一次)
def init_detection_model(config, num_agent, com="lowerbound", ckpt_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if com == "lowerbound":
        model = FaFNet(config, layer=3, kd_flag=0, num_agent=num_agent)
    elif com == "disco":
        model = DiscoNet(config, layer=3, kd_flag=0, num_agent=num_agent)
    elif com == "when2com":
        model = When2com(config, layer=3, warp_flag=0, 
                         num_agent=num_agent)
    elif com == "v2v":
        model = V2VNet(config, gnn_iter_times=3, layer=3, layer_channel=256, num_agent=num_agent)
    else:
        raise ValueError(f"Unsupported model type: {com}")

    model = nn.DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    fafmodule = FaFModule(model, model, config, optimizer, criterion, kd_flag=0)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
        #print(f"Loaded detection model from {ckpt_path}")

    fafmodule.model.eval()
    return fafmodule, device

import torch




def map_boxes_to_conf_map(local_conf_map, boxes, scores, config):
    """
    将预测的3D检测框及其置信度映射到BEV网格，生成置信度热力图。
    
    Args:
        boxes: numpy.ndarray [N, 4]，每个box的(x, y, w, h)，单位：米
        scores: numpy.ndarray [N]，每个检测框的置信度
        config: Config对象，提供map_dims, area_extents, voxel_size等
    
    Returns:
        conf_map: numpy.ndarray，形状 (map_dims[0], map_dims[1])，置信度分布
    """
    H, W = local_conf_map.shape[0], local_conf_map.shape[1]
    voxel_size_x, voxel_size_y = config.voxel_size[0], config.voxel_size[1]
    x_min, x_max = config.area_extents[0, 0], config.area_extents[0, 1]
    y_min, y_max = config.area_extents[1, 0], config.area_extents[1, 1]

    conf_map = np.zeros((H, W), dtype=np.float32)

    for box, score in zip(boxes, scores):
        x, y, w, h = box[:4]  # 取中心和长宽

        # 转换到BEV网格坐标
        x_min_box = int((x - w / 2 - x_min) / voxel_size_x)
        x_max_box = int((x + w / 2 - x_min) / voxel_size_x)
        y_min_box = int((y - h / 2 - y_min) / voxel_size_y)
        y_max_box = int((y + h / 2 - y_min) / voxel_size_y)

        # 裁剪到合法范围
        x_min_box = max(0, min(H - 1, x_min_box))
        x_max_box = max(0, min(H - 1, x_max_box))
        y_min_box = max(0, min(W - 1, y_min_box))
        y_max_box = max(0, min(W - 1, y_max_box))

        if x_min_box < x_max_box and y_min_box < y_max_box:
            conf_map[x_min_box:x_max_box, y_min_box:y_max_box] = np.maximum(
                conf_map[x_min_box:x_max_box, y_min_box:y_max_box], score
            )

    return conf_map

def visualize_conf_map(conf_map, title="Local Confidence Map", save_path=None):
    """
    可视化置信度BEV热力图
    Args:
        conf_map: numpy.ndarray, 形状 [H, W]
        title: 图像标题
        save_path: 如果提供路径，就保存图片；否则直接plt.show()
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_map.T, cmap="hot", origin="lower")  # 注意转置 + 下方原点
    plt.colorbar(label="Confidence Score")
    plt.title(title)
    plt.xlabel("X-axis (grid)")
    plt.ylabel("Y-axis (grid)")

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def map_local_conf_to_global(local_conf_map, trans_matrix, config):
    """
    local_conf_map: np.array (H, W)，置信度
    trans_matrix: torch.Tensor [4,4]，从local到global
    config: 包含 voxel_size, area_extents
    """
    H, W = config.map_dims
    global_conf_map = np.zeros((H, W), dtype=np.float32)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)], axis=-1).reshape(-1, 4).T

    # local → global
    trans = trans_matrix.cpu().numpy()
    global_coords = np.dot(trans, coords)[:2, :]

    # 映射到 BEV 栅格坐标
    voxel_size = config.voxel_size
    area_extents = config.area_extents
    gx = ((global_coords[0] - area_extents[0][0]) / voxel_size[0]).astype(int)
    gy = ((global_coords[1] - area_extents[1][0]) / voxel_size[1]).astype(int)

    valid = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    global_conf_map[gy[valid], gx[valid]] = local_conf_map.reshape(-1)[valid]

    return global_conf_map

def conf_map_decay(conf_map, decay_factor=0.9):
    """
    对置信度图进行衰减
    Args:
        conf_map: np.array (H, W)，置信度图
        decay_factor: 衰减系数，0 < decay_factor < 1
    Returns:
        decayed_map: np.array (H, W)，衰减后的置信度图
    """
    return conf_map * decay_factor

def fused_local_and_global_conf(local_conf_map, global_conf_map, alpha=0.5):
    """
    融合局部和全局置信度图
    Args:
        local_conf_map: np.array (H, W)，局部置信度
        global_conf_map: np.array (H, W)，全局置信度
        alpha: 局部置信度的权重
    Returns:
        fused_map: np.array (H, W)，融合后的置信度图
    """
    return alpha * local_conf_map + (1 - alpha) * global_conf_map

def get_trans_matrix_to_global(nusc_data, sample_data):
    """
    获取从当前传感器坐标系到全局坐标系的转换矩阵
    Args:
        config: Config对象，包含 area_extents 和 voxel_size
        car_from_current: 当前传感器到ego车的转换矩阵
        global_from_car: 全局坐标系到ego车的转换矩阵
    Returns:
    """
    current_pose_rec = nusc_data.get("ego_pose", sample_data["ego_pose_token"])
    global_from_car = transform_matrix(
        np.sum([current_pose_rec["translation"]], axis=0),
        Quaternion(current_pose_rec["rotation"]),
        inverse=False,
    )

    # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
    current_cs_rec = nusc_data.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    car_from_current = transform_matrix(
        current_cs_rec["translation"],
        Quaternion(current_cs_rec["rotation"]),
        inverse=False,
    )

    # Fuse four transformation matrices into one and perform transform.
    trans_matrix = reduce(
        np.dot, [global_from_car, car_from_current]
    )

    return trans_matrix

def fuse_global_conf_maps(prev_global_map, prev_trans_matrix,
                          curr_global_map, curr_trans_matrix,
                          global_config, decay=0.9):
    """
    融合前一帧和当前帧的global_conf_map（考虑ego pose差异）
    """
    H, W = global_config.map_dims[:2]
    fused_map = np.copy(curr_global_map)

    # Step 1: 构建上一帧 BEV voxel 的像素坐标
    xs = np.linspace(global_config.area_extents[0][0],
                     global_config.area_extents[0][1],
                     W, endpoint=False)
    ys = np.linspace(global_config.area_extents[1][0],
                     global_config.area_extents[1][1],
                     H, endpoint=False)
    xv, yv = np.meshgrid(xs, ys)
    coords = np.stack([xv, yv, np.zeros_like(xv), np.ones_like(xv)], axis=-1)  # (H,W,4)
    coords_flat = coords.reshape(-1, 4).T  # (4, H*W)

    # Step 2: 找出非零置信度 voxel
    mask_prev = prev_global_map > 0
    prev_values = prev_global_map.flatten()[mask_prev.flatten()]
    world_coords_prev = coords_flat[:, mask_prev.flatten()]

    # Step 3: 把上一帧的 voxel 从 “prev global BEV坐标” 转回世界坐标
    world_coords = np.dot(prev_trans_matrix, world_coords_prev)

    # Step 4: 把世界坐标投影到当前global BEV栅格
    coords_in_curr = np.dot(np.linalg.inv(curr_trans_matrix), world_coords)

    x_indices = ((coords_in_curr[0] - global_config.area_extents[0][0]) /
                 global_config.voxel_size[0]).astype(int)
    y_indices = ((coords_in_curr[1] - global_config.area_extents[1][0]) /
                 global_config.voxel_size[1]).astype(int)

    # Step 5: 边界检查
    valid_mask = (
        (x_indices >= 0) & (x_indices < W) &
        (y_indices >= 0) & (y_indices < H)
    )
    x_indices = x_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    prev_values = prev_values[valid_mask]

    # Step 6: 融合（取最大值 + 折旧）
    for xi, yi, val in zip(x_indices, y_indices, prev_values):
        fused_map[yi, xi] = max(fused_map[yi, xi], decay * val)

    return fused_map


def _to_homo(pts_xy):
    M = pts_xy.shape[0]
    out = np.zeros((M,4), dtype=np.float32)
    out[:, :2] = pts_xy
    out[:, 3]  = 1.0
    return out

def bev_local_to_world_corners(T_local_to_world, corners_local_xy):
    """
    corners_local_xy: (N,4,2) 本地BEV角点
    CoPerception 的 BEV 与几何系 x 轴相反：用 A=diag([-1,1,1,1]) 一次性并入矩阵
    """
    A = np.eye(4, dtype=np.float32); A[0,0] = -1.0
    T = np.asarray(T_local_to_world, dtype=np.float32)
    if T.ndim == 3:  # (K,4,4) 时取第0个，或改成与你的agent索引对应
        T = T[0]
    T_eff = A @ T @ A

    N = corners_local_xy.shape[0]
    flat = corners_local_xy.reshape(-1, 2).astype(np.float32)   # (N*4,2)
    homo = _to_homo(flat)                                      # (N*4,4)
    world = (homo @ T_eff.T)[:, :2]                            # (N*4,2)
    return world.reshape(N, 1, 4, 2)                           # 与原实现一致的形状


def _compose_T_world_sensor(nusc, sample_data):
    """
    从 nuScenes 结构里拿 T_world_ego 和 T_ego_sensor，得到 T_world_sensor.
    sample_data: nusc.get('sample_data', token) 的返回；或包含 'ego_pose_token'/'calibrated_sensor_token' 的 dict
    """
    cs = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    pose = nusc.get('ego_pose',        sample_data['ego_pose_token'])
    T_world_ego  = transform_matrix(pose['translation'],  Quaternion(pose['rotation']))
    T_ego_sensor = transform_matrix(cs['translation'],    Quaternion(cs['rotation']))
    return T_world_ego @ T_ego_sensor, T_world_ego



import numpy as np
import torch

def voxel_to_confidence_map(padded_voxel_points,
                            take_last_t=True,
                            z_reduce="sum",           # "sum" 或 "max"
                            smooth_sigma=0.0,         # >0 时用高斯平滑(可选)
                            norm_mode="percentile",   # "minmax" | "percentile"
                            p_low=1.0, p_high=99.0):
    """
    将 [T,H,W,Z] 或 [H,W,Z] 或 [Z,H,W] 的体素 grid 转为 2D 置信度图 (H,W)，并归一化到 [0,1]。
    - take_last_t: 当输入是 [T,H,W,Z] 时，取最后一帧 T=-1；否则对 T 求和也可按需修改。
    - z_reduce: 在 Z 维度上求和("sum")或取最大("max")。
    - smooth_sigma: 若>0，将对 2D 图高斯平滑(需 scipy)；没有 scipy 就设为 0.
    - norm_mode:
        * "minmax": (x - x.min) / (x.max - x.min + 1e-8)
        * "percentile": 按分位数 [p_low, p_high] 裁剪再线性归一化
    """
    voxel = padded_voxel_points
    if isinstance(voxel, torch.Tensor):
        voxel = voxel.detach().cpu().numpy()

    # 统一到 [H,W,Z]
    if voxel.ndim == 4:
        # 可能是 [T,H,W,Z]
        T, H, W, Z = voxel.shape
        if take_last_t:
            voxel = voxel[-1]  # -> [H,W,Z]
        else:
            voxel = voxel.sum(axis=0)  # -> [H,W,Z]
    elif voxel.ndim == 3:
        # [H,W,Z] 或 [Z,H,W]：通过判断哪一维像 Z
        shp = voxel.shape
        # 如果最后一维比较小/离散(如 13)，认为是 [H,W,Z]
        if shp[-1] <= 64 and shp[-1] <= min(shp[0], shp[1]):
            # [H,W,Z]
            pass
        else:
            # 认为是 [Z,H,W]，转成 [H,W,Z]
            voxel = np.transpose(voxel, (1, 2, 0))
    else:
        raise ValueError(f"Unexpected voxel ndim={voxel.ndim}, shape={voxel.shape}")

    # 在 Z 上聚合
    if z_reduce == "sum":
        conf = voxel.sum(axis=-1)   # [H,W]
    elif z_reduce == "max":
        conf = voxel.max(axis=-1)   # [H,W]
    else:
        raise ValueError("z_reduce must be 'sum' or 'max'")

    # 可选：平滑
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            conf = gaussian_filter(conf, sigma=float(smooth_sigma))
        except Exception:
            pass  # 没装 scipy 就跳过

    # 归一化到 [0,1]
    conf = conf.astype(np.float32)
    if norm_mode == "minmax":
        mn, mx = float(conf.min()), float(conf.max())
        if mx > mn:
            conf = (conf - mn) / (mx - mn + 1e-8)
        else:
            conf[:] = 0.0
    elif norm_mode == "percentile":
        lo = np.percentile(conf, p_low)
        hi = np.percentile(conf, p_high)
        if hi > lo:
            conf = np.clip((conf - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        else:
            conf[:] = 0.0
    else:
        raise ValueError("norm_mode must be 'minmax' or 'percentile'")

    return conf  # [H,W], float32 in [0,1]

def heatmap_to_boxes(conf_map,
                     thresh="percentile",  # "percentile" 或 "value"
                     thr_value=0.5,        # 当 thresh="value" 时使用
                     thr_percentile=95.0,  # 当 thresh="percentile" 时使用
                     min_pixels=10):       # 过滤太小的区域
    """
    将置信度图阈值化，做连通域，输出 (N, 4, 2) 的矩形框(像素坐标)和每框分数(平均置信度)。
    """
    import numpy as np
    from scipy.ndimage import label, find_objects

    cm = conf_map
    if thresh == "percentile":
        t = float(np.percentile(cm, thr_percentile))
    elif thresh == "value":
        t = float(thr_value)
    else:
        raise ValueError("thresh must be 'percentile' or 'value'")

    mask = (cm >= t)
    labeled, n = label(mask)
    objs = find_objects(labeled)

    boxes = []
    scores = []
    for s in objs:
        if s is None: continue
        yslice, xslice = s
        h = yslice.stop - yslice.start
        w = xslice.stop - xslice.start
        if h * w < min_pixels:
            continue
        y1, y2 = yslice.start, yslice.stop
        x1, x2 = xslice.start, xslice.stop

        # 矩形四角(像素坐标，左上为(0,0))
        corners = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ], dtype=np.float32)
        boxes.append(corners)

        # 框分数：区域平均置信度
        scores.append(float(cm[y1:y2, x1:x2].mean()))

    if len(boxes) == 0:
        return np.zeros((0,4,2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(boxes, axis=0), np.array(scores, dtype=np.float32)


def overlay_confidence_maps(maps, alphas=None, title="Overlay of 5 vehicle confidence maps", out_path = None):
    """
    Overlay N confidence maps (already aligned in the same global grid)
    on a single figure using transparency.
    
    Args:
        maps: np.ndarray or list of np.ndarray with shape (N,H,W) or list of (H,W).
        alphas: list/array of length N with transparency values in (0,1].
                If None, uses a gentle decreasing sequence.
        title: plot title string.
    """
    # Normalize input to numpy array (N,H,W)
    if isinstance(maps, list):
        maps = np.stack(maps, axis=0)
    assert maps.ndim == 3, f"Expected (N,H,W), got {maps.shape}"
    N, H, W = maps.shape
    
    # Normalize each map to [0,1] to make overlay comparable
    # maps_norm = np.zeros_like(maps, dtype=np.float32)
    # for i in range(N):
    #     m = maps[i].astype(np.float32)
    #     m_min, m_max = m.min(), m.max()
    #     if m_max > m_min:
    #         maps_norm[i] = (m - m_min) / (m_max - m_min)
    #     else:
    #         maps_norm[i] = np.zeros_like(m)
    
    # Alpha schedule: slightly decreasing so earlier layers are more visible
    if alphas is None:
        base, step = 0.7, 0.1
        alphas = np.clip(base - step * np.arange(N), 0.2, 0.9)

    # Plot
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.zeros((H, W)), interpolation="nearest", origin="upper")

    for i in range(N):
        ax.imshow(maps[i], interpolation="nearest", origin="upper", alpha=float(alphas[i]))

    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, dpi=150)
        plt.close()
        #print(f"Saved overlay image to {out_path}")

# # ---- Demo with synthetic data (replace with your actual 5 maps) ----
# N, H, W = 5, 128, 128
# rng = np.random.default_rng(0)
# # Create five smooth-ish maps with different "hot" regions
# demo_maps = []
# centers = [(40,40), (90,35), (64,90), (30,100), (100,80)]
# for cx, cy in centers:
#     X, Y = np.meshgrid(np.arange(W), np.arange(H))
#     dist2 = (X - cx)**2 + (Y - cy)**2
#     blob = np.exp(-dist2 / (2*(15**2))) + 0.15*rng.random((H,W))
#     demo_maps.append(blob)

# overlay_confidence_maps(demo_maps, title="Demo overlay (replace with your 5 global-aligned maps)")

# # Also save one example image
# summed = np.sum(np.stack(demo_maps, axis=0), axis=0)
# summed_norm = summed / np.max(summed)
# plt.imsave('/mnt/data/overlay_confidence_maps_demo.png', summed_norm)



def topk_2d(a: np.ndarray, k: int, *, keepdims=False):
    """
    在二维数组 a 中寻找最大的 k 个元素。
    返回:
      values: (k,) 或 (k,1,1) 的数组（按从大到小排序）
      rows:   (k,) 的行索引
      cols:   (k,) 的列索引
    说明:
      - 使用 argpartition O(n) 选出前 k，再全排序这 k 个。
      - 若含 NaN，默认把 NaN 当作最小（不会进入 Top-K）。
    """

    # 将 NaN 处理为 -inf，避免进入 Top-K
    a_flat = a.reshape(-1)
    safe = np.where(np.isnan(a_flat), -np.inf, a_flat)
    # 先用 argpartition 取出前 k 个（位置无序，但比其余元素都大）
    part_idx = np.argpartition(safe, -k)[-k:]
    # 再对这 k 个进行排序（从大到小）
    order = np.argsort(safe[part_idx])[::-1]
    topk_flat_idx = part_idx[order]

    values = a_flat[topk_flat_idx]
    rows, cols = np.unravel_index(topk_flat_idx, a.shape)

    if keepdims:
        # 可选：保持二维 map 的形状语义（比如做可视化时方便）
        values = values[:, None, None]

    return values, rows, cols
