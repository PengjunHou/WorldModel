
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
        print(f"Loaded detection model from {ckpt_path}")

    fafmodule.model.eval()
    return fafmodule, device

import torch

@torch.no_grad()
def run_detection(fafmodule, device, config, 
                  padded_voxel_points, reg_target, reg_loss_mask,
                  anchors_map, vis_maps, trans_matrix,
                  target_agent_id, num_agent, flag="lowerbound"):
    """
    对单个 agent 的点云 BEV 输入进行目标检测，输出 scores & boxes
    参考 test_codet.py 的推理流程
    """    # 构造 data，与 test_codet.py 保持一致
    data = {
        "bev_seq": padded_voxel_points.unsqueeze(0).to(device),
        "labels": torch.zeros_like(reg_target).unsqueeze(0).to(device),  # dummy
        "reg_targets": reg_target.unsqueeze(0).to(device),
        "anchors": anchors_map.unsqueeze(0).to(device),
        "vis_maps": vis_maps.unsqueeze(0).to(device),
        "reg_loss_mask": reg_loss_mask.unsqueeze(0).to(device).type(dtype=torch.bool),
        "target_agent_ids": torch.tensor([[target_agent_id]]).to(device),
        "num_agent": torch.tensor([[num_agent]]).to(device),
        "trans_matrices": trans_matrix.unsqueeze(0).to(device),
    }

    # # 调用推理
    # if flag == "lowerbound_box_com":
    #     loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
    #         data, data["trans_matrices"], validation=False
    #     )
    # else:
    #     result = fafmodule.predict_all(
    #         data, 1, validation=False, num_agent = 1
    #     )

    # LOG.info(f"Detection result for agent {target_agent_id}: {result}")
    # # 解析结果
    # scores, boxes = [], []

    # if isinstance(result, list) and len(result) > 0:
    #     # result[0] 是一个 tuple (detections, some_tensor)
    #     detections, _ = result[0]

    #     if isinstance(detections, list) and len(detections) > 0:
    #         agent_detections = detections[0]  # [[{...}]]
    #         if len(agent_detections) > 0:
    #             pred_dict = agent_detections[0]  # {'pred': ..., 'score': ..., ...}

    #             if "score" in pred_dict and "pred" in pred_dict:
    #                 scores = pred_dict["score"]
    #                 if torch.is_tensor(scores):
    #                     scores = scores.detach().cpu().numpy()
    #                 boxes = pred_dict["pred"]
    #                 if torch.is_tensor(boxes):
    #                     boxes = boxes.detach().cpu().numpy()

    scores = np.random.rand(10)  # 10个检测框的置信度
    boxes = np.random.rand(10, 4) * 30  # 10个检测框的(x, y, w, h)，范围在0-10米之间


    return scores, boxes



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



