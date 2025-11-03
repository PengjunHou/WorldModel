import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.det.FastCNNDet import FasterRCNNDetector
from src.det.utils import plot_detections

import numpy as np

class CoordinateTransformer:
    """
    Coordinate Transformer for V2X-Sim 2.0 (CARLA-based)
    -----------------------------------------------------
    • Right-hand coordinate system: x-forward, y-right, z-up
    • Quaternion format: [x, y, z, w]
    • Units: meters
    """

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        Convert quaternion [x, y, z, w] → 3×3 rotation matrix
        """
        x, y, z, w = q
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        return R

    # -----------------------------------------------------

    @staticmethod
    def build_transform_matrix(translation, rotation, inverse=False):
        """
        Build 4×4 homogeneous transform matrix
        Args:
            translation : [x, y, z]
            rotation    : [x, y, z, w]
            inverse     : return inverse matrix if True
        """
        R = CoordinateTransformer.quaternion_to_rotation_matrix(rotation)
        t = np.array(translation).reshape(3, 1)

        T = np.eye(4)
        if inverse:
            R_inv = R.T
            t_inv = -R_inv @ t
            T[:3, :3] = R_inv
            T[:3, 3:] = t_inv
        else:
            T[:3, :3] = R
            T[:3, 3:] = t
        return T

    # -----------------------------------------------------

    @staticmethod
    def transform_point(point, transform_matrix):
        """
        Apply 4×4 transform to a 3-D point.
        """
        p_h = np.append(point, 1.0)
        p_t = transform_matrix @ p_h
        return p_t[:3]

    # -----------------------------------------------------

    @staticmethod
    def compose_transforms(T1, T2):
        """
        Compose two transforms: result = T1 ∘ T2  (apply T2, then T1)
        """
        return T1 @ T2

    # -----------------------------------------------------

    @staticmethod
    def verify_with_official(pose_dict, atol=1e-3):
        """
        Verify transform against the official ego2world matrix
        pose_dict must contain:
            pose_dict['position']           -> [x, y, z]
            pose_dict['rotation']           -> [x, y, z, w]
            pose_dict['transform_matrix']   -> 4×4 numpy array
        """
        T_pred = CoordinateTransformer.build_transform_matrix(
            pose_dict['position'], pose_dict['rotation'])
        T_gt = np.array(pose_dict['transform_matrix'])
        return np.allclose(T_pred, T_gt, atol=atol)


class ImageToBEVProjectorWithGlobal:
    """
    支持全局坐标系的BEV投影器
    """
    
    def __init__(self, camera_params, ego_vehicle_params, global_bev_config, local_bev_config = None):
        """
        参数:
            camera_params: 相机参数（相对于车辆）
                - translation: [x, y, z] 相对于车辆
                - rotation: [x, y, z, w] 相对于车辆
                - camera_intrinsic: 3x3内参矩阵
            
            ego_vehicle_params: 车辆参数（在全局坐标系中）
                - translation: [x, y, z] 全局位置
                - rotation: [x, y, z, w] 全局朝向
            
            global_bev_config: BEV配置
        """
        self.camera_params = camera_params
        self.ego_vehicle_params = ego_vehicle_params
        self.global_bev_config = global_bev_config
        
        self.intrinsic = np.array(camera_params['camera_intrinsic'])
        
        # 构建变换矩阵
        self._build_transform_chain()
        
        # BEV参数
        assert global_bev_config['voxel_size'] == local_bev_config['voxel_size'], "global_bev_config must specify 'voxel_size'"
        self.global_bev_range = global_bev_config['area_extents']
        self.voxel_size = global_bev_config['voxel_size']
        self.global_grid_size = global_bev_config['grid_size']
        self.local_bev_range = local_bev_config['area_extents']
        self.local_grid_size = local_bev_config['grid_size']

    
    def _build_transform_chain(self):
        """
        构建完整的变换链
        
        变换顺序：
        1. T_cam_to_vehicle: 相机 -> 车辆坐标系
        2. T_vehicle_to_global: 车辆 -> 全局坐标系
        3. T_cam_to_global = T_vehicle_to_global @ T_cam_to_vehicle
        """
        # 1. 相机到车辆的变换
        self.T_cam_to_vehicle = CoordinateTransformer.build_transform_matrix(
            self.camera_params['translation'],
            self.camera_params['rotation']
        )
        
        # 2. 车辆到全局的变换
        self.T_vehicle_to_global = CoordinateTransformer.build_transform_matrix(
            self.ego_vehicle_params['translation'],
            self.ego_vehicle_params['rotation']
        )
        
        # 3. 相机到全局的组合变换
        self.T_cam_to_global = CoordinateTransformer.compose_transforms(
            self.T_vehicle_to_global,
            self.T_cam_to_vehicle
        )
        
        # 相机在全局坐标系中的位置
        self.camera_global_position = self.T_cam_to_global[:3, 3]
        
        # print(f"相机在车辆坐标系中的位置: {self.camera_params['translation']}")
        # print(f"车辆在全局坐标系中的位置: {self.ego_vehicle_params['translation']}")
        # print(f"相机在全局坐标系中的位置: {self.camera_global_position}")

    def project_detections_to_bev(self, detections, method='ground_plane', 
                                local_bev_config=None,
                                local_center='vehicle'):
        """
        将检测结果投影到全局BEV和局部BEV（修正版）
        
        注意：此数据集的坐标系定义
        - 全局X: 东西方向
        - 全局Y: 南北方向（主要移动方向）
        - 全局Z: 高度 + 左右偏移
        
        BEV表示:
        - BEV X轴: 对应全局Y轴（前后）
        - BEV Y轴: 对应全局Z轴（左右）
        """
        
        # ========== 1. 投影到全局BEV ==========
        H, W = self.global_grid_size
        bev_confidence_global = np.zeros((H, W))
        bev_coverage_global = np.zeros((H, W))
        projected_positions = []
        
        for det in detections:
            box = det['box']
            score = det['score']
            
            if method == 'ground_plane':
                world_pos_global = self._project_box_to_ground_global(box)
            elif method == 'depth_estimation':
                depth = self._estimate_depth(box, det['label'])
                world_pos_global = self._project_box_with_depth_global(box, depth)
            else:
                continue
            
            if world_pos_global is None:
                continue
            world_pos_global[1] = world_pos_global[2] + world_pos_global[1]
            grid_pos = self._world_to_grid(world_pos_global)
            # print(f"检测 {det['label']} 世界坐标: {world_pos_global} -> BEV网格: {grid_pos}")
            
            if grid_pos is not None:
                gx, gy = grid_pos
                self._fill_bev_region(bev_confidence_global, gx, gy, score, radius=1)
                bev_coverage_global[gy, gx] = 1.0
                
                projected_positions.append({
                    'world_pos': world_pos_global,
                    'grid_pos': (gx, gy),
                    'score': score,
                    'label': det['label']
                })
                # print(f"检测 {det['label']} 投影到全局BEV网格: ({gx}, {gy})，世界坐标: {world_pos_global}, 置信度: {score:.2f}")
        
        # ========== 2. 处理局部BEV配置 ==========
        if local_bev_config is None:
            local_bev_config = {
                'area_extents': [[-32, 32], [-32, 32]],
                'voxel_size': self.voxel_size,
                'grid_size': None
            }
        
        local_area_extents = local_bev_config['area_extents']
        local_voxel_size = local_bev_config.get('voxel_size', self.voxel_size)
        
        range_x = local_area_extents[0][1] - local_area_extents[0][0]
        range_y = local_area_extents[1][1] - local_area_extents[1][0]
        
        calculated_grid_w = int(range_x / local_voxel_size[0])
        calculated_grid_h = int(range_y / local_voxel_size[1])
        
        local_grid_size = (calculated_grid_h, calculated_grid_w)
        local_h, local_w = local_grid_size
        
        local_bev_config_final = {
            'area_extents': local_area_extents,
            'voxel_size': local_voxel_size,
            'grid_size': [local_h, local_w]
        }
        
        # print(f"\n局部BEV配置 (中心: {local_center}):")
        # print(f"  范围: X{local_area_extents[0]}, Y{local_area_extents[1]} 米")
        # print(f"  体素大小: {local_voxel_size}")
        # print(f"  网格大小: {local_h} x {local_w}")
        
        # ========== 3. 确定裁剪中心 ==========
        if local_center == 'camera':
            center_x_global = self.camera_global_position[0]
            center_y_global = self.camera_global_position[1]
            center_name = "Camera"
        else:  # 'vehicle'
            center_x_global = self.ego_vehicle_params['translation'][0]
            center_y_global = self.ego_vehicle_params['translation'][1]
            center_name = "Vehicle"
        
        # print(f"  中心位置 ({center_name}): ({center_x_global:.2f}, {center_y_global:.2f})")
        
        # ========== 4. 从全局BEV裁剪局部BEV ==========
        center_grid_x = int((center_x_global - self.global_bev_range[0][0]) / self.voxel_size[0])
        center_grid_y = int((center_y_global - self.global_bev_range[1][0]) / self.voxel_size[1])
        
        local_x_min, local_x_max = local_area_extents[0]
        local_y_min, local_y_max = local_area_extents[1]
        
        center_local_grid_x = int((0.0 - local_x_min) / local_voxel_size[0])
        center_local_grid_y = int((0.0 - local_y_min) / local_voxel_size[1])
        
        x_start = center_grid_x - center_local_grid_x
        x_end = x_start + local_w
        y_start = center_grid_y - center_local_grid_y
        y_end = y_start + local_h
        
        # print(f"\n裁剪信息:")
        # print(f"  中心全局网格: ({center_grid_x}, {center_grid_y})")
        # print(f"  中心局部网格: ({center_local_grid_x}, {center_local_grid_y})")
        # print(f"  全局裁剪范围: X[{x_start}:{x_end}], Y[{y_start}:{y_end}]")
        
        # 边界处理
        x_start_clip = max(0, x_start)
        x_end_clip = min(W, x_end)
        y_start_clip = max(0, y_start)
        y_end_clip = min(H, y_end)
        
        # 裁剪全局BEV
        bev_confidence_crop = bev_confidence_global[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
        bev_coverage_crop = bev_coverage_global[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
        
        # 创建局部BEV
        bev_confidence_local = np.zeros((local_h, local_w))
        bev_coverage_local = np.zeros((local_h, local_w))
        
        local_x_start = max(0, -x_start)
        local_y_start = max(0, -y_start)
        local_x_end = local_x_start + (x_end_clip - x_start_clip)
        local_y_end = local_y_start + (y_end_clip - y_start_clip)
        
        bev_confidence_local[local_y_start:local_y_end, local_x_start:local_x_end] = bev_confidence_crop
        bev_coverage_local[local_y_start:local_y_end, local_x_start:local_x_end] = bev_coverage_crop
        
        # ========== 5. 筛选局部范围内的检测结果（修正坐标轴映射） ==========
        projected_positions_local = []
        
        # print(f"\n范围内的检测结果:")
        for i, proj in enumerate(projected_positions):
            det_grid_x, det_grid_y = proj['grid_pos']
            
            if x_start <= det_grid_x < x_end and y_start <= det_grid_y < y_end:
                # 网格坐标
                local_det_x = det_grid_x - x_start
                local_det_y = det_grid_y - y_start
                
                # 全局世界坐标
                wx, wy, wz = proj['world_pos'][0], proj['world_pos'][1], proj['world_pos'][2]
                
                # ⭐ 修正：使用正确的坐标轴映射
                # BEV X轴（前后）= 全局Y轴差异
                bev_x = wy - center_y_global
                
                # BEV Y轴（左右）= 全局Z轴（直接使用，因为中心Z=0）
                bev_y = wz
                
                proj_local = proj.copy()
                proj_local['grid_pos_local'] = (local_det_x, local_det_y)
                proj_local['pos_relative_to_center'] = (bev_x, bev_y)  # (前后, 左右)
                proj_local['distance_to_center'] = np.sqrt(bev_x**2 + bev_y**2)
                
                projected_positions_local.append(proj_local)
                
                # print(f"  {i}. {proj['label']}:")
                # print(f"     全局: X={wx:.2f}, Y={wy:.2f}, Z={wz:.2f}")
                # print(f"     BEV: 前后={bev_x:.2f}m, 左右={bev_y:.2f}m")
        
        print(f"\n总计: {len(projected_positions_local)} 个检测在局部BEV范围内")
        
        # ========== 6. 局部BEV信息 ==========
        local_info = {
            'center_type': local_center,
            'center_name': center_name,
            'center_pos_global': (center_x_global, center_y_global),
            'center_grid_pos_local': (center_local_grid_x, center_local_grid_y),
            'center_grid_pos_global': (center_grid_x, center_grid_y),
            'center_pos_meters': (0.0, 0.0),
            'projected_positions_local': projected_positions_local,
            'local_config': local_bev_config_final,
            'crop_info': {
                'x_start': x_start,
                'x_end': x_end,
                'y_start': y_start,
                'y_end': y_end,
                'x_start_clip': x_start_clip,
                'x_end_clip': x_end_clip,
                'y_start_clip': y_start_clip,
                'y_end_clip': y_end_clip
            },
            'coordinate_system_note': 'BEV X=forward/back (global Y), BEV Y=left/right (global Z)'
        }
        
        # 添加车辆和相机的相对位置
        if local_center == 'camera':
            veh_x = self.ego_vehicle_params['translation'][0]
            veh_y = self.ego_vehicle_params['translation'][1]
            veh_z = self.ego_vehicle_params['translation'][2]
            
            # BEV坐标
            veh_bev_x = veh_y - center_y_global
            veh_bev_y = veh_z
            
            local_info['vehicle_pos_relative'] = (veh_bev_x, veh_bev_y)
        else:  # 'vehicle'
            cam_x = self.camera_global_position[0]
            cam_y = self.camera_global_position[1]
            cam_z = self.camera_global_position[2]
            
            # BEV坐标
            cam_bev_x = cam_y - center_y_global
            cam_bev_y = cam_z
            
            local_info['camera_pos_relative'] = (cam_bev_x, cam_bev_y)
        
        return (bev_confidence_global, bev_coverage_global, projected_positions,
                bev_confidence_local, bev_coverage_local, local_info)

    def _project_box_to_ground_global(self, box):
        """
        将检测框投影到全局坐标系的地面
        
        步骤：
        1. 图像坐标 -> 相机坐标系的射线
        2. 射线与地面求交（在全局坐标系中）
        """
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = y2  # 底部
        
        # 1. 图像坐标 -> 归一化相机坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # 相机坐标系中的射线方向
        ray_camera = np.array([x_norm, y_norm, 1.0])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
        # 2. 将射线转换到全局坐标系
        # 射线方向 = 旋转矩阵 @ 相机射线
        R_cam_to_global = self.T_cam_to_global[:3, :3]
        ray_global = R_cam_to_global @ ray_camera
        
        # 相机在全局坐标系中的位置
        camera_pos_global = self.camera_global_position
        
        # 3. 射线与地面 (z=0) 求交
        # 参数方程: P = camera_pos_global + t * ray_global
        # 约束: P[2] = 0 (地面)
        
        if abs(ray_global[2]) < 1e-6:
            return None  # 射线平行于地面
        
        t = (0.0 - camera_pos_global[2]) / ray_global[2]
        
        if t < 0:
            return None  # 交点在相机后方
        
        world_pos_global = camera_pos_global + t * ray_global
        
        return world_pos_global
    
    def _project_box_with_depth_global(self, box, depth):
        """
        使用深度估计投影到全局坐标系
        """
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        
        # 1. 图像坐标 + 深度 -> 相机坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        point_camera = np.array([x_cam, y_cam, z_cam])
        
        # 2. 相机坐标 -> 全局坐标
        point_global = CoordinateTransformer.transform_point(
            point_camera,
            self.T_cam_to_global
        )
        
        return point_global
    
    def _estimate_depth(self, box, label):
        """深度估计（与之前相同）"""
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        
        typical_heights = {
            'car': 1.5,
            'person': 1.7,
            'bus': 3.0,
            'truck': 2.5,
        }
        
        object_height = typical_heights.get(label, 1.5)
        fy = self.intrinsic[1, 1]
        depth = (object_height * fy) / max(box_height, 1)
        
        return np.clip(depth, 1.0, 100.0)
    
    def _world_to_grid(self, world_pos):
        """全局坐标 -> BEV网格索引"""
        if world_pos is None:
            return None
        
        x, y, z = world_pos
        
        gx = int((x - self.global_bev_range[0][0]) / self.voxel_size[0])
        gy = int((y - self.global_bev_range[1][0]) / self.voxel_size[1])
        
        if 0 <= gx < self.global_grid_size[1] and 0 <= gy < self.global_grid_size[0]:
            return (gx, gy)
        else:
            return None
    
    def _fill_bev_region(self, bev_map, gx, gy, value, radius=2):
        """在BEV地图上填充区域"""
        H, W = bev_map.shape
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = gx + dx, gy + dy
                
                if 0 <= nx < W and 0 <= ny < H:
                    dist = np.sqrt(dx**2 + dy**2)
                    weight = np.exp(-dist / radius)
                    bev_map[ny, nx] = max(bev_map[ny, nx], value * weight)

def verify_global_local_consistency_corrected(projected_positions, local_info):
    """
    验证全局BEV和局部BEV中物体之间的距离是否一致（修正版）
    """
    print("\n" + "="*80)
    print("全局BEV vs 局部BEV 距离一致性检查（修正版）")
    print("="*80)
    
    local_projs = local_info['projected_positions_local']
    
    if len(local_projs) < 2:
        print("局部BEV中物体数量少于2个，无法进行距离验证")
        return []
    
    print(f"\n检测到 {len(local_projs)} 个物体在局部BEV范围内")
    print(f"坐标系说明: {local_info.get('coordinate_system_note', 'N/A')}")
    print("\n物体两两之间的距离比较:")
    print("-"*80)
    
    errors = []
    
    for i in range(len(local_projs)):
        for j in range(i+1, len(local_projs)):
            label1 = local_projs[i]['label']
            label2 = local_projs[j]['label']
            
            # ===== 全局坐标中的距离 =====
            w1_x, w1_y, w1_z = local_projs[i]['world_pos']
            w2_x, w2_y, w2_z = local_projs[j]['world_pos']
            
            # 在Y-Z平面（BEV平面）的距离
            dist_y_global = w2_y - w1_y  # 前后
            dist_z_global = w2_z - w1_z  # 左右
            dist_global_bev = np.sqrt(dist_y_global**2 + dist_z_global**2)
            
            # ===== 局部BEV坐标中的距离 =====
            l1_x, l1_y = local_projs[i]['pos_relative_to_center']  # (前后, 左右)
            l2_x, l2_y = local_projs[j]['pos_relative_to_center']
            
            dist_x_local = l2_x - l1_x  # 前后
            dist_y_local = l2_y - l1_y  # 左右
            dist_local_bev = np.sqrt(dist_x_local**2 + dist_y_local**2)
            
            # ===== 比较 =====
            diff_forward = abs(dist_y_global - dist_x_local)  # 前后方向
            diff_lateral = abs(dist_z_global - dist_y_local)  # 左右方向
            diff_total = abs(dist_global_bev - dist_local_bev)
            
            print(f"\n物体对 #{i} ({label1}) ↔ #{j} ({label2}):")
            print(f"  全局坐标 (Y-Z平面):")
            print(f"    前后(Y): {dist_y_global:.3f}m")
            print(f"    左右(Z): {dist_z_global:.3f}m")
            print(f"    总距离: {dist_global_bev:.3f}m")
            
            print(f"  局部BEV坐标:")
            print(f"    前后(X): {dist_x_local:.3f}m")
            print(f"    左右(Y): {dist_y_local:.3f}m")
            print(f"    总距离: {dist_local_bev:.3f}m")
            
            print(f"  差异:")
            print(f"    前后方向: {diff_forward:.4f}m")
            print(f"    左右方向: {diff_lateral:.4f}m")
            print(f"    总距离: {diff_total:.4f}m")
            
            threshold = 0.01
            if diff_forward > threshold or diff_lateral > threshold:
                print(f"    ❌ 差异过大！")
                errors.append({
                    'pair': (i, j),
                    'labels': (label1, label2),
                    'diff_forward': diff_forward,
                    'diff_lateral': diff_lateral,
                    'diff_total': diff_total
                })
            else:
                print(f"    ✓ 一致")
    
    print("\n" + "="*80)
    if len(errors) > 0:
        print(f"❌ 发现 {len(errors)} 对物体的距离不一致！")
    else:
        print("✓ 所有物体间的距离完全一致！")
    print("="*80)
    
    return errors    


if __name__ == "__main__":
    # =============== 1. 配置 ===============
    camera_params = {
        "translation": [
            -2.0,
            -0.0,
            1.3787471055984497
        ],
        "rotation": [
            0.5,
            -0.5,
            -0.5,
            0.5
        ],
        "camera_intrinsic": [
            [
                560.1660305677678,
                0.0,
                800.0
            ],
            [
                0.0,
                560.1660305677678,
                450.0
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ]
    }

    ego_vehicle_params = {
        "rotation": [
            0.6978317588164028,
            0.0,
            -0.0,
            -0.7162617094241502
        ],
        "translation": [
            -3.5915331840515137,
            157.4496307373047,
            0.0
        ]
    }

    global_bev_config = {
        'area_extents': [[-256, 256], [-256, 256]],  # 覆盖64m x 64m区域
        'voxel_size': [0.25, 0.25],              # 每个网格0.25m
        'grid_size': [2048, 2048]                  # 256x256网格
    }
    local_bev_config = {
        'area_extents': [[-32, 32], [-32, 32]],  
        'voxel_size': [0.25, 0.25],              
        'grid_size': [256, 256]                  
    }
    print("BEV配置:", global_bev_config)
    # =============== 2. 创建投影器 ===============
    projector = ImageToBEVProjectorWithGlobal(camera_params, ego_vehicle_params, global_bev_config, local_bev_config)


    # =============== 3. 目标检测 ===============
    detector = FasterRCNNDetector(device='cuda', conf_threshold=0.2)

    import cv2
    img_file = '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_BACK_id_1/scene_5_000006.jpg'
    image = cv2.imread(img_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect(image)

    # =============== 4. 投影到BEV ===============
    # bev_confidence, bev_coverage, projected_positions = projector.project_detections_to_bev(
    #     detections,
    #     method=  'depth_estimation' # 或 'ground_plane', 'depth_estimation', 'multi_height'
    # )

    bev_confidence_global, bev_coverage_global, projected_positions,\
                bev_confidence_local, bev_coverage_local, local_info = projector.project_detections_to_bev(
        detections,
        method='depth_estimation',
        local_bev_config=local_bev_config,
        local_center='vehicle'  # 或 'camera'
    )

    # =============== 5. 可视化 ===============
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像（模拟）
    axes[0].set_title('Image Detections')
    axes[0].imshow(plot_detections(image_rgb, detections))
    axes[0].set_xlim(0, 1600)
    axes[0].set_ylim(900, 0)
    # for det in detections:
    #     x1, y1, x2, y2 = det['box']
    #     rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
    #                          fill=False, color='red', linewidth=2)
    #     axes[0].add_patch(rect)
    #     axes[0].text(x1, y1-10, f"{det['label']}: {det['score']:.2f}", 
    #                 color='red', fontsize=10)

    print(f"{bev_confidence_local.size} {bev_confidence_global.size}")
    print(f"number of greater than zero in local: {np.sum(bev_confidence_local>0)}")

    # BEV置信度地图
    im1 = axes[1].imshow(bev_confidence_global, cmap='hot', origin='lower', 
                        extent=[global_bev_config['area_extents'][0][0], 
                                global_bev_config['area_extents'][0][1],
                                global_bev_config['area_extents'][1][0], 
                                global_bev_config['area_extents'][1][1]])
    axes[1].set_title('BEV Confidence Map')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    plt.colorbar(im1, ax=axes[1])

    # 标记相机位置
    # cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
    cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]  # 全局坐标系中位置
    axes[1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
    axes[1].legend()

    # Local BEV覆盖地图
    # im2 = axes[2].imshow(bev_confidence_local, cmap='hot', origin='lower', 
    #             extent=[local_bev_config['area_extents'][0][0], 
    #                     local_bev_config['area_extents'][0][1],
    #                     local_bev_config['area_extents'][1][0], 
    #                     local_bev_config['area_extents'][1][1]])
    # axes[2].set_title('Local BEV Confidence Map')
    # axes[2].set_xlabel('X (meters)')
    # axes[2].set_ylabel('Y (meters)')
    # plt.colorbar(im2, ax=axes[2])

    # # 标记相机位置
    # cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1] # 车辆坐标系中位置
    # # cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]  # 全局坐标系中位置
    # axes[2].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
    # axes[2].legend()

    # # BEV覆盖地图
    axes[2].imshow(bev_coverage_global, cmap='Blues', origin='lower',
                extent=[global_bev_config['area_extents'][0][0], 
                        global_bev_config['area_extents'][0][1],
                        global_bev_config['area_extents'][1][0], 
                        global_bev_config['area_extents'][1][1]])
    axes[2].set_title('BEV Coverage Map')
    axes[2].set_xlabel('X (meters)')
    axes[2].set_ylabel('Y (meters)')

    # 标注投影位置
    for proj in projected_positions:
        wx, wy = proj['world_pos'][0], proj['world_pos'][1]
        axes[2].plot(wx, wy, 'ro', markersize=8)
        axes[2].text(wx+0.5, wy+0.5, proj['label'], fontsize=8)

    plt.tight_layout()
    plt.savefig('bev_projection_result.png', dpi=150)
    plt.show()

    # =============== 6. 打印投影结果 ===============
    cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]
    print("投影结果:")
    print("-" * 60)
    for i, proj in enumerate(projected_positions):
        wx, wy, wz = proj['world_pos']
        gx, gy = proj['grid_pos']
        print(f"{i+1}. {proj['label']} (置信度: {proj['score']:.2f})")
        print(f"   世界坐标: ({wx:.2f}, {wy:.2f}, {wz:.2f}) 米")
        print(f"   网格坐标: ({gx}, {gy})")
        print(f"   距离相机: {np.linalg.norm([wx-cam_x, wy-cam_y]):.2f} 米")
        print()

    # verify_global_local_consistency_corrected(projected_positions, local_info)