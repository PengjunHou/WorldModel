import numpy as np
import cv2
import matplotlib.pyplot as plt
from .FastCNNDet import FasterRCNNDetector
from .utils import plot_detections


class CoordinateTransformer:
    """
    Coordinate Transformer for V2X-Sim 2.0 (CARLA-based)
    -----------------------------------------------------
    • Right-hand coordinate system: x-forward, y-right, z-up
    • Quaternion format: [w, x, y, z]  # ⭐ 修正：V2X-Sim使用 [w,x,y,z]
    • Units: meters
    """

    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        Convert quaternion [w, x, y, z] → 3×3 rotation matrix
        """
        w, x, y, z = q  # ⭐ 修正：从 [x,y,z,w] 改为 [w,x,y,z]
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
            rotation    : [w, x, y, z]  # ⭐ 修正注释
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
            pose_dict['rotation']           -> [w, x, y, z]  # ⭐ 修正注释
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
    
    def __init__(self, camera_params, ego_vehicle_params, global_bev_config, local_bev_config=None):
        """
        参数:
            camera_params: 相机参数（相对于车辆）
                - translation: [x, y, z] 相对于车辆
                - rotation: [w, x, y, z] 相对于车辆  # ⭐ 修正注释
                - camera_intrinsic: 3x3内参矩阵
            
            ego_vehicle_params: 车辆参数（在全局坐标系中）
                - translation: [x, y, z] 全局位置
                - rotation: [w, x, y, z] 全局朝向  # ⭐ 修正注释
            
            global_bev_config: BEV配置
        """
        self.camera_params = camera_params
        self.ego_vehicle_params = ego_vehicle_params
        self.global_bev_config = global_bev_config
        
        self.intrinsic = np.array(camera_params['camera_intrinsic'])
        
        # 构建变换矩阵
        self._build_transform_chain()
        
        # BEV参数
        if local_bev_config is not None:
            assert global_bev_config['voxel_size'] == local_bev_config['voxel_size'], \
                "global_bev_config and local_bev_config must have same 'voxel_size'"
        
        self.global_bev_range = global_bev_config['area_extents']
        self.voxel_size = global_bev_config['voxel_size']
        self.global_grid_size = global_bev_config['grid_size']
        
        if local_bev_config is not None:
            self.local_bev_range = local_bev_config['area_extents']
            self.local_grid_size = local_bev_config['grid_size']

    
    def _build_transform_chain(self):
        """
        构建完整的变换链（V2X-Sim版本）
        
        变换顺序：
        1. T_cam_to_vehicle: 相机 -> 车辆坐标系
        2. T_vehicle_to_global: 车辆 -> 全局坐标系
        3. T_cam_to_global = T_vehicle_to_global @ T_cam_to_vehicle
        
        注意：V2X-Sim的相机外参给的是vehicle_to_camera，需要取逆
        """
        # ⭐ 1. 相机到车辆的变换（不需要取逆）
        self.T_cam_to_vehicle = CoordinateTransformer.build_transform_matrix(
            self.camera_params['translation'],
            self.camera_params['rotation'],
            inverse=False  
        )
        
        # 2. 车辆到全局的变换
        self.T_vehicle_to_global = CoordinateTransformer.build_transform_matrix(
            self.ego_vehicle_params['translation'],
            self.ego_vehicle_params['rotation'],
            inverse=False
        )
        
        # 3. 相机到全局的组合变换
        self.T_cam_to_global = CoordinateTransformer.compose_transforms(
            self.T_vehicle_to_global,
            self.T_cam_to_vehicle
        )
        
        # 相机在全局坐标系中的位置
        self.camera_global_position = self.T_cam_to_global[:3, 3]

    def project_detections_to_bev(self, detections, method='ground_plane', 
                                local_bev_config=None,
                                local_center='vehicle'):
        """
        将检测结果投影到全局BEV和局部BEV
        
        坐标系定义：
        - 全局X: 前方（车辆主要移动方向）
        - 全局Y: 右方
        - 全局Z: 上方（高度）
        
        BEV表示:
        - BEV X轴: 对应全局X轴（前后）
        - BEV Y轴: 对应全局Y轴（左右）
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
            
            print(f"Detected object at global position: {world_pos_global}")
            grid_pos = self._world_to_grid(world_pos_global)
            
            if grid_pos is not None:
                gx, gy = grid_pos
                self._fill_bev_region(bev_confidence_global, gx, gy, score, radius=0)
                bev_coverage_global[gy, gx] = 1.0
                
                projected_positions.append({
                    'world_pos': world_pos_global,
                    'grid_pos': (gx, gy),
                    'score': score,
                    'label': det['label']
                })
        
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
        
        # ========== 3. 确定裁剪中心 ==========
        if local_center == 'camera':
            center_x_global = self.camera_global_position[0]
            center_y_global = self.camera_global_position[1]
            center_name = "Camera"
        else:  # 'vehicle'
            center_x_global = self.ego_vehicle_params['translation'][0]
            center_y_global = self.ego_vehicle_params['translation'][1]
            center_name = "Vehicle"
        
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
        
        # 边界处理
        x_start_clip = max(0, x_start)
        x_end_clip = min(W, x_end)
        y_start_clip = max(0, y_start)
        y_end_clip = min(H, y_end)
        
        # 裁剪全局BEV
        bev_confidence_crop = bev_confidence_global[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
        bev_coverage_crop = bev_coverage_global[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
        # print(f"x_start: {x_start}, x_end: {x_end}, y_start: {y_start}, y_end: {y_end}")
        # print(f"x_start_clip: {x_start_clip}, x_end_clip: {x_end_clip}, y_start_clip: {y_start_clip}, y_end_clip: {y_end_clip}")
        # 创建局部BEV
        bev_confidence_local = np.zeros((local_h, local_w))
        bev_coverage_local = np.zeros((local_h, local_w))
        
        local_x_start = max(0, -x_start)
        local_y_start = max(0, -y_start)
        local_x_end = local_x_start + (x_end_clip - x_start_clip)
        local_y_end = local_y_start + (y_end_clip - y_start_clip)
        
        print(f"local y: {local_y_start} to {local_y_end}, local x: {local_x_start} to {local_x_end}")
        print(f"bev confidence local shape: {bev_confidence_local.shape}, crop shape: {bev_confidence_crop.shape}")
        
        bev_confidence_local[local_y_start:local_y_end, local_x_start:local_x_end] = bev_confidence_crop
        bev_coverage_local[local_y_start:local_y_end, local_x_start:local_x_end] = bev_coverage_crop
        
        # ========== 5. 筛选局部范围内的检测结果 ==========
        projected_positions_local = []
        
        for i, proj in enumerate(projected_positions):
            det_grid_x, det_grid_y = proj['grid_pos']
            
            if x_start <= det_grid_x < x_end and y_start <= det_grid_y < y_end:
                # 局部网格坐标
                local_det_x = det_grid_x - x_start
                local_det_y = det_grid_y - y_start
                
                # 全局世界坐标
                wx, wy, wz = proj['world_pos'][0], proj['world_pos'][1], proj['world_pos'][2]
                
                # BEV相对坐标（相对于中心）
                bev_x = wx - center_x_global  # X方向（前后）
                bev_y = wy - center_y_global  # Y方向（左右）
                
                proj_local = proj.copy()
                proj_local['grid_pos_local'] = (local_det_x, local_det_y)
                proj_local['pos_relative_to_center'] = (bev_x, bev_y)
                proj_local['distance_to_center'] = np.sqrt(bev_x**2 + bev_y**2)
                
                projected_positions_local.append(proj_local)
        
        print(f"ego vehicle global pos: {self.ego_vehicle_params['translation']}")
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
            'coordinate_system_note': 'BEV X=forward/back, BEV Y=left/right'
        }
        
        # 添加车辆和相机的相对位置
        if local_center == 'camera':
            veh_x = self.ego_vehicle_params['translation'][0]
            veh_y = self.ego_vehicle_params['translation'][1]
            veh_z = self.ego_vehicle_params['translation'][2]
            
            veh_bev_x = veh_x - center_x_global
            veh_bev_y = veh_y - center_y_global
            
            local_info['vehicle_pos_relative'] = (veh_bev_x, veh_bev_y)
        else:  # 'vehicle'
            cam_x = self.camera_global_position[0]
            cam_y = self.camera_global_position[1]
            cam_z = self.camera_global_position[2]
            
            cam_bev_x = cam_x - center_x_global
            cam_bev_y = cam_y - center_y_global
            
            local_info['camera_pos_relative'] = (cam_bev_x, cam_bev_y)
        
        return (bev_confidence_global, bev_coverage_global, projected_positions,
                bev_confidence_local, bev_coverage_local, local_info)

    def verify_left_right_symmetry(self):
        """验证左右对称性"""
        
        print("\n" + "="*80)
        print("验证左右对称性")
        print("="*80)
        
        # 测试两个点：图像左侧和右侧
        test_boxes = [
            ([200, 400, 300, 500], "图像左侧"),
            ([1300, 400, 1400, 500], "图像右侧")
        ]
        
        for box, label in test_boxes:
            point = self._project_box_with_depth_global(box, 20.0)
            
            # 转换为相对车辆的坐标
            veh_x = self.ego_vehicle_params['translation'][0]
            veh_y = self.ego_vehicle_params['translation'][1]
            
            rel_x = point[0] - veh_x  # 前后
            rel_y = point[1] - veh_y  # 左右
            
            print(f"\n{label}:")
            print(f"  图像坐标: u={(box[0]+box[2])/2:.0f}")
            print(f"  相对车辆: X={rel_x:.2f}m (前后), Y={rel_y:.2f}m (左右)")
            print(f"  应该: {'Y<0 (左侧)' if '左' in label else 'Y>0 (右侧)'}")
            print(f"  实际: {'✓' if (rel_y < 0 and '左' in label) or (rel_y > 0 and '右' in label) else '✗'}")

    def _project_box_to_ground_global(self, box):
        """
        将检测框投影到全局坐标系的地面
        """
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = y2  # 底部
        
        # 1. 图像坐标 -> 归一化相机坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # ⭐ 修正：X轴取反
        x_norm = -x_norm
        
        # 相机坐标系中的射线方向
        ray_camera = np.array([x_norm, y_norm, 1.0])
        ray_camera = ray_camera / np.linalg.norm(ray_camera)
        
        # 2. 将射线转换到全局坐标系
        R_cam_to_global = self.T_cam_to_global[:3, :3]
        ray_global = R_cam_to_global @ ray_camera
        
        # 相机在全局坐标系中的位置
        camera_pos_global = self.camera_global_position
        
        # 3. 射线与地面 (z=0) 求交
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
        
        # 1. 图像坐标 + 深度 -> 归一化坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # 2. CARLA相机坐标系转换
        # 相机坐标系: X-right, Y-down, Z-forward
        # 图像坐标u对应相机X(右), v对应相机Y(下), depth对应Z(前)
        # ⭐ 关键修正：完整的坐标系映射
        x_cam = x_norm * depth      # 右方
        y_cam = y_norm * depth      # 下方  
        z_cam = depth               # 前方
        
        point_camera = np.array([x_cam, y_cam, z_cam])
        
        # 3. 相机坐标 -> 全局坐标 (变换矩阵已经包含了坐标系转换)
        point_global = CoordinateTransformer.transform_point(
            point_camera,
            self.T_cam_to_global
        )
        
        return point_global
    
    def _estimate_depth(self, box, label):
        """深度估计"""
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


    def test_all_flip_combinations(self):
        """测试所有可能的坐标取反组合"""
        
        # 选择第一个检测框
        test_box = [551, 410, 1045, 802]  # 从你的输出中
        test_depth = 20.0
        
        x1, y1, x2, y2 = test_box
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        
        # 计算相机坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_cam = (u - cx) * test_depth / fx
        y_cam = (v - cy) * test_depth / fy
        z_cam = test_depth
        
        # print("\n" + "="*80)
        # print("测试所有坐标取反组合")
        # print("="*80)
        # print(f"\n相机坐标: x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}")
        
        # 测试所有可能的取反组合
        flip_combinations = [
            ("不取反",      [1, 1, 1]),
            ("翻转X",       [-1, 1, 1]),
            ("翻转Y",       [1, -1, 1]),
            ("翻转Z",       [1, 1, -1]),
            ("翻转X+Y",     [-1, -1, 1]),
            ("翻转X+Z",     [-1, 1, -1]),
            ("翻转Y+Z",     [1, -1, -1]),
            ("翻转X+Y+Z",   [-1, -1, -1]),
        ]
        
        # print(f"\n车辆全局位置: {self.ego_vehicle_params['translation']}")
        # print(f"车辆Z坐标: {self.ego_vehicle_params['translation'][2]:.2f}m\n")
        
        results = []
        
        for name, flip in flip_combinations:
            # 应用翻转到相机坐标
            point_camera = np.array([
                x_cam * flip[0],
                y_cam * flip[1],
                z_cam * flip[2]
            ])
            
            # 转换到全局坐标
            point_global = CoordinateTransformer.transform_point(
                point_camera,
                self.T_cam_to_global
            )
            
            # 评估结果
            z_val = point_global[2]
            z_ok = 0 <= z_val <= 5  # Z应该在0-5米范围
            
            # 距离车辆
            veh_pos = self.ego_vehicle_params['translation']
            dist = np.sqrt((point_global[0] - veh_pos[0])**2 + 
                        (point_global[1] - veh_pos[1])**2)
            
            results.append({
                'name': name,
                'flip': flip,
                'point': point_global,
                'z_ok': z_ok,
                'dist': dist
            })
            
            status = "✓" if z_ok else "✗"
            print(f"{name:15s}: X={point_global[0]:7.2f}m, "
                f"Y={point_global[1]:7.2f}m, "
                f"Z={point_global[2]:7.2f}m  "
                f"距离={dist:6.2f}m  {status}")
        
        # 找出最合理的组合
        print("\n" + "="*80)
        print("推荐的翻转组合:")
        print("="*80)
        
        valid_results = [r for r in results if r['z_ok']]
        if valid_results:
            for r in valid_results:
                print(f"✓ {r['name']:15s}: Z={r['point'][2]:.2f}m 在合理范围内")
                print(f"   翻转: X{'×(-1)' if r['flip'][0]==-1 else ''}, "
                    f"Y{'×(-1)' if r['flip'][1]==-1 else ''}, "
                    f"Z{'×(-1)' if r['flip'][2]==-1 else ''}")
        else:
            print("⚠️ 没有找到Z坐标在合理范围内的组合")
            print("可能需要检查深度估计或坐标系定义")
        
        print("="*80)

def verify_global_local_consistency_corrected(projected_positions, local_info):
    """
    验证全局BEV和局部BEV中物体之间的距离是否一致
    """
    print("\n" + "="*80)
    print("全局BEV vs 局部BEV 距离一致性检查")
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
            
            # BEV平面距离（X-Y平面）
            dist_x_global = w2_x - w1_x
            dist_y_global = w2_y - w1_y
            dist_global_bev = np.sqrt(dist_x_global**2 + dist_y_global**2)
            
            # ===== 局部BEV坐标中的距离 =====
            l1_x, l1_y = local_projs[i]['pos_relative_to_center']
            l2_x, l2_y = local_projs[j]['pos_relative_to_center']
            
            dist_x_local = l2_x - l1_x
            dist_y_local = l2_y - l1_y
            dist_local_bev = np.sqrt(dist_x_local**2 + dist_y_local**2)
            
            # ===== 比较 =====
            diff_x = abs(dist_x_global - dist_x_local)
            diff_y = abs(dist_y_global - dist_y_local)
            diff_total = abs(dist_global_bev - dist_local_bev)
            
            print(f"\n物体对 #{i} ({label1}) ↔ #{j} ({label2}):")
            print(f"  全局坐标 (X-Y平面):")
            print(f"    ΔX: {dist_x_global:.3f}m")
            print(f"    ΔY: {dist_y_global:.3f}m")
            print(f"    总距离: {dist_global_bev:.3f}m")
            
            print(f"  局部BEV坐标:")
            print(f"    ΔX: {dist_x_local:.3f}m")
            print(f"    ΔY: {dist_y_local:.3f}m")
            print(f"    总距离: {dist_local_bev:.3f}m")
            
            print(f"  差异:")
            print(f"    X方向: {diff_x:.4f}m")
            print(f"    Y方向: {diff_y:.4f}m")
            print(f"    总距离: {diff_total:.4f}m")
            
            threshold = 0.01
            if diff_x > threshold or diff_y > threshold:
                print(f"    ❌ 差异过大！")
                errors.append({
                    'pair': (i, j),
                    'labels': (label1, label2),
                    'diff_x': diff_x,
                    'diff_y': diff_y,
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
        'area_extents': [[-256, 256], [-256, 256]],
        'voxel_size': [0.25, 0.25],
        'grid_size': [2048, 2048]
    }
    
    local_bev_config = {
        'area_extents': [[-32, 32], [-32, 32]],  
        'voxel_size': [0.25, 0.25],              
        'grid_size': [256, 256]                  
    }
    
    print("="*80)
    print("V2X-Sim BEV投影系统")
    print("="*80)
    print(f"全局BEV配置: {global_bev_config['grid_size']}")
    print(f"局部BEV配置: {local_bev_config['grid_size']}")
    
    # =============== 2. 创建投影器 ===============
    projector = ImageToBEVProjectorWithGlobal(
        camera_params, 
        ego_vehicle_params, 
        global_bev_config, 
        local_bev_config
    )

    # ⭐ 验证坐标变换
    print("\n" + "="*80)
    print("验证坐标变换")
    print("="*80)
    
    # 测试相机旋转矩阵
    print(f"\n相机旋转四元数: {camera_params['rotation']}")
    R_cam = CoordinateTransformer.quaternion_to_rotation_matrix(camera_params['rotation'])
    print("\n相机旋转矩阵:")
    print(R_cam)
    print("\n相机坐标轴在车辆坐标系中:")
    print(f"  相机X(右) → {R_cam[:, 0]}")
    print(f"  相机Y(下) → {R_cam[:, 1]}")
    print(f"  相机Z(前) → {R_cam[:, 2]}")
    
    # 检查相机朝向
    if R_cam[0, 2] > 0.8:
        print("\n✓ 相机朝向正确（朝前）")
    elif R_cam[0, 2] < -0.8:
        print("\n⚠️ 相机朝向后方")
    else:
        print(f"\n⚠️ 相机朝向异常: Z轴在车辆X方向的分量 = {R_cam[0, 2]:.3f}")
    
    # 验证车辆Z坐标
    veh_z = ego_vehicle_params['translation'][2]
    print(f"\n车辆Z坐标: {veh_z:.2f}m")
    if abs(veh_z) < 2:
        print("✓ Z值合理（接近地面）")
    
    print("\n相机全局位置:", projector.camera_global_position)
    print("="*80)

    # =============== 3. 目标检测 ===============
    detector = FasterRCNNDetector(device='cuda', conf_threshold=0.2)

    img_file = '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_BACK_id_1/scene_5_000006.jpg'
    image = cv2.imread(img_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect(image)

    # projector.test_all_flip_combinations()
    
    print(f"\n检测到 {len(detections)} 个物体")

    # =============== 4. 投影到BEV ===============
    bev_confidence_global, bev_coverage_global, projected_positions,\
                bev_confidence_local, bev_coverage_local, local_info = projector.project_detections_to_bev(
        detections,
        method='depth_estimation',
        local_bev_config=local_bev_config,
        local_center='vehicle'
    )

    # ⭐ 验证投影结果
    print("\n" + "="*80)
    print("检查投影结果的坐标")
    print("="*80)
    
    for i, proj in enumerate(projected_positions[:5]):  # 只看前5个
        wx, wy, wz = proj['world_pos']
        print(f"\n物体 {i+1}: {proj['label']} (置信度: {proj['score']:.2f})")
        print(f"  X={wx:.2f}m")
        print(f"  Y={wy:.2f}m")
        print(f"  Z={wz:.2f}m")
        
        # Z应该在合理范围
        if abs(wz) < 5:
            print(f"  ✓ Z值合理（高度范围）")
        else:
            print(f"  ⚠️ Z值={wz:.2f}m 异常！应该在0-5米范围")
        
        # 计算距离
        veh_pos = ego_vehicle_params['translation']
        dist = np.sqrt((wx - veh_pos[0])**2 + (wy - veh_pos[1])**2)
        print(f"  距离车辆: {dist:.2f}m")

    # =============== 5. 可视化 ===============
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像
    axes[0].set_title('Image Detections')
    axes[0].imshow(plot_detections(image_rgb, detections))
    axes[0].set_xlim(0, 1600)
    axes[0].set_ylim(900, 0)

    print(f"\nBEV地图统计:")
    print(f"  局部BEV大小: {bev_confidence_local.shape}")
    print(f"  全局BEV大小: {bev_confidence_global.shape}")
    print(f"  局部非零元素: {np.sum(bev_confidence_local>0)}")
    print(f"  全局非零元素: {np.sum(bev_confidence_global>0)}")

    # 全局BEV置信度地图
    im1 = axes[1].imshow(bev_confidence_global, cmap='hot', origin='lower', 
                        extent=[global_bev_config['area_extents'][0][0], 
                                global_bev_config['area_extents'][0][1],
                                global_bev_config['area_extents'][1][0], 
                                global_bev_config['area_extents'][1][1]])
    axes[1].set_title('Global BEV Confidence Map')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    plt.colorbar(im1, ax=axes[1])

    # 标记相机和车辆位置
    cam_x, cam_y = projector.camera_global_position[0], projector.camera_global_position[1]
    veh_x, veh_y = ego_vehicle_params['translation'][0], ego_vehicle_params['translation'][1]
    axes[1].plot(cam_x, cam_y, 'b*', markersize=15, label='Camera', markeredgecolor='white', markeredgewidth=2)
    axes[1].plot(veh_x, veh_y, 'g^', markersize=15, label='Vehicle', markeredgecolor='white', markeredgewidth=2)
    axes[1].legend()

    # 全局BEV覆盖地图
    axes[2].imshow(bev_coverage_global, cmap='Blues', origin='lower',
                extent=[global_bev_config['area_extents'][0][0], 
                        global_bev_config['area_extents'][0][1],
                        global_bev_config['area_extents'][1][0], 
                        global_bev_config['area_extents'][1][1]])
    axes[2].set_title('Global BEV Coverage Map')
    axes[2].set_xlabel('X (meters)')
    axes[2].set_ylabel('Y (meters)')

    # 标注投影位置
    for proj in projected_positions:
        wx, wy = proj['world_pos'][0], proj['world_pos'][1]
        axes[2].plot(wx, wy, 'ro', markersize=8)
        axes[2].text(wx+0.5, wy+0.5, proj['label'], fontsize=8, color='white',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))

    plt.tight_layout()
    plt.savefig('bev_projection_result.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化已保存: bev_projection_result.png")
    # plt.show()

    # =============== 6. 打印投影结果 ===============
    print("\n" + "="*80)
    print("投影结果汇总")
    print("="*80)
    
    for i, proj in enumerate(projected_positions):
        wx, wy, wz = proj['world_pos']
        gx, gy = proj['grid_pos']
        dist = np.sqrt((wx - veh_x)**2 + (wy - veh_y)**2)
        
        print(f"\n{i+1}. {proj['label']} (置信度: {proj['score']:.2f})")
        print(f"   世界坐标: X={wx:.2f}m, Y={wy:.2f}m, Z={wz:.2f}m")
        print(f"   网格坐标: ({gx}, {gy})")
        print(f"   距离车辆: {dist:.2f}m")

    # =============== 7. 验证距离一致性 ===============
    # if len(local_info['projected_positions_local']) >= 2:
    #     verify_global_local_consistency_corrected(projected_positions, local_info)

    projector.verify_left_right_symmetry()