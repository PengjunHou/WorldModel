import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.det.FastCNNDet import FasterRCNNDetector
from src.det.utils import plot_detections

class ImageToBEVProjector:
    """
    将图像检测结果投影到BEV网格
    """
    def __init__(self, camera_params, bev_config):
        """
        参数:
            camera_params: 相机参数字典
            bev_config: BEV配置
                - area_extents: [[-32, 32], [-32, 32]] (米)
                - voxel_size: [0.25, 0.25] (米)
                - grid_size: [256, 256] (网格数)
        """
        self.camera_params = camera_params
        self.bev_config = bev_config
        
        # 解析相机参数
        self.translation = np.array(camera_params['translation'])
        self.rotation_quat = np.array(camera_params['rotation'])
        self.intrinsic = np.array(camera_params['camera_intrinsic'])
        
        # 构建外参矩阵
        self.extrinsic = self._build_extrinsic()
        
        # BEV参数
        self.bev_range = bev_config['area_extents']  # [[x_min, x_max], [y_min, y_max]]
        self.voxel_size = bev_config['voxel_size']
        self.grid_size = bev_config['grid_size']
        
    def _build_extrinsic(self):
        """构建外参矩阵"""
        R = self._quaternion_to_matrix(self.rotation_quat)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = self.translation
        return extrinsic
    
    def _quaternion_to_matrix(self, q):
        """四元数转旋转矩阵"""
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),         2*(x*z + w*y)],
            [2*(x*y + w*z),           1 - 2*(x**2 + z**2),   2*(y*z - w*x)],
            [2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*(x**2 + y**2)]
        ])
        return R
    
    def project_detections_to_bev(self, detections, method='ground_plane'):
        """
        将检测结果投影到BEV
        
        参数:
            detections: 检测结果列表
                [{
                    'box': [x1, y1, x2, y2],  # 图像坐标
                    'score': 0.95,
                    'label': 'car'
                }, ...]
            method: 投影方法
                - 'ground_plane': 假设物体在地面（z=0）
                - 'depth_estimation': 使用深度估计
                - 'multi_height': 多个高度假设
        
        返回:
            bev_map: (H, W) BEV置信度地图
            projected_positions: 投影后的位置列表
        """
        H, W = self.grid_size
        bev_confidence = np.zeros((H, W))
        bev_coverage = np.zeros((H, W))
        projected_positions = []
        
        for det in detections:
            box = det['box']
            score = det['score']
            
            # 方法1: 假设物体在地面
            if method == 'ground_plane':
                world_pos = self._project_box_to_ground(box)
            
            # 方法2: 使用深度估计
            elif method == 'depth_estimation':
                depth = self._estimate_depth(box, det['label'])
                world_pos = self._project_box_with_depth(box, depth)
            
            # 方法3: 多高度假设
            elif method == 'multi_height':
                world_positions = self._project_box_multi_height(box)
                for world_pos, height_weight in world_positions:
                    grid_pos = self._world_to_grid(world_pos)
                    if grid_pos is not None:
                        gx, gy = grid_pos
                        self._fill_bev_region(
                            bev_confidence, gx, gy, 
                            score * height_weight, 
                            radius=2
                        )
                continue
            
            # 转换为网格坐标
            grid_pos = self._world_to_grid(world_pos)
            
            if grid_pos is not None:
                gx, gy = grid_pos
                
                # 在BEV上标记
                self._fill_bev_region(
                    bev_confidence, gx, gy, score, radius=3
                )
                bev_coverage[gy, gx] = 1.0
                
                projected_positions.append({
                    'world_pos': world_pos,
                    'grid_pos': (gx, gy),
                    'score': score,
                    'label': det['label']
                })
        
        return bev_confidence, bev_coverage, projected_positions
    
    def _project_box_to_ground(self, box):
        """
        假设物体在地面，投影检测框到世界坐标
        
        策略：取检测框底部中心点
        """
        x1, y1, x2, y2 = box
        
        # 底部中心点（物体与地面接触点）
        u = (x1 + x2) / 2
        v = y2  # 底部
        
        # 反投影到地面 (z=0)
        world_pos = self._unproject_to_ground_plane(u, v)
        
        return world_pos
    
    def _unproject_to_ground_plane(self, u, v, ground_z=0.0):
        """
        将图像点反投影到地面
        
        数学原理：
        1. 图像坐标 (u, v) -> 相机坐标射线
        2. 射线与地面 z=ground_z 求交
        """
        # 1. 图像坐标 -> 归一化平面坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        
        # 相机坐标系中的射线方向
        ray_camera = np.array([x_norm, y_norm, 1.0])
        
        # 2. 相机坐标 -> 世界坐标
        # 相机在世界坐标系中的位置
        camera_pos_world = self.translation
        
        # 射线方向转换到世界坐标系
        R = self.extrinsic[:3, :3]
        ray_world = R @ ray_camera
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        # 3. 射线与地面求交
        # 参数方程: P = camera_pos + t * ray
        # 约束: P[2] = ground_z
        
        if abs(ray_world[2]) < 1e-6:
            # 射线平行于地面
            return None
        
        t = (ground_z - camera_pos_world[2]) / ray_world[2]
        
        if t < 0:
            # 交点在相机后方
            return None
        
        world_pos = camera_pos_world + t * ray_world
        
        return world_pos
    
    def _estimate_depth(self, box, label):
        """
        根据检测框大小估计深度
        
        简单启发式方法
        """
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        
        # 根据类别的典型高度估算
        typical_heights = {
            'car': 1.5,      # 米
            'person': 1.7,
            'bus': 3.0,
            'truck': 2.5,
        }
        
        object_height = typical_heights.get(label, 1.5)
        
        # 从像素高度反推深度
        fy = self.intrinsic[1, 1]
        depth = (object_height * fy) / max(box_height, 1)
        
        return np.clip(depth, 1.0, 100.0)
    
    def _project_box_with_depth(self, box, depth):
        """使用估计的深度投影"""
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = (y1 + y2) / 2
        
        # 图像坐标 + 深度 -> 相机坐标
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        point_cam = np.array([x_cam, y_cam, z_cam])
        
        # 相机坐标 -> 世界坐标
        R = self.extrinsic[:3, :3]
        t = self.translation
        point_world = R @ point_cam + t
        
        return point_world
    
    def _project_box_multi_height(self, box):
        """
        多高度假设：在不同高度平面上投影
        
        返回多个可能的世界位置及其权重
        """
        x1, y1, x2, y2 = box
        u = (x1 + x2) / 2
        v = y2  # 底部
        
        # 尝试不同高度
        heights = [0.0, -0.2, 0.2]  # 地面、稍低、稍高
        weights = [0.6, 0.2, 0.2]
        
        results = []
        for height, weight in zip(heights, weights):
            world_pos = self._unproject_to_ground_plane(u, v, ground_z=height)
            if world_pos is not None:
                results.append((world_pos, weight))
        
        return results
    
    def _world_to_grid(self, world_pos):
        """
        世界坐标 -> BEV网格索引
        
        参数:
            world_pos: [x, y, z] 世界坐标
        
        返回:
            (gx, gy): 网格索引，如果超出范围返回None
        """
        if world_pos is None:
            return None
        
        x, y, z = world_pos
        
        # 转换为网格索引
        gx = int((x - self.bev_range[0][0]) / self.voxel_size[0])
        gy = int((y - self.bev_range[1][0]) / self.voxel_size[1])
        
        # 检查范围
        if 0 <= gx < self.grid_size[1] and 0 <= gy < self.grid_size[0]:
            return (gx, gy)
        else:
            return None
    
    def _fill_bev_region(self, bev_map, gx, gy, value, radius=2):
        """
        在BEV地图上填充一个区域
        
        考虑检测框的实际大小，不只是一个点
        """
        H, W = bev_map.shape
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = gx + dx, gy + dy
                
                if 0 <= nx < W and 0 <= ny < H:
                    # 距离衰减
                    dist = np.sqrt(dx**2 + dy**2)
                    weight = np.exp(-dist / radius)
                    
                    bev_map[ny, nx] = max(bev_map[ny, nx], value * weight)



if __name__ == "__main__":
    # =============== 1. 配置 ===============
    camera_params = {
        "translation": [
                1.5,
                -0.0,
                1.3787471055984497
            ],
            "rotation": [
                0.5,
                -0.5,
                0.5,
                -0.5
            ],
        "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
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

    bev_config = {
        'area_extents': [[-256, 256], [-256, 256]],  # 覆盖64m x 64m区域
        'voxel_size': [0.25, 0.25],              # 每个网格0.25m
        'grid_size': [2048, 2048]                  # 256x256网格
    }

    # =============== 2. 创建投影器 ===============
    projector = ImageToBEVProjector(camera_params, bev_config)

    # =============== 3. 模拟检测结果 ===============
    detector = FasterRCNNDetector(device='cuda', conf_threshold=0.3)
    import cv2
    img_file = '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_BACK_id_1/scene_5_000006.jpg'
    image = cv2.imread(img_file)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect(image)

    # =============== 4. 投影到BEV ===============
    bev_confidence, bev_coverage, projected_positions = projector.project_detections_to_bev(
        detections,
        method='depth_estimation'  # 或 'depth_estimation', 'multi_height'
    )

    # =============== 5. 可视化 ===============
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像（模拟）
    axes[0].set_title('Image Detections')
    axes[0].imshow(plot_detections(image_rgb, detections))
    axes[0].set_xlim(0, 1600)
    axes[0].set_ylim(900, 0)

    # BEV置信度地图
    im1 = axes[1].imshow(bev_confidence, cmap='hot', origin='lower', 
                        extent=[bev_config['area_extents'][0][0], 
                                bev_config['area_extents'][0][1],
                                bev_config['area_extents'][1][0], 
                                bev_config['area_extents'][1][1]])
    axes[1].set_title('BEV Confidence Map')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    plt.colorbar(im1, ax=axes[1])

    # 标记相机位置
    cam_x, cam_y = camera_params['translation'][0], camera_params['translation'][1]
    axes[1].plot(cam_x, cam_y, 'b*', markersize=10, label='Camera')
    axes[1].legend()

    # BEV覆盖地图
    axes[2].imshow(bev_coverage, cmap='Blues', origin='lower',
                extent=[bev_config['area_extents'][0][0], 
                        bev_config['area_extents'][0][1],
                        bev_config['area_extents'][1][0], 
                        bev_config['area_extents'][1][1]])
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