import torch
import torch.nn as nn
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.segmentation import deeplabv3_resnet50

class ImageBasedRegionQualityEvaluator:
    def __init__(self, config):
        """
        基于图像的区域感知质量评估
        
        参数:
            config: 包含相机参数、BEV网格配置等
        """
        self.config = config
        
        # 加载预训练的检测模型
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.eval()
        
        # 加载分割模型（用于评估遮挡）
        self.segmentor = deeplabv3_resnet50(pretrained=True)
        self.segmentor.eval()
        
        # 相机参数
        self.camera_intrinsic = config.camera_intrinsic  # 内参矩阵
        self.camera_extrinsic = config.camera_extrinsic  # 外参矩阵
        
        # BEV网格参数
        self.bev_range = config.area_extents  # [[-32, 32], [-32, 32]]
        self.bev_resolution = config.voxel_size[0]  # 0.25m
        self.grid_size = (256, 256)  # H, W
        
    def evaluate_from_images(self, images, agent_id):
        """
        从图像评估每个BEV区域的感知质量
        
        参数:
            images: 字典 {'front': img, 'left': img, 'right': img, 'back': img}
            agent_id: 当前车辆ID
            
        返回:
            quality_map: (H, W) 每个BEV网格的感知质量分数 [0, 1]
        """
        H, W = self.grid_size
        quality_map = np.zeros((H, W))
        confidence_map = np.zeros((H, W))
        occlusion_map = np.zeros((H, W))
        coverage_map = np.zeros((H, W))
        
        # 对每个相机视角进行处理
        for camera_name, image in images.items():
            # 1. 目标检测 - 获取置信度
            detections = self._detect_objects(image)
            
            # 2. 语义分割 - 评估遮挡
            segmentation = self._segment_image(image)
            
            # 3. 将图像坐标投影到BEV网格
            camera_quality = self._project_to_bev(
                detections, 
                segmentation,
                camera_name
            )
            
            # 4. 融合多个相机的质量评估
            confidence_map = np.maximum(confidence_map, camera_quality['confidence'])
            occlusion_map = np.maximum(occlusion_map, camera_quality['occlusion'])
            coverage_map = np.maximum(coverage_map, camera_quality['coverage'])
        
        # 5. 综合多个指标计算最终质量分数
        quality_map = self._compute_quality_score(
            confidence_map,
            occlusion_map,
            coverage_map
        )
        
        return quality_map, {
            'confidence': confidence_map,
            'occlusion': occlusion_map,
            'coverage': coverage_map
        }
    
    def _detect_objects(self, image):
        """
        使用检测模型获取目标和置信度
        
        返回:
            detections: List[Dict]
                - boxes: (N, 4) 边界框坐标
                - scores: (N,) 置信度分数
                - labels: (N,) 类别标签
        """
        with torch.no_grad():
            image_tensor = self._preprocess_image(image)
            predictions = self.detector([image_tensor])[0]
        
        # 过滤低置信度检测
        keep = predictions['scores'] > 0.3
        
        return {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy()
        }
    
    def _segment_image(self, image):
        """
        语义分割 - 用于评估遮挡情况
        
        返回:
            segmentation: (H, W) 分割掩码
        """
        with torch.no_grad():
            image_tensor = self._preprocess_image(image)
            output = self.segmentor(image_tensor.unsqueeze(0))
            seg_mask = output['out'][0].argmax(0).cpu().numpy()
        
        return seg_mask
    
    def _project_to_bev(self, detections, segmentation, camera_name):
        """
        将图像空间的信息投影到BEV网格
        
        这是核心函数！
        """
        H, W = self.grid_size
        confidence_grid = np.zeros((H, W))
        occlusion_grid = np.zeros((H, W))
        coverage_grid = np.zeros((H, W))
        
        # 获取该相机的外参
        extrinsic = self.camera_extrinsic[camera_name]
        intrinsic = self.camera_intrinsic[camera_name]
        
        # 对每个检测框进行处理
        for box, score, label in zip(
            detections['boxes'], 
            detections['scores'], 
            detections['labels']
        ):
            x1, y1, x2, y2 = box
            
            # 估计3D位置（简化：假设物体在地面上）
            # 更精确的方法：使用深度估计或立体视觉
            depth = self._estimate_depth(box, camera_name, label)
            
            # 将图像坐标 + 深度转换为3D世界坐标
            center_2d = [(x1 + x2) / 2, (y1 + y2) / 2]
            world_coords = self._image_to_world(
                center_2d, depth, intrinsic, extrinsic
            )
            
            # 转换为BEV网格索引
            grid_x, grid_y = self._world_to_grid(world_coords)
            
            if 0 <= grid_x < W and 0 <= grid_y < H:
                # 更新置信度（使用检测分数）
                confidence_grid[grid_y, grid_x] = max(
                    confidence_grid[grid_y, grid_x], 
                    score
                )
                
                # 标记覆盖区域
                coverage_grid[grid_y, grid_x] = 1.0
                
                # 根据物体大小扩散到周围网格
                size = max(x2 - x1, y2 - y1)
                radius = int(size / 50)  # 根据图像大小调整
                for dy in range(-radius, radius+1):
                    for dx in range(-radius, radius+1):
                        ny, nx = grid_y + dy, grid_x + dx
                        if 0 <= nx < W and 0 <= ny < H:
                            dist_weight = 1.0 / (1 + np.sqrt(dx**2 + dy**2))
                            confidence_grid[ny, nx] = max(
                                confidence_grid[ny, nx],
                                score * dist_weight
                            )
                            coverage_grid[ny, nx] = 1.0
        
        # 评估遮挡情况
        occlusion_grid = self._evaluate_occlusion_from_segmentation(
            segmentation, intrinsic, extrinsic
        )
        
        return {
            'confidence': confidence_grid,
            'occlusion': occlusion_grid,
            'coverage': coverage_grid
        }
    
    def _estimate_depth(self, box, camera_name, label):
        """
        根据检测框大小和类别估计深度
        
        更好的方法：
        1. 使用单目深度估计网络（如MiDaS）
        2. 使用多视图几何
        3. 使用激光雷达标注的深度真值
        """
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        
        # 简单启发式：物体在图像中越大，距离越近
        # 实际应用中应该用学习的深度估计
        if label == 1:  # 假设1是车辆
            typical_height = 1.5  # 车辆典型高度（米）
            focal_length = self.camera_intrinsic[camera_name][1, 1]
            depth = (typical_height * focal_length) / max(box_height, 1)
        elif label == 2:  # 行人
            typical_height = 1.7
            focal_length = self.camera_intrinsic[camera_name][1, 1]
            depth = (typical_height * focal_length) / max(box_height, 1)
        else:
            depth = 15.0  # 默认距离
        
        return np.clip(depth, 1.0, 50.0)
    
    def _image_to_world(self, pixel, depth, intrinsic, extrinsic):
        """
        图像坐标 → 相机坐标 → 世界坐标
        """
        # 图像坐标 → 相机坐标
        u, v = pixel
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # 相机坐标 → 世界坐标
        point_world = extrinsic @ point_cam
        
        return point_world[:3]
    
    def _world_to_grid(self, world_coords):
        """
        世界坐标 → BEV网格索引
        """
        x, y, z = world_coords
        
        # 转换为网格索引
        grid_x = int((x - self.bev_range[0][0]) / self.bev_resolution)
        grid_y = int((y - self.bev_range[1][0]) / self.bev_resolution)
        
        return grid_x, grid_y
    
    def _evaluate_occlusion_from_segmentation(self, segmentation, intrinsic, extrinsic):
        """
        从语义分割评估遮挡程度
        
        思路：
        1. 识别前景物体（车辆、建筑等）
        2. 计算它们在BEV中遮挡的区域
        3. 被遮挡的区域质量降低
        """
        H, W = self.grid_size
        occlusion_grid = np.zeros((H, W))
        
        # 前景类别（会造成遮挡的物体）
        occluding_classes = [7, 26, 19]  # 车辆、建筑、墙等（COCO类别）
        
        # 找到遮挡物体的像素
        occluding_mask = np.isin(segmentation, occluding_classes)
        
        # 对每个遮挡像素，标记其后方的BEV区域为遮挡
        h_img, w_img = segmentation.shape
        for v in range(0, h_img, 10):  # 采样以提高效率
            for u in range(0, w_img, 10):
                if occluding_mask[v, u]:
                    # 这个像素对应的世界位置
                    depth = self._estimate_depth(
                        [u-5, v-5, u+5, v+5], 'front', 1
                    )
                    world_pos = self._image_to_world(
                        [u, v], depth, intrinsic, extrinsic
                    )
                    grid_x, grid_y = self._world_to_grid(world_pos)
                    
                    # 标记该位置后方的区域为遮挡
                    if 0 <= grid_x < W and 0 <= grid_y < H:
                        # 遮挡区域：该位置到视野边缘
                        self._mark_occluded_region(
                            occlusion_grid, grid_x, grid_y, world_pos
                        )
        
        return occlusion_grid
    
    def _mark_occluded_region(self, occlusion_grid, grid_x, grid_y, occluder_pos):
        """标记被遮挡的区域"""
        # 简化：标记该网格后方一定范围内的区域
        for dy in range(5):  # 后方5个网格
            ny = grid_y + dy
            if 0 <= ny < occlusion_grid.shape[0]:
                occlusion_grid[ny, grid_x] = min(
                    occlusion_grid[ny, grid_x] + 0.3,
                    1.0
                )
    
    def _compute_quality_score(self, confidence_map, occlusion_map, coverage_map):
        """
        综合多个指标计算感知质量分数
        
        公式: Q = w1 * confidence * (1 - occlusion) + w2 * coverage
        """
        w1, w2, w3 = 0.5, 0.3, 0.2
        
        quality = (
            w1 * confidence_map * (1.0 - occlusion_map) +  # 高置信度 且 无遮挡
            w2 * coverage_map +                              # 有覆盖
            w3 * confidence_map                              # 基础置信度
        )
        
        # 归一化到 [0, 1]
        quality = np.clip(quality, 0, 1)
        
        return quality
    
    def _preprocess_image(self, image):
        """图像预处理"""
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((800, 800)),
            T.ToTensor(),
        ])
        return transform(image)