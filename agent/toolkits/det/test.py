import numpy as np
import cv2
import matplotlib.pyplot as plt
from .FastCNNDet import FasterRCNNDetector
from .utils import plot_detections

import numpy as np


class CoordinateTransformer:
    """
    处理多级坐标系变换（V2X-Sim/nuScenes 版本）
    """
    
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        """
        四元数转旋转矩阵
        
        参数:
            q: [w, x, y, z] 格式的四元数 (V2X-Sim/nuScenes 标准)
        """
        w, x, y, z = q  # ⭐ V2X-Sim 使用 [w, x, y, z]
        R = np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - w*z),         2*(x*z + w*y)],
            [2*(x*y + w*z),           1 - 2*(x**2 + z**2),   2*(y*z - w*x)],
            [2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*(x**2 + y**2)]
        ])
        return R
    
    @staticmethod
    def build_transform_matrix(translation, rotation_quat, inverse=False):
        """
        构建4x4变换矩阵
        
        参数:
            translation: [x, y, z]
            rotation_quat: [w, x, y, z] (V2X-Sim 格式)
            inverse: 是否返回逆变换矩阵
        """
        R = CoordinateTransformer.quaternion_to_rotation_matrix(rotation_quat)
        t = np.array(translation)
        
        if inverse:
            R_inv = R.T
            t_inv = -R_inv @ t
            
            T = np.eye(4)
            T[:3, :3] = R_inv
            T[:3, 3] = t_inv
        else:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
        
        return T
    
def test_coordinate_fix():
    """测试修正后的坐标变换"""
    
    # V2X-Sim 的相机参数（从截图）
    camera_params = {
        'translation': [-2.0, -0.0, 1.3787471055984497],
        'rotation': [0.5, -0.5, -0.5, 0.5]  # [w, x, y, z]
    }
    
    # 测试四元数转换
    R = CoordinateTransformer.quaternion_to_rotation_matrix(
        camera_params['rotation']
    )
    
    print("旋转矩阵:")
    print(R)
    print("\n相机坐标轴在车辆坐标系中的方向:")
    print(f"  相机X轴(右) → {R[:, 0]}")
    print(f"  相机Y轴(下) → {R[:, 1]}")
    print(f"  相机Z轴(前) → {R[:, 2]}")
    
    # 检查是否合理
    # 相机前方(Z)应该主要朝车辆前方(X)
    if abs(R[0, 2]) > 0.8:
        print("\n✓ 相机朝向合理")
    else:
        print("\n⚠️ 相机朝向可能有问题")

test_coordinate_fix()