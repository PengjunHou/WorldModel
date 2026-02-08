"""
BEV 检测框融合对比可视化系统 - 支持背景图片

核心改进:
- ✅ 支持添加场景的BEV背景图片
- ✅ 支持添加鸟瞰图作为参考
- ✅ 支持从confidence map生成热力图背景
- ✅ 支持自定义背景透明度
- ✅ 更美观的可视化效果
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from typing import Dict, List, Tuple, Optional
from PIL import Image
import cv2


class BEVFusionDetectionVisualizer:
    """支持背景图片的BEV融合可视化器"""
    
    def __init__(self, output_dir: str = './bev_fusion_results'):
        """
        初始化可视化器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 颜色定义
        self.colors = {
            'own_detection': '#FF0000',        # 红色 - 自车检测
            'others_detection': '#00AA00',     # 绿色 - 其他车检测
        }
    
    @staticmethod
    def _is_detection_in_bev_range_relative(relative_pos: Tuple[float, float, float],
                                           area_extents: List[List[float]]) -> bool:
        """检查相对坐标是否在BEV范围内"""
        bev_x, bev_y, _ = relative_pos
        x_min, x_max = area_extents[0]
        y_min, y_max = area_extents[1]
        
        return (x_min <= bev_x <= x_max and y_min <= bev_y <= y_max)
    
    @staticmethod
    def _world_to_bev_relative(world_pos: Tuple[float, float, float],
                               ego_center: Tuple[float, float, float]) -> Tuple[float, float]:
        """将世界坐标转换为相对于自车中心的BEV坐标"""
        x, y, z = world_pos
        ego_x, ego_y, ego_z = ego_center
        
        bev_x = x - ego_x
        bev_y = y - ego_y
        
        return (bev_x, bev_y)
    
    def visualize_fusion_comparison_with_background(self,
                                                   vehicle_id: str,
                                                   ego_center_world: Tuple[float, float, float],
                                                   own_projected_positions: List[Dict],
                                                   others_projected_positions: List[Dict],
                                                   local_bev_config: Dict,
                                                   bev_background_image: Optional[np.ndarray] = None,
                                                   bev_confidence_map: Optional[np.ndarray] = None,
                                                   background_alpha: float = 0.7,
                                                   save_path: Optional[str] = None) -> Dict:
        """
        生成融合前后的对比可视化图（支持背景图片）
        
        参数:
            vehicle_id: 车辆ID
            ego_center_world: 自车中心的世界坐标 (x, y, z)
            own_projected_positions: 自车检测列表
            others_projected_positions: 其他车检测列表
            local_bev_config: 局部BEV配置
            bev_background_image: BEV背景图片 (numpy array)
                - 如果提供，将作为背景显示
                - 形状: (height, width, 3) 或 (height, width)
                - 值: 0-255 (uint8) 或 0-1 (float)
            bev_confidence_map: BEV置信度热力图 (numpy array)
                - 形状: (height, width)
                - 如果提供，将转换为热力图背景
            background_alpha: 背景透明度 (0-1)
                - 0: 完全透明（仅显示检测框）
                - 1: 完全不透明（背景覆盖所有）
            save_path: 保存路径
        
        返回:
            stats: 统计信息
        """
        
        # 创建对比图 (1行2列)
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        area_extents = local_bev_config['area_extents']
        
        # ========== 左图: 融合前 (仅自车检测) ==========
        ax_before = axes[0]
        stats_before = self._draw_bev_detections_with_background(
            ax_before,
            ego_center_world=ego_center_world,
            own_positions=[],
            others_positions=own_projected_positions,
            area_extents=area_extents,
            bev_background_image=bev_background_image,
            bev_confidence_map=bev_confidence_map,
            background_alpha=background_alpha,
            title=f'{vehicle_id} - Before Fusion\n(Ego Only)',
            show_as_own=True
        )
        
        # ========== 右图: 融合后 (自车+其他车检测) ==========
        ax_after = axes[1]
        stats_after = self._draw_bev_detections_with_background(
            ax_after,
            ego_center_world=ego_center_world,
            own_positions=own_projected_positions,
            others_positions=others_projected_positions,
            area_extents=area_extents,
            bev_background_image=bev_background_image,
            bev_confidence_map=bev_confidence_map,
            background_alpha=background_alpha,
            title=f'{vehicle_id} - After Fusion\n(Ego + Others)',
            show_as_own=False
        )
        
        # 收集统计信息
        stats = {
            'own_total': len(own_projected_positions),
            'own_in_range': stats_after['own_drawn'],
            'own_out_range': len(own_projected_positions) - stats_after['own_drawn'],
            'others_total': len(others_projected_positions),
            'others_in_range': stats_after['others_drawn'],
            'others_out_range': len(others_projected_positions) - stats_after['others_drawn'],
        }
        
        # 打印过滤信息
        print(f"\n[{vehicle_id}] BEV范围内的检测:")
        print(f"  自车检测: {stats['own_in_range']} / {stats['own_total']} (超出范围: {stats['own_out_range']})")
        print(f"  其他车检测: {stats['others_in_range']} / {stats['others_total']} (超出范围: {stats['others_out_range']})")
        
        # 添加统计信息
        before_count = stats['own_in_range']
        after_count = stats['own_in_range'] + stats['others_in_range']
        new_count = stats['others_in_range']
        
        if before_count > 0:
            improvement = new_count / before_count * 100
        else:
            improvement = 0 if new_count == 0 else float('inf')
        
        # 在图下方添加统计文字
        fig.text(0.5, 0.02, 
                f'Before: {before_count} | After: {after_count} | New: +{new_count} (+{improvement:.1f}%) | Out of Range: {stats["own_out_range"] + stats["others_out_range"]}',
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # 保存图像
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'fusion_comparison_{vehicle_id}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图已保存: {save_path}")
        
        plt.close(fig)
        
        return stats
    
    def _draw_bev_detections_with_background(self,
                                            ax,
                                            ego_center_world: Tuple[float, float, float],
                                            own_positions: List[Dict],
                                            others_positions: List[Dict],
                                            area_extents: List[List[float]],
                                            bev_background_image: Optional[np.ndarray] = None,
                                            bev_confidence_map: Optional[np.ndarray] = None,
                                            background_alpha: float = 0.7,
                                            title: str = 'BEV Detections',
                                            show_as_own: bool = False) -> Dict:
        """
        在BEV图上绘制检测框，支持背景图片
        """
        
        x_min, x_max = area_extents[0]
        y_min, y_max = area_extents[1]
        
        # 设置坐标轴
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # BEV坐标系: Y轴向下
        
        ax.set_xlabel('X (meters) - Forward/Back', fontsize=11)
        ax.set_ylabel('Y (meters) - Left/Right', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        # ========== 绘制背景 ==========
        
        # 1. 添加置信度热力图背景
        if bev_confidence_map is not None:
            # 标准化置信度图
            confidence_normalized = bev_confidence_map.astype(np.float32)
            if confidence_normalized.max() > 0:
                confidence_normalized = confidence_normalized / confidence_normalized.max()
            
            # 转换为图像坐标并显示为热力图
            im = ax.imshow(confidence_normalized, 
                          extent=[x_min, x_max, y_max, y_min],  # 注意：extent的y顺序
                          cmap='hot',
                          alpha=0.6,
                          zorder=1)
            plt.colorbar(im, ax=ax, label='Confidence')
        
        # 2. 添加BEV背景图片
        elif bev_background_image is not None:
            # 处理图片
            if isinstance(bev_background_image, str):
                # 如果是路径，读取图片
                img = Image.open(bev_background_image)
                img = np.array(img)
            else:
                img = bev_background_image.copy()
            
            # 确保图片是uint8格式
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            # 处理灰度图
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # 显示背景图片
            ax.imshow(img,
                     extent=[x_min, x_max, y_max, y_min],
                     aspect='auto',
                     alpha=background_alpha,
                     zorder=1)
        
        # 3. 设置网格和背景色
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=2)
        
        # ========== 绘制自车位置 ==========
        ax.plot(0, 0, 'p', color='purple', markersize=15, 
               label='Ego Vehicle', zorder=100, markeredgecolor='darkviolet', markeredgewidth=2)
        
        # ========== 绘制自车检测 ==========
        own_drawn = 0
        own_filtered = 0
        
        for det in own_positions:
            world_pos = det['world_pos']
            score = det['score']
            label = det.get('label', 'unknown')
            
            # 转换到相对坐标
            bev_x, bev_y = self._world_to_bev_relative(world_pos, ego_center_world)
            
            # 检查是否在BEV范围内（使用相对坐标判断）
            if not self._is_detection_in_bev_range_relative((bev_x, bev_y, 0), area_extents):
                own_filtered += 1
                continue
            
            own_drawn += 1
            
            # 绘制矩形框
            box_size = 1.0
            rect = Rectangle(
                (bev_x - box_size, bev_y - box_size), 
                2 * box_size, 2 * box_size,
                linewidth=2.5,
                edgecolor='#FF0000',
                facecolor='#FF0000',
                alpha=0.3,
                linestyle='-',
                zorder=5
            )
            ax.add_patch(rect)
            
            # 添加边框（不填充）
            rect_outline = Rectangle(
                (bev_x - box_size, bev_y - box_size), 
                2 * box_size, 2 * box_size,
                linewidth=3,
                edgecolor='#FF0000',
                facecolor='none',
                linestyle='-',
                zorder=6
            )
            ax.add_patch(rect_outline)
            
            # 添加标签
            text_label = f'E#{own_drawn}\n{score:.2f}'
            ax.text(bev_x - box_size, bev_y - box_size - 1.5, text_label,
                   fontsize=10, color='#FF0000', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCCCC', 
                            alpha=0.9, edgecolor='#FF0000', linewidth=2),
                   ha='center', va='top', zorder=10)
        
        # ========== 绘制其他车检测 ==========
        others_drawn = 0
        others_filtered = 0
        
        for det in others_positions:
            world_pos = det['world_pos']
            score = det['score']
            label = det.get('label', 'unknown')
            
            # 转换到相对坐标
            bev_x, bev_y = self._world_to_bev_relative(world_pos, ego_center_world)
            
            # 检查是否在BEV范围内
            if not self._is_detection_in_bev_range_relative((bev_x, bev_y, 0), area_extents):
                others_filtered += 1
                continue
            
            others_drawn += 1
            
            # 选择颜色和线型
            if show_as_own:
                edge_color = '#FF0000'
                face_color = '#FF0000'
                linestyle = '-'
                prefix = 'E'
            else:
                edge_color = '#00AA00'
                face_color = '#00AA00'
                linestyle = '--'
                prefix = 'O'
            
            # 绘制矩形框（带填充）
            box_size = 1.0
            rect = Rectangle(
                (bev_x - box_size, bev_y - box_size),
                2 * box_size, 2 * box_size,
                linewidth=2.5,
                edgecolor=edge_color,
                facecolor=face_color,
                alpha=0.3,
                linestyle=linestyle,
                zorder=5
            )
            ax.add_patch(rect)
            
            # 添加边框（不填充）
            rect_outline = Rectangle(
                (bev_x - box_size, bev_y - box_size),
                2 * box_size, 2 * box_size,
                linewidth=3,
                edgecolor=edge_color,
                facecolor='none',
                linestyle=linestyle,
                zorder=6
            )
            ax.add_patch(rect_outline)
            
            # 添加标签
            text_label = f'{prefix}#{others_drawn}\n{score:.2f}'
            label_color = '#FFCCCC' if show_as_own else '#CCFFCC'
            ax.text(bev_x - box_size, bev_y - box_size - 1.5, text_label,
                   fontsize=10, color=edge_color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=label_color, 
                            alpha=0.9, edgecolor=edge_color, linewidth=2),
                   ha='center', va='top', zorder=10)
        
        # ========== 添加图例 ==========
        own_line = plt.Line2D([0], [0], color='#FF0000', linewidth=3, 
                             label=f'Ego Detections ({own_drawn})')
        others_line = plt.Line2D([0], [0], color='#00AA00', linewidth=3, linestyle='--',
                                label=f'Other Vehicles Detections ({others_drawn})')
        ego_marker = plt.Line2D([0], [0], marker='p', color='w', 
                               markerfacecolor='purple', markersize=10,
                               label='Ego Vehicle Center', markeredgecolor='darkviolet')
        
        ax.legend(handles=[ego_marker, own_line, others_line], 
                 loc='upper right', fontsize=10, framealpha=0.95)
        
        # ========== 添加统计信息框 ==========
        info_text = f'In Range: {own_drawn + others_drawn}\n' \
                   f'Ego: {own_drawn} | Others: {others_drawn}\n' \
                   f'Out of Range: {own_filtered + others_filtered}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#FFFFEE', alpha=0.95, 
                        edgecolor='black', linewidth=1.5))
        
        return {
            'own_drawn': own_drawn,
            'own_filtered': own_filtered,
            'others_drawn': others_drawn,
            'others_filtered': others_filtered,
        }


# ========== 使用示例 ==========

def example_with_background():
    """
    带背景图片的使用示例
    """
    
    visualizer = BEVFusionVisualizerWithBackground(output_dir='./bev_results_with_bg')
    
    # 示例数据
    ego_center_world = (100.0, 50.0, 0.0)
    
    own_projected_positions = [
        {
            'world_pos': (105.0, 50.0, 0.0),
            'score': 0.92,
            'label': 'car'
        },
        {
            'world_pos': (100.0, 55.0, 0.0),
            'score': 0.87,
            'label': 'pedestrian'
        },
    ]
    
    others_projected_positions = [
        {
            'world_pos': (110.0, 45.0, 0.0),
            'score': 0.89,
            'label': 'car'
        },
    ]
    
    local_bev_config = {
        'area_extents': [[-32, 32], [-32, 32]],
        'voxel_size': (0.2, 0.2),
        'grid_size': [320, 320]
    }
    
    # ========== 方案1: 使用BEV背景图片 ==========
    print("方案1: 使用BEV背景图片")
    
    # 假设你有一个BEV背景图片
    # bev_img = cv2.imread('your_bev_image.png')
    # bev_img = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
    
    # 生成虚拟背景图片（示例）
    bev_background = np.random.randint(0, 256, (320, 320, 3), dtype=np.uint8)
    
    stats = visualizer.visualize_fusion_comparison_with_background(
        'vehicle_001',
        ego_center_world,
        own_projected_positions,
        others_projected_positions,
        local_bev_config,
        bev_background_image=bev_background,
        background_alpha=0.6  # 背景透明度60%
    )
    
    # ========== 方案2: 使用置信度热力图 ==========
    print("\n方案2: 使用置信度热力图")
    
    # 生成虚拟置信度图（示例）
    confidence_map = np.random.rand(320, 320) * 255
    
    stats = visualizer.visualize_fusion_comparison_with_background(
        'vehicle_002',
        ego_center_world,
        own_projected_positions,
        others_projected_positions,
        local_bev_config,
        bev_confidence_map=confidence_map
    )
    
    # ========== 方案3: 无背景（仅框） ==========
    print("\n方案3: 无背景（仅框）")
    
    stats = visualizer.visualize_fusion_comparison_with_background(
        'vehicle_003',
        ego_center_world,
        own_projected_positions,
        others_projected_positions,
        local_bev_config
    )


if __name__ == '__main__':
    print("BEV Detection Fusion Visualizer - With Background Support")
    print("="*70)
    
    example_with_background()
    
    print("\n✓ 完成！")