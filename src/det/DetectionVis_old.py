"""
BEV 检测框融合对比可视化系统 - 适配用户代码结构

核心改进:
- ✅ 直接使用 projected_positions 数据
- ✅ 使用 world_pos_global (世界坐标) 而不是 box 坐标
- ✅ 每辆车以自己为中心的BEV视角
- ✅ 根据 area_extents 进行边界检查
- ✅ 支持融合前后对比

数据结构兼容:
    projected_positions: [
        {
            'world_pos': (x, y, z),        # 世界坐标
            'grid_pos': (gx, gy),          # 网格坐标
            'score': float,                # 置信度
            'label': str                   # 类别
        },
        ...
    ]
    
    local_bev_config: {
        'area_extents': [[-32, 32], [-32, 32]],    # [X范围, Y范围]
        'voxel_size': (0.2, 0.2),                   # 体素大小
        'grid_size': [height, width]                # 网格大小
    }
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from typing import Dict, List, Tuple, Optional


class BEVFusionDetectionVisualizer:
    """针对用户代码结构的BEV融合可视化器"""
    
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
    def _is_detection_in_bev_range(world_pos: Tuple[float, float, float],
                                   area_extents: List[List[float]]) -> bool:
        """
        检查世界坐标是否在BEV范围内
        
        参数:
            world_pos: (x, y, z) - 世界坐标
            area_extents: [[x_min, x_max], [y_min, y_max]] - BEV范围
        
        返回:
            True: 在范围内, False: 超出范围
        """
        x, y, z = world_pos
        x_min, x_max = area_extents[0]
        y_min, y_max = area_extents[1]
        
        return (x_min <= x <= x_max and y_min <= y <= y_max)

    @staticmethod
    def _is_detection_in_bev_range_relative(relative_pos: Tuple[float, float, float],
                                           area_extents: List[List[float]]) -> bool:
        """
        检查相对坐标是否在BEV范围内
        
        参数:
            relative_pos: (bev_x, bev_y, z) - 相对于自车中心的坐标
            area_extents: [[x_min, x_max], [y_min, y_max]] - BEV范围
        
        返回:
            True: 在范围内, False: 超出范围
        """
        bev_x, bev_y, _ = relative_pos
        x_min, x_max = area_extents[0]
        y_min, y_max = area_extents[1]
        
        return (x_min <= bev_x <= x_max and y_min <= bev_y <= y_max)

    @staticmethod
    def _world_to_bev_relative(world_pos: Tuple[float, float, float],
                               ego_center: Tuple[float, float, float]) -> Tuple[float, float]:
        """
        将世界坐标转换为相对于自车中心的BEV坐标
        
        参数:
            world_pos: (x, y, z) - 物体的世界坐标
            ego_center: (x, y, z) - 自车中心的世界坐标
        
        返回:
            (bev_x, bev_y) - 相对于自车的BEV坐标
        """
        x, y, z = world_pos
        ego_x, ego_y, ego_z = ego_center
        
        # BEV坐标: 相对位置
        bev_x = x - ego_x  # X方向（前后）
        bev_y = y - ego_y  # Y方向（左右）
        
        return (bev_x, bev_y)
    
    def visualize_fusion_comparison(self,
                                   vehicle_id: str,
                                   ego_center_world: Tuple[float, float, float],
                                   own_projected_positions: List[Dict],
                                   others_projected_positions: List[Dict],
                                   local_bev_config: Dict,
                                   save_path: Optional[str] = None) -> Dict:
        """
        生成融合前后的对比可视化图
        
        参数:
            vehicle_id: 车辆ID
            ego_center_world: 自车中心的世界坐标 (x, y, z)
            own_projected_positions: 自车检测列表 (来自 projected_positions)
            others_projected_positions: 其他车检测列表
            local_bev_config: 局部BEV配置
                {
                    'area_extents': [[-32, 32], [-32, 32]],
                    'voxel_size': (0.2, 0.2),
                    'grid_size': [320, 320]
                }
            save_path: 保存路径
        
        返回:
            stats: 过滤和显示统计
        """
        
        # 创建对比图 (1行2列)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        area_extents = local_bev_config['area_extents']
        
        # ========== 左图: 融合前 (仅自车检测) ==========
        ax_before = axes[0]
        stats_before = self._draw_bev_detections(
            ax_before,
            ego_center_world=ego_center_world,
            own_positions=own_projected_positions,  # 左图不显示自车
            others_positions=[],  # 用"其他"显示自车（红色）
            area_extents=area_extents,
            title=f'{vehicle_id} - Before Fusion\n(Ego Only)',
            show_as_own=True  # 用红色显示
        )
        
        # ========== 右图: 融合后 (自车+其他车检测) ==========
        ax_after = axes[1]
        stats_after = self._draw_bev_detections(
            ax_after,
            ego_center_world=ego_center_world,
            own_positions=own_projected_positions,
            others_positions=others_projected_positions,
            area_extents=area_extents,
            title=f'{vehicle_id} - After Fusion\n(Ego + Others)',
            show_as_own=False  # 用实际颜色显示
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
    
    def _draw_bev_detections(self,
                            ax,
                            ego_center_world: Tuple[float, float, float],
                            own_positions: List[Dict],
                            others_positions: List[Dict],
                            area_extents: List[List[float]],
                            title: str = 'BEV Detections',
                            show_as_own: bool = False) -> Dict:
        """
        在BEV图上绘制检测框
        
        参数:
            ax: matplotlib axes
            ego_center_world: 自车中心的世界坐标
            own_positions: 自车检测位置列表
            others_positions: 其他车检测位置列表
            area_extents: BEV范围 [[x_min, x_max], [y_min, y_max]]
            title: 标题
            show_as_own: 是否将others显示为自车颜色（用于融合前显示）
        
        返回:
            stats: 绘制统计
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
        
        # 设置背景
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 绘制自车位置 (原点)
        ax.plot(0, 0, 'p', color='purple', markersize=15, 
               label='Ego Vehicle', zorder=100, markeredgecolor='darkviolet', markeredgewidth=2)
        
        # ========== 绘制自车检测 (红色实线) ==========
        own_drawn = 0
        own_filtered = 0
        
        for det in own_positions:
            world_pos = det['world_pos']
            score = det['score']
            label = det.get('label', 'unknown')
            
            # 转换到相对坐标
            bev_x, bev_y = self._world_to_bev_relative(world_pos, ego_center_world)
            
            # 检查是否在BEV范围内
            #if not self._is_detection_in_bev_range(world_pos, area_extents):
            if not self._is_detection_in_bev_range_relative((bev_x, bev_y, 0), area_extents):
                own_filtered += 1
                continue
            

            
            own_drawn += 1
            
            # 绘制矩形框 (假设物体大小为 2m × 2m)
            box_size = 1.0  # 半宽半高
            rect = Rectangle(
                (bev_x - box_size, bev_y - box_size), 
                2 * box_size, 2 * box_size,
                linewidth=2.5,
                edgecolor='#FF0000',
                facecolor='none',
                linestyle='-',
                zorder=5
            )
            ax.add_patch(rect)
            
            # 在框上方添加信息标签
            # text_label = f'E#{own_drawn}\n{score:.2f}'
            # ax.text(bev_x - box_size, bev_y - box_size - 1, text_label,
            #        fontsize=9, color='#FF0000', fontweight='bold',
            #        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCCCC', alpha=0.8),
            #        ha='center', va='top', zorder=10)
        
        # ========== 绘制其他车检测 (绿色虚线 或 红色实线) ==========
        others_drawn = 0
        others_filtered = 0
        
        for det in others_positions:
            world_pos = det['world_pos']
            score = det['score']
            label = det.get('label', 'unknown')
            
            # 转换到相对坐标
            bev_x, bev_y = self._world_to_bev_relative(world_pos, ego_center_world)
            
            # 检查是否在BEV范围内
            # if not self._is_detection_in_bev_range(world_pos, area_extents):
            if not self._is_detection_in_bev_range_relative((bev_x, bev_y, 0), area_extents):
                others_filtered += 1
                continue
            
            others_drawn += 1
            
            # 选择颜色和线型
            if show_as_own:
                # 在融合前显示时，用红色显示自车
                edge_color = '#FF0000'
                linestyle = '-'
                prefix = 'E'
            else:
                # 在融合后显示时，用绿色显示其他车
                edge_color = '#00AA00'
                linestyle = '--'
                prefix = 'O'
            
            # 绘制矩形框
            box_size = 1.0
            rect = Rectangle(
                (bev_x - box_size, bev_y - box_size),
                2 * box_size, 2 * box_size,
                linewidth=2.5,
                edgecolor=edge_color,
                facecolor='none',
                linestyle=linestyle,
                zorder=4
            )
            ax.add_patch(rect)
            
            # 在框上方添加信息标签
            # text_label = f'{prefix}#{others_drawn}\n{score:.2f}'
            # ax.text(bev_x - box_size, bev_y - box_size - 1, text_label,
            #        fontsize=9, color=edge_color, fontweight='bold',
            #        bbox=dict(boxstyle='round,pad=0.3', 
            #                 facecolor='#FFCCCC' if show_as_own else '#CCFFCC', 
            #                 alpha=0.8),
            #        ha='center', va='top', zorder=10)
        
        # 添加图例
        own_line = plt.Line2D([0], [0], color='#FF0000', linewidth=3, 
                             label=f'Ego Detections ({own_drawn})')
        others_line = plt.Line2D([0], [0], color='#00AA00', linewidth=3, linestyle='--',
                                label=f'Other Vehicles Detections ({others_drawn})')
        ego_marker = plt.Line2D([0], [0], marker='p', color='w', 
                               markerfacecolor='purple', markersize=10,
                               label='Ego Vehicle Center', markeredgecolor='darkviolet')
        
        ax.legend(handles=[ego_marker, own_line, others_line], 
                 loc='upper right', fontsize=10)
        
        # 添加统计信息框
        info_text = f'In Range: {own_drawn + others_drawn}\n' \
                   f'Ego: {own_drawn} | Others: {others_drawn}\n' \
                   f'Out of Range: {own_filtered + others_filtered}'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#FFFFEE', alpha=0.9, edgecolor='black'))
        
        return {
            'own_drawn': own_drawn,
            'own_filtered': own_filtered,
            'others_drawn': others_drawn,
            'others_filtered': others_filtered,
        }
    
    def generate_fusion_statistics(self,
                                  vehicle_id: str,
                                  ego_center_world: Tuple[float, float, float],
                                  own_projected_positions: List[Dict],
                                  others_projected_positions: List[Dict],
                                  area_extents: List[List[float]],
                                  output_file: Optional[str] = None) -> str:
        """
        生成融合统计报告
        
        参数:
            vehicle_id: 车辆ID
            ego_center_world: 自车中心世界坐标
            own_projected_positions: 自车检测
            others_projected_positions: 其他车检测
            area_extents: BEV范围
            output_file: 输出文件路径
        
        返回:
            report: 报告文本
        """
        
        x_min, x_max = area_extents[0]
        y_min, y_max = area_extents[1]
        
        # 统计在范围内和范围外的检测
        own_in_range = sum(1 for det in own_projected_positions 
                          if self._is_detection_in_bev_range(det['world_pos'], area_extents))
        own_out_range = len(own_projected_positions) - own_in_range
        
        others_in_range = sum(1 for det in others_projected_positions 
                             if self._is_detection_in_bev_range(det['world_pos'], area_extents))
        others_out_range = len(others_projected_positions) - others_in_range
        
        total_before = own_in_range
        total_after = own_in_range + others_in_range
        new_detections = others_in_range
        improvement_rate = (new_detections / total_before * 100) if total_before > 0 else 0
        
        report = f"""
{'='*70}
BEV FUSION COMPARISON REPORT
{'='*70}

Vehicle ID: {vehicle_id}
Report Generated: 2025-12-18

BEV Configuration:
  Area Extents: X=[{x_min}, {x_max}], Y=[{y_min}, {y_max}]
  Range: X={x_max - x_min}m, Y={y_max - y_min}m

FUSION EFFECTIVENESS METRICS:
{'-'*70}
Before Fusion (Ego Only - In Range):     {total_before} detections
After Fusion (Ego + Others - In Range):  {total_after} detections
New Detections from Others (In Range):   +{new_detections}
Improvement Rate:                        +{improvement_rate:.1f}%

OUT-OF-RANGE DETECTIONS (Filtered Out):
{'-'*70}
Ego Detections Out of Range:             {own_out_range} / {len(own_projected_positions)}
Others Detections Out of Range:          {others_out_range} / {len(others_projected_positions)}
Total Filtered Out:                      {own_out_range + others_out_range} / {len(own_projected_positions) + len(others_projected_positions)}

DETECTION DETAILS (IN RANGE ONLY):
{'-'*70}

IN-RANGE EGO DETECTIONS:
"""
        
        ego_count = 1
        ego_center_x, ego_center_y, _ = ego_center_world
        
        for det in own_projected_positions:
            if self._is_detection_in_bev_range(det['world_pos'], area_extents):
                wx, wy, wz = det['world_pos']
                bev_x = wx - ego_center_x
                bev_y = wy - ego_center_y
                distance = np.sqrt(bev_x**2 + bev_y**2)
                
                report += f"\n  Detection E#{ego_count}:\n"
                report += f"    - World Position: ({wx:.2f}, {wy:.2f}, {wz:.2f})\n"
                report += f"    - BEV Relative Position: ({bev_x:.2f}, {bev_y:.2f})\n"
                report += f"    - Distance to Center: {distance:.2f}m\n"
                report += f"    - Confidence: {det['score']:.3f}\n"
                report += f"    - Label: {det.get('label', 'unknown')}\n"
                ego_count += 1
        
        report += f"\n\nIN-RANGE OTHER VEHICLES DETECTIONS (After Fusion Only):\n"
        
        others_count = 1
        for det in others_projected_positions:
            if self._is_detection_in_bev_range(det['world_pos'], area_extents):
                wx, wy, wz = det['world_pos']
                bev_x = wx - ego_center_x
                bev_y = wy - ego_center_y
                distance = np.sqrt(bev_x**2 + bev_y**2)
                
                report += f"\n  Detection O#{others_count}:\n"
                report += f"    - World Position: ({wx:.2f}, {wy:.2f}, {wz:.2f})\n"
                report += f"    - BEV Relative Position: ({bev_x:.2f}, {bev_y:.2f})\n"
                report += f"    - Distance to Center: {distance:.2f}m\n"
                report += f"    - Confidence: {det['score']:.3f}\n"
                report += f"    - Label: {det.get('label', 'unknown')}\n"
                others_count += 1
        
        report += f"\n\nFUSION ANALYSIS:\n"
        report += f"-{'-'*68}\n"
        
        if new_detections > 0:
            report += f"✓ Fusion successfully added {new_detections} new detection(s) in BEV range\n"
            report += f"✓ Detection capability improved by {improvement_rate:.1f}%\n"
            report += f"✓ Total detection coverage increased from {total_before} to {total_after}\n"
        else:
            report += f"ℹ No additional in-range detections from other vehicles\n"
        
        if own_out_range + others_out_range > 0:
            report += f"\nℹ Note: {own_out_range + others_out_range} detection(s) were out of BEV range and filtered out\n"
        
        report += f"\n{'='*70}\n"
        
        # 保存报告
        if output_file is None:
            output_file = os.path.join(self.output_dir, f'report_{vehicle_id}.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 报告已保存: {output_file}")
        
        return report


# ========== 使用示例 ==========

if __name__ == '__main__':
    print("BEV Detection Fusion Visualizer - Adapted for Your Code")
    print("="*70)
    
    # 创建可视化器
    visualizer = BEVFusionVisualizerForYourCode(output_dir='./bev_fusion_results_adapted')
    
    # 示例数据 - 模拟你的代码结构
    # 自车中心在世界坐标系中的位置
    ego_center_world = (100.0, 50.0, 0.0)  # (x, y, z)
    
    # 自车检测 (来自 projected_positions)
    own_projected_positions = [
        {
            'world_pos': (105.0, 50.0, 0.0),   # 前方5m
            'grid_pos': (525, 250),
            'score': 0.92,
            'label': 'car'
        },
        {
            'world_pos': (100.0, 55.0, 0.0),   # 右方5m
            'grid_pos': (500, 275),
            'score': 0.87,
            'label': 'pedestrian'
        },
        {
            'world_pos': (150.0, 50.0, 0.0),   # 前方50m (可能超出范围)
            'grid_pos': (750, 250),
            'score': 0.85,
            'label': 'car'
        },
    ]
    
    # 其他车检测 (融合新增)
    others_projected_positions = [
        {
            'world_pos': (110.0, 45.0, 0.0),   # 前右方向
            'grid_pos': (550, 225),
            'score': 0.89,
            'label': 'car'
        },
        {
            'world_pos': (95.0, 40.0, 0.0),    # 左后方向
            'grid_pos': (475, 200),
            'score': 0.91,
            'label': 'truck'
        },
    ]
    
    # BEV配置 (来自你的代码)
    local_bev_config = {
            'area_extents': [[-32, 32], [-32, 32]],  
            'voxel_size': [4, 4],              
            'grid_size': [16, 16]                  
        }
    
    
    print(f"\n自车中心位置: {ego_center_world}")
    print(f"BEV范围: X={local_bev_config['area_extents'][0]}, Y={local_bev_config['area_extents'][1]}")
    
    # 生成对比可视化
    stats = visualizer.visualize_fusion_comparison(
        vehicle_id='vehicle_001',
        ego_center_world=ego_center_world,
        own_projected_positions=own_projected_positions,
        others_projected_positions=others_projected_positions,
        local_bev_config=local_bev_config
    )
    
    print(f"\n统计结果:")
    print(f"  自车: {stats['own_total']} → {stats['own_in_range']} (超出范围: {stats['own_out_range']})")
    print(f"  其他: {stats['others_total']} → {stats['others_in_range']} (超出范围: {stats['others_out_range']})")
    
    # 生成统计报告
    report = visualizer.generate_fusion_statistics(
        vehicle_id='vehicle_001',
        ego_center_world=ego_center_world,
        own_projected_positions=own_projected_positions,
        others_projected_positions=others_projected_positions,
        area_extents=local_bev_config['area_extents']
    )
    
    print(report)