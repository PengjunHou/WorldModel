import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
from src.det.FastCNNDet import FasterRCNNDetector
from src.det.ImageToBEV import ImageToBEVProjectorWithGlobal

class MultiAgentBEVFusion:
    """
    多车辆BEV感知融合系统
    """
    def __init__(self, bev_config, local_bev_config):
        """
        参数:
            bev_config: BEV配置
        """
        self.bev_config = bev_config
        self.local_bev_config = local_bev_config
        self.global_bev_range = bev_config['area_extents']
        self.voxel_size = bev_config['voxel_size']
        self.grid_size = bev_config['grid_size']
        
        # 为每个车辆分配不同的颜色
        self.agent_colors = [
            (1.0, 0.0, 0.0),  # 红色
            (0.0, 1.0, 0.0),  # 绿色
            (0.0, 0.0, 1.0),  # 蓝色
            (1.0, 1.0, 0.0),  # 黄色
            (1.0, 0.0, 1.0),  # 品红
            (0.0, 1.0, 1.0),  # 青色
        ]
        
    def fuse_multi_agent_bev(self, agents_data):
        """
        融合多个车辆的BEV数据
        
        参数:
            agents_data: List[Dict]
                [{
                    'agent_id': agent_config['agent_id'],
                    'camera_params': agent_config['camera_params'],
                    'ego_params': agent_config['ego_params'],
                    'camera_global_position': projector.camera_global_position,
                    'image_path': agent_config['image_path'],
                    'detections': detections,
                    'bev_confidence_global': bev_confidence_global,
                    'bev_coverage_global': bev_coverage_global,
                    'bev_confidence_local': bev_confidence_local,
                    'bev_coverage_local': bev_coverage_local,
                    'projected_positions': projected_positions
                }, ...]
        
        返回:
            fused_bev: Dict
                - confidence_map: 融合后的置信度地图
                - agent_contribution: 每个agent的贡献地图
                - all_projections: 所有投影点
        """
        H, W = self.grid_size
        
        # 初始化融合地图
        fused_confidence = np.zeros((H, W))
        fused_coverage = np.zeros((H, W))
        area_vehicle_counts = np.zeros((H, W))
        
        # 每个agent的贡献地图（用于可视化）
        agent_contribution_maps = []
        
        # 收集所有投影点
        all_projections = []
        
        for agent_data in agents_data:
            agent_id = agent_data['agent_id']
            bev_confidence = agent_data['bev_confidence_global']
            bev_coverage = agent_data['bev_coverage_global']
            projected_positions = agent_data['projected_positions']
            
            # 融合策略1: 取最大值（最乐观）
            fused_confidence = np.maximum(fused_confidence, bev_confidence)
            
            # 融合策略2: 覆盖范围取并集
            fused_coverage = np.maximum(fused_coverage, bev_coverage)

            if 'local_info' in agent_data:
                local_info = agent_data['local_info']
                
                # 获取车辆在全局坐标系中的位置（原始坐标）
                if local_info['center_type'] == 'vehicle':
                    vehicle_x_global = local_info['center_pos_global'][0]
                    vehicle_y_global_orig = local_info['center_pos_global'][1]  # 原始Y
                else:
                    # 如果local_info是以相机为中心，需要获取车辆位置
                    vehicle_x_global = agent_data['ego_params']['translation'][0]
                    vehicle_y_global_orig = agent_data['ego_params']['translation'][1]  # 原始Y
                
                # 获取车辆的Z坐标
                vehicle_z_global = agent_data['ego_params']['translation'][2]
                
                # ⭐ 计算修改后的Y坐标（与 project_detections_to_bev 一致）
                vehicle_y_global_bev = vehicle_z_global + vehicle_y_global_orig
                
                # 获取局部BEV的范围（相对于车辆）
                local_config = local_info['local_config']
                local_x_min, local_x_max = local_config['area_extents'][0]  # 例如 [-32, 32]
                local_y_min, local_y_max = local_config['area_extents'][1]  # 例如 [-32, 32]
                
                print(f"  车辆原始全局位置: X={vehicle_x_global:.2f}, Y={vehicle_y_global_orig:.2f}, Z={vehicle_z_global:.2f}")
                print(f"  车辆BEV全局位置: Y_bev={vehicle_y_global_bev:.2f}")
                print(f"  局部BEV范围（相对）: X[{local_x_min}, {local_x_max}], Y[{local_y_min}, {local_y_max}]")
                
                # ⭐ 关键：根据坐标系定义计算全局范围
                # 在 project_detections_to_bev 中：
                # - world_pos_global[1] = world_pos_global[2] + world_pos_global[1]  # Y_bev = Z + Y
                # - BEV X轴对应全局的Y_bev（即 Z+Y）
                # - BEV Y轴对应全局的Z
                
                # 所以：
                # 局部BEV的 X范围 [local_x_min, local_x_max] 对应全局的 Y_bev 范围
                # 局部BEV的 Y范围 [local_y_min, local_y_max] 对应全局的 Z 范围
                
                # 计算全局BEV网格中的范围
                # BEV网格的X轴对应 Y_bev
                global_y_bev_min = vehicle_y_global_bev + local_x_min
                global_y_bev_max = vehicle_y_global_bev + local_x_max
                
                # BEV网格的Y轴对应 Z
                global_z_min = vehicle_z_global + local_y_min
                global_z_max = vehicle_z_global + local_y_max
                
                print(f"  全局Y_bev范围: [{global_y_bev_min:.2f}, {global_y_bev_max:.2f}]")
                print(f"  全局Z范围: [{global_z_min:.2f}, {global_z_max:.2f}]")
                
                # 转换为网格索引
                # self.global_bev_range[0] 对应 Y_bev 轴的范围
                # self.global_bev_range[1] 对应 Z 轴的范围
                grid_x_min = int((global_y_bev_min - self.global_bev_range[0][0]) / self.voxel_size[0])
                grid_x_max = int((global_y_bev_max - self.global_bev_range[0][0]) / self.voxel_size[0])
                
                grid_y_min = int((global_z_min - self.global_bev_range[1][0]) / self.voxel_size[1])
                grid_y_max = int((global_z_max - self.global_bev_range[1][0]) / self.voxel_size[1])
                
                # 边界处理
                grid_x_min = max(0, grid_x_min)
                grid_x_max = min(W, grid_x_max)
                grid_y_min = max(0, grid_y_min)
                grid_y_max = min(H, grid_y_max)
                
                print(f"  网格范围: X[{grid_x_min}, {grid_x_max}], Y[{grid_y_min}, {grid_y_max}]")
                
                # 在该范围内的所有点 +1
                if grid_x_max > grid_x_min and grid_y_max > grid_y_min:
                    area_vehicle_counts[grid_y_min:grid_y_max, grid_x_min:grid_x_max] += 1
                    
                    coverage_pixels = (grid_y_max - grid_y_min) * (grid_x_max - grid_x_min)
                    print(f"  覆盖像素数: {coverage_pixels}")
                else:
                    print(f"  ⚠️ 警告：网格范围无效，跳过")
            
            else:
                print(f"  ⚠️ 警告：车辆 {agent_id} 没有 local_info，跳过 area_vehicle_counts 统计")
        
            
            # 记录每个agent的贡献
            agent_contribution_maps.append({
                'agent_id': agent_id,
                'confidence': bev_confidence,
                'coverage': bev_coverage
            })
            
            # 添加agent信息到投影点
            for proj in projected_positions:
                proj_with_agent = proj.copy()
                proj_with_agent['agent_id'] = agent_id
                proj_with_agent['color'] = self.agent_colors[agent_id % len(self.agent_colors)]
                all_projections.append(proj_with_agent)
        
        
        return {
            'fused_confidence': fused_confidence,
            'fused_coverage': fused_coverage,
            "area_vehicle_counts": area_vehicle_counts,
            'agent_contributions': agent_contribution_maps,
            'all_projections': all_projections
        }
    
    def visualize_multi_agent_bev(self, agents_data, fused_result, save_path='multi_agent_bev.png'):
        """
        可视化多车辆BEV融合结果
        
        布局：
        - 第一行：各个agent的原始图像
        - 第二行：各个agent的BEV地图
        - 第三行：融合后的BEV地图（大图）
        """
        num_agents = len(agents_data)
        print(f"✓ 开始可视化 {num_agents} 辆车的BEV融合结果...")
        
        # 创建子图
        fig = plt.figure(figsize=(6*num_agents, 18))
        gs = fig.add_gridspec(3, max(num_agents, 2), height_ratios=[1, 1, 1.5])
        
        # ========== 第一行：原始图像 ==========
        for i, agent_data in enumerate(agents_data):
            ax = fig.add_subplot(gs[0, i])
            
            # 读取并显示图像
            image = cv2.imread(agent_data['image_path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 绘制检测框
            for det in agent_data['detections']:
                x1, y1, x2, y2 = det['box']
                rect = Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, 
                               color=self.agent_colors[agent_data['agent_id'] % len(self.agent_colors)],
                               linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1-10, 
                       f"{det['label']}: {det['score']:.2f}",
                       color=self.agent_colors[agent_data['agent_id'] % len(self.agent_colors)],
                       fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.imshow(image_rgb)
            ax.set_title(f'Agent {agent_data["agent_id"]} - Camera View', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # ========== 第二行：各agent的BEV ==========
        for i, agent_data in enumerate(agents_data):
            ax = fig.add_subplot(gs[1, i])
            
            # 显示该agent的BEV置信度地图
            im = ax.imshow(
                agent_data['bev_confidence_local'],
                cmap='hot',
                origin='lower',
                extent=[
                    self.local_bev_config['area_extents'][0][0],
                    self.local_bev_config['area_extents'][0][1],
                    self.local_bev_config['area_extents'][1][0],
                    self.local_bev_config['area_extents'][1][1]
                ],
                alpha=0.8
            )
            
            # 标记相机位置
            cam_x = agent_data['camera_params']['translation'][0]
            cam_y = agent_data['camera_params']['translation'][1] # local coords
            ax.plot(cam_x, cam_y, '*',
                   color=self.agent_colors[agent_data['agent_id'] % len(self.agent_colors)],
                   markersize=10,
                   markeredgecolor='white',
                   markeredgewidth=2,
                   label=f'Agent {agent_data["agent_id"]}')
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title(f'Agent {agent_data["agent_id"]} - BEV Map', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # ========== 第三行：融合BEV（占满整行） ==========
        ax_fused_conf = fig.add_subplot(gs[2, :num_agents//2 if num_agents > 1 else 1])
        ax_fused_cov = fig.add_subplot(gs[2, num_agents//2 if num_agents > 1 else 0:])
        
        # 融合置信度地图
        im1 = ax_fused_conf.imshow(
            fused_result['fused_confidence'],
            cmap='hot',
            origin='lower',
            extent=[
                self.bev_config['area_extents'][0][0],
                self.bev_config['area_extents'][0][1],
                self.bev_config['area_extents'][1][0],
                self.bev_config['area_extents'][1][1]
            ]
        )
        
        # 标记所有相机位置
        for i, agent_data in enumerate(agents_data):
            # cam_x = agent_data['camera_params']['translation'][0]
            # cam_y = agent_data['camera_params']['translation'][1]
            cam_x = agent_data['camera_global_position'][0]
            cam_y = agent_data['camera_global_position'][1]
            ax_fused_conf.plot(cam_x, cam_y, '*',
                             color=self.agent_colors[agent_data['agent_id'] % len(self.agent_colors)],
                             markersize=10,
                             markeredgecolor='white',
                             markeredgewidth=2,
                             label=f'Agent {agent_data["agent_id"]}')
        
        # 标注所有投影点
        for proj in fused_result['all_projections']:
            wx, wy = proj['world_pos'][0], proj['world_pos'][1]
            ax_fused_conf.plot(wx, wy, 'o',
                             color=proj['color'],
                             markersize=8,
                             markeredgecolor='white',
                             markeredgewidth=1.5)
            
            # 添加标签
            ax_fused_conf.text(wx+0.5, wy+0.5,
                             f"{proj['label']}",
                             fontsize=8,
                             color='white',
                             bbox=dict(boxstyle='round', 
                                     facecolor=proj['color'],
                                     alpha=0.7))
        
        ax_fused_conf.set_xlabel('X (meters)', fontsize=12)
        ax_fused_conf.set_ylabel('Y (meters)', fontsize=12)
        ax_fused_conf.set_title('Fused Confidence Map (All Agents)', 
                                fontsize=14, fontweight='bold')
        ax_fused_conf.grid(True, alpha=0.3)
        ax_fused_conf.legend(loc='upper right', fontsize=10)
        plt.colorbar(im1, ax=ax_fused_conf, fraction=0.046, pad=0.04)
        
        # 融合覆盖地图
        im2 = ax_fused_cov.imshow(
            fused_result['fused_coverage'],
            cmap='Blues',
            origin='lower',
            extent=[
                self.bev_config['area_extents'][0][0],
                self.bev_config['area_extents'][0][1],
                self.bev_config['area_extents'][1][0],
                self.bev_config['area_extents'][1][1]
            ]
        )
        
        # 标记相机和检测
        for i, agent_data in enumerate(agents_data):
            # cam_x = agent_data['camera_params']['translation'][0]
            # cam_y = agent_data['camera_params']['translation'][1]
            cam_x = agent_data['camera_global_position'][0]
            cam_y = agent_data['camera_global_position'][1]
            ax_fused_cov.plot(cam_x, cam_y, '*',
                            color=self.agent_colors[agent_data['agent_id'] % len(self.agent_colors)],
                            markersize=20,
                            markeredgecolor='white',
                            markeredgewidth=2)
        
        ax_fused_cov.set_xlabel('X (meters)', fontsize=12)
        ax_fused_cov.set_ylabel('Y (meters)', fontsize=12)
        ax_fused_cov.set_title('Fused Coverage Map (All Agents)', 
                              fontsize=14, fontweight='bold')
        ax_fused_cov.grid(True, alpha=0.3)
        plt.colorbar(im2, ax=ax_fused_cov, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化结果已保存到: {save_path}")
        plt.show()
        
    def generate_comparison_stats(self, agents_data, fused_result):
        """
        生成融合前后的统计对比
        """
        print("\n" + "="*70)
        print("多车辆协同感知统计")
        print("="*70)
        
        # 单车统计
        print("\n单车感知统计:")
        print("-"*70)
        for agent_data in agents_data:
            agent_id = agent_data['agent_id']
            num_detections = len(agent_data['detections'])
            coverage_ratio = (agent_data['bev_coverage_local'] > 0).sum() / agent_data['bev_coverage_local'].size
            avg_confidence = agent_data['bev_confidence_local'][agent_data['bev_confidence_local'] > 0].mean() if (agent_data['bev_confidence_local'] > 0).any() else 0
            
            print(f"Agent {agent_id}:")
            print(f"  检测数量: {num_detections}")
            # print(f"  BEV覆盖率: {coverage_ratio*100:.2f}%")
            # print(f"  平均置信度: {avg_confidence:.3f}")
            print()
        
        # 融合后统计
        # print("融合后感知统计:")
        print("-"*70)
        total_projections = len(fused_result['all_projections'])
        fused_coverage_ratio = (fused_result['fused_coverage'] > 0).sum() / fused_result['fused_coverage'].size
        fused_avg_confidence = fused_result['fused_confidence'][fused_result['fused_confidence'] > 0].mean() if (fused_result['fused_confidence'] > 0).any() else 0
        
        print(f"总检测数量: {total_projections}")
        # print(f"融合BEV覆盖率: {fused_coverage_ratio*100:.2f}%")
        # print(f"融合平均置信度: {fused_avg_confidence:.3f}")
        
        # 计算提升
        single_avg_coverage = np.mean([
            (agent_data['bev_coverage_local'] > 0).sum() / agent_data['bev_coverage_local'].size
            for agent_data in agents_data
        ])
        
        coverage_improvement = (fused_coverage_ratio - single_avg_coverage) / single_avg_coverage * 100
        
        # print(f"\n协同感知提升:")
        # print(f"  覆盖率提升: {coverage_improvement:.1f}%")
        print("="*70)


    def visualize_area_vehicle_counts(self, area_vehicle_counts, agents_data, save_path='area_vehicle_counts.png'):
        """
        可视化每个区域被多少辆车覆盖
        
        参数:
            area_vehicle_counts: (H, W) 每个点的车辆覆盖数
            agents_data: 车辆数据列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：车辆覆盖数热力图
        ax1 = axes[0]
        
        extent = [
            self.global_bev_range[0][0],
            self.global_bev_range[0][1],
            self.global_bev_range[1][0],
            self.global_bev_range[1][1]
        ]
        
        im1 = ax1.imshow(area_vehicle_counts, cmap='YlOrRd', origin='lower',
                        extent=extent, alpha=0.8, vmin=0, vmax=len(agents_data))
        
        # 绘制每个车辆的位置和局部范围
        for agent_idx, agent_data in enumerate(agents_data):
            veh_x = agent_data['ego_params']['translation'][0]
            veh_y = agent_data['ego_params']['translation'][1]
            
            # 车辆位置
            ax1.plot(veh_y, veh_x, 'o', markersize=12,
                    color=self.agent_colors[agent_idx % len(self.agent_colors)],
                    markeredgecolor='white', markeredgewidth=2,
                    label=f"Vehicle {agent_data['agent_id']}")
            
            # 局部BEV范围框
            if 'local_info' in agent_data:
                local_info = agent_data['local_info']
                local_config = local_info['local_config']
                local_x_range = local_config['area_extents'][0]
                local_y_range = local_config['area_extents'][1]
                
                # 全局坐标中的范围（需要根据你的坐标系调整）
                box_y_min = veh_y + local_x_range[0]
                box_y_max = veh_y + local_x_range[1]
                box_x_min = local_y_range[0]
                box_x_max = local_y_range[1]
                
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (box_y_min, box_x_min),
                    box_y_max - box_y_min,
                    box_x_max - box_x_min,
                    linewidth=2,
                    edgecolor=self.agent_colors[agent_idx % len(self.agent_colors)],
                    facecolor='none',
                    linestyle='--',
                    alpha=0.7
                )
                ax1.add_patch(rect)
        
        ax1.set_xlabel('Global Y (meters)', fontsize=12)
        ax1.set_ylabel('Global X (meters)', fontsize=12)
        ax1.set_title('Area Vehicle Coverage Counts', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        plt.colorbar(im1, ax=ax1, label='Number of Vehicles')
        
        # 右图：统计直方图
        ax2 = axes[1]
        
        counts = area_vehicle_counts.flatten()
        unique_counts = np.unique(counts)
        
        hist_data = []
        labels = []
        for count in unique_counts:
            if count == 0:
                continue
            num_pixels = (counts == count).sum()
            hist_data.append(num_pixels)
            labels.append(f'{int(count)} vehicles')
        
        ax2.bar(range(len(hist_data)), hist_data, color='steelblue', alpha=0.7)
        ax2.set_xticks(range(len(hist_data)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Number of Grid Cells', fontsize=12)
        ax2.set_title('Coverage Statistics', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(hist_data):
            ax2.text(i, v + max(hist_data)*0.02, str(v),
                    ha='center', va='bottom', fontsize=10)
        
        # 添加统计信息
        total_covered = (area_vehicle_counts > 0).sum()
        multi_covered = (area_vehicle_counts > 1).sum()
        coverage_ratio = total_covered / area_vehicle_counts.size * 100
        overlap_ratio = multi_covered / max(total_covered, 1) * 100
        
        stats_text = f"""
        Coverage Statistics:
        - Total grid cells: {area_vehicle_counts.size}
        - Covered cells: {total_covered} ({coverage_ratio:.1f}%)
        - Multi-covered: {multi_covered} ({overlap_ratio:.1f}% of covered)
        - Max coverage: {area_vehicle_counts.max()} vehicles
        """
        
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Area vehicle counts 可视化已保存: {save_path}")
        plt.show()

if __name__ == "__main__":
    # =============== 1. 准备多车辆数据 ===============

    # 车辆配置
    agents_config = [
        {
            'agent_id': 1,
            'image_path': '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_FRONT_id_1/scene_5_000006.jpg',
            'camera_params': {
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
            },
            'ego_params': {
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
        },
        {
            'agent_id': 2,
            'image_path': '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_FRONT_id_2/scene_5_000006.jpg',  # 第二辆车的图像
            'camera_params': {
                "translation": [
                    1.5,
                    -0.0,
                    1.6171096563339233
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
            },
            'ego_params': {
                "rotation": [
                    0.7178262421324725,
                    0.0,
                    -0.0,
                    -0.6962222964728816
                ],
                "translation": [
                    -84.80706787109375,
                    152.3368377685547,
                    0.0
                ]
            }
        },
        {
            'agent_id': 3,
            'image_path': '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_FRONT_id_3/scene_5_000006.jpg',  # 第三辆车的图像
            'camera_params': {
                "translation": [
                    1.5,
                    -0.0,
                    1.6672759056091309
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
            },
            'ego_params': {
                "rotation": [
                    0.0025357261144973733,
                    0.0,
                    0.0,
                    0.9999967850413681
                ],
                "translation": [
                    -55.24314498901367,
                    2.900322198867798,   
                    0.0
                ]           
            }
        }
    ]

    # BEV配置
    bev_config = {
        'area_extents': [[-256, 256], [-256, 256]],  # 覆盖64m x 64m区域
        'voxel_size': [0.25, 0.25],              # 每个网格0.25m
        'grid_size': [2048, 2048]                  # 256x256网格
    }

    local_bev_config = {
        'area_extents': [[-32, 32], [-32, 32]],  
        'voxel_size': [0.25, 0.25],              
        'grid_size': [256, 256]                  
    }

    # =============== 2. 处理每个车辆的数据 ===============

    detector = FasterRCNNDetector(device='cuda', conf_threshold=0.7)
    agents_data = []

    for agent_config in agents_config:
        print(f"\n处理 Agent {agent_config['agent_id']}...")
        
        # 创建投影器
        projector = ImageToBEVProjectorWithGlobal(agent_config['camera_params'], agent_config['ego_params'], bev_config, local_bev_config)
        
        # 读取图像
        image = cv2.imread(agent_config['image_path'])
        
        if image is None:
            print(f"  ⚠️  无法读取图像: {agent_config['image_path']}")
            continue
        
        # 目标检测
        detections = detector.detect(image)
        print(f"  检测到 {len(detections)} 个物体")
        
        # 投影到BEV
        bev_confidence_global, bev_coverage_global, projected_positions, \
            bev_confidence_local, bev_coverage_local, local_info = projector.project_detections_to_bev(
            detections=detections,
            method='depth_estimation',
            local_bev_config=local_bev_config
            # 或 'depth_estimation', 'ground_plane'
        )
        
        # 保存该agent的数据
        agents_data.append({
            'agent_id': agent_config['agent_id'],
            'camera_params': agent_config['camera_params'],
            'ego_params': agent_config['ego_params'],
            'camera_global_position': projector.camera_global_position,
            'image_path': agent_config['image_path'],
            'detections': detections,
            'bev_confidence_global': bev_confidence_global,
            'bev_coverage_global': bev_coverage_global,
            'bev_confidence_local': bev_confidence_local,
            'bev_coverage_local': bev_coverage_local,
            'projected_positions': projected_positions,
            'local_info': local_info
        })

    # =============== 3. 多车辆BEV融合 ===============

    fusion_system = MultiAgentBEVFusion(bev_config, local_bev_config)

    # 融合所有车辆的BEV数据
    fused_result = fusion_system.fuse_multi_agent_bev(agents_data)

    # =============== 4. 可视化 ===============

    fusion_system.visualize_multi_agent_bev(
        agents_data,
        fused_result,
        save_path='multi_agent_bev_fusion.png'
    )

    # =============== 5. 统计分析 ===============

    fusion_system.generate_comparison_stats(agents_data, fused_result)

    fusion_system.visualize_area_vehicle_counts(area_vehicle_counts=fused_result['area_vehicle_counts'],
                                  agents_data=agents_data,
                                  save_path='area_vehicle_counts.png')