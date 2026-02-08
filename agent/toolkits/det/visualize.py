

def create_interactive_comparison(agents_data, fused_result, bev_config):
    """
    创建交互式对比图：可以切换查看不同agent的贡献
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 初始化显示
    extent = [
        bev_config['area_extents'][0][0],
        bev_config['area_extents'][0][1],
        bev_config['area_extents'][1][0],
        bev_config['area_extents'][1][1]
    ]
    
    # 左图：各agent的独立贡献
    agent_maps = {}
    for agent_data in agents_data:
        agent_id = agent_data['agent_id']
        agent_maps[agent_id] = ax1.imshow(
            agent_data['bev_confidence'],
            cmap='hot',
            origin='lower',
            extent=extent,
            alpha=0.5,
            label=f'Agent {agent_id}'
        )
    
    # 右图：融合结果
    im2 = ax2.imshow(
        fused_result['fused_confidence'],
        cmap='hot',
        origin='lower',
        extent=extent
    )
    
    # 标记相机位置
    for agent_data in agents_data:
        cam_x = agent_data['camera_params']['translation'][0]
        cam_y = agent_data['camera_params']['translation'][1]
        ax1.plot(cam_x, cam_y, '*', markersize=15)
        ax2.plot(cam_x, cam_y, '*', markersize=15)
    
    ax1.set_title('Individual Agent Contributions')
    ax2.set_title('Fused Result')
    
    plt.tight_layout()
    # plt.show()

# 使用
create_interactive_comparison(agents_data, fused_result, bev_config)

def quick_multi_agent_visualization(agents_data, bev_config):
    """
    快速版本：适合2-3个车辆的简单可视化
    """
    num_agents = len(agents_data)
    
    fig, axes = plt.subplots(2, num_agents + 1, figsize=(5*(num_agents+1), 10))
    
    # 第一行：原始图像
    for i, agent_data in enumerate(agents_data):
        image = cv2.imread(agent_data['image_path'])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        axes[0, i].imshow(image_rgb)
        axes[0, i].set_title(f'Agent {agent_data["agent_id"]}')
        axes[0, i].axis('off')
    
    # 第二行：BEV地图
    extent = [
        bev_config['area_extents'][0][0],
        bev_config['area_extents'][0][1],
        bev_config['area_extents'][1][0],
        bev_config['area_extents'][1][1]
    ]
    
    # 各agent的BEV
    for i, agent_data in enumerate(agents_data):
        axes[1, i].imshow(agent_data['bev_confidence'], 
                         cmap='hot', origin='lower', extent=extent)
        axes[1, i].set_title(f'Agent {agent_data["agent_id"]} BEV')
    
    # 融合后的BEV
    fused_confidence = np.zeros(agents_data[0]['bev_confidence'].shape)
    for agent_data in agents_data:
        fused_confidence = np.maximum(fused_confidence, agent_data['bev_confidence'])
    
    axes[1, -1].imshow(fused_confidence, cmap='hot', origin='lower', extent=extent)
    axes[1, -1].set_title('Fused BEV')
    
    # 标记所有相机位置
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    for i, agent_data in enumerate(agents_data):
        cam_x = agent_data['camera_params']['translation'][0]
        cam_y = agent_data['camera_params']['translation'][1]
        axes[1, -1].plot(cam_x, cam_y, '*', 
                        color=colors[i % len(colors)],
                        markersize=15,
                        label=f'Agent {agent_data["agent_id"]}')
    
    axes[1, -1].legend()
    
    plt.tight_layout()
    plt.savefig('quick_multi_agent_bev.png', dpi=150)
    # plt.show()

# 使用
quick_multi_agent_visualization(agents_data, bev_config)
```

## 五、输出效果

运行后你会得到：

1. **第一行**：各个车辆的相机视图（带检测框）
2. **第二行**：各个车辆独立的BEV地图  
3. **第三行**：融合后的BEV地图（大图），显示所有车辆的相机位置和检测结果

同时还会打印统计信息：
```
多车辆协同感知统计
======================================================================

单车感知统计:
----------------------------------------------------------------------
Agent 0:
  检测数量: 5
  BEV覆盖率: 12.34%
  平均置信度: 0.856

Agent 1:
  检测数量: 7
  BEV覆盖率: 15.67%
  平均置信度: 0.823

融合后感知统计:
----------------------------------------------------------------------
总检测数量: 12
融合BEV覆盖率: 23.45%
融合平均置信度: 0.891

协同感知提升:
  覆盖率提升: 67.8%
======================================================================