
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

g_camera_params = {
    "translation": [
        -12.906107092526855,
        -10.834492683410645,
        7.0
    ],
    "rotation": [
        6.258226974788741e-17,
        -6.258226974788741e-17,
        0.8870108331782217,
        -0.46174861323503386
    ],
    "camera_intrinsic": [
        [1142.5184053936916, 0.0, 800.0],
        [0.0, 1142.5184053936916, 450.0],
        [0.0, 0.0, 1.0]
    ]
}

def plot_detections(image, detections, output_path=None):
    import cv2
    for det in detections:
        print(f"{det['class_name']}: {det['score']:.3f} at {det['box']}")
        x1, y1, x2, y2 = map(int, det['box'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{det['label']} {det['score']:.2f}", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imwrite('detected.jpg', image)
    return image


def visualize_detections(detections, image_width=800, image_height=600, output_path=None):
    """
    可视化目标检测结果
    
    参数:
        detections: list of dict，每个字典包含 'box' 和 'score' 字段
                   box: [x1, y1, x2, y2] 格式的坐标
                   score: 置信度分数 (0-1)
        image_width: 画布宽度，默认800
        image_height: 画布高度，默认600
        output_path: 保存图像的路径，如果为None则只显示
    """
    
    # 创建图像和坐标轴
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # 设置空白背景
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)  # 反向y轴以匹配图像坐标系
    ax.set_aspect('equal')
    
    # 设置背景颜色
    ax.set_facecolor('white')
    
    # 添加网格（可选）
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 为每个检测结果绘制矩形框和置信度
    for i, detection in enumerate(detections):
        box = detection['box']  # [x1, y1, x2, y2]
        score = detection['score']
        
        x1, y1, x2, y2 = box
        
        # 计算矩形的宽度和高度
        width = x2 - x1
        height = y2 - y1
        
        # 绘制矩形框
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            label='Detection' if i == 0 else ''
        )
        ax.add_patch(rect)
        
        # 在矩形上方显示置信度分数
        score_text = f'Score: {score:.3f}'
        ax.text(
            x1, y1 - 10,  # 在矩形上方
            score_text,
            fontsize=10,
            color='red',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )
        
        # 可选：在矩形中心显示检测ID
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        ax.text(
            center_x, center_y,
            f'#{i+1}',
            fontsize=9,
            color='blue',
            ha='center',
            va='center',
            fontweight='bold'
        )
    
    # 设置标签和标题
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Object Detection Results', fontsize=14, fontweight='bold')
    
    # 添加图例
    if detections:
        ax.text(
            0.02, 0.98,
            f'Total Detections: {len(detections)}',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {output_path}")
    else:
        # plt.show()
        pass
    
    return fig, ax


# 使用示例
if __name__ == "__main__":
    # 示例数据
    example_detections = [
        {'box': [100, 150, 200, 250], 'score': 0.92},
        {'box': [300, 180, 420, 350], 'score': 0.87},
        {'box': [500, 200, 650, 400], 'score': 0.95},
        {'box': [150, 400, 280, 550], 'score': 0.78},
    ]
    
    # 可视化
    visualize_detections(
        example_detections,
        image_width=800,
        image_height=600,
        output_path='detection_results.png'
    )