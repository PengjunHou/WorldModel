import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 加载预训练模型
detector = fasterrcnn_resnet50_fpn(pretrained=True)
detector.eval()  # 设置为评估模式

# 使用
image = torch.rand(1, 3, 800, 800)  # (batch, channels, height, width)
with torch.no_grad():
    predictions = detector(image)

# 输出格式
print(predictions[0].keys())  
# dict_keys(['boxes', 'labels', 'scores'])
# - boxes: (N, 4) [x1, y1, x2, y2] 边界框坐标
# - labels: (N,) 类别ID (COCO 80类)
# - scores: (N,) 置信度分数 [0, 1]