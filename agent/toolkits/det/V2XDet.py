from .FastCNNDet import FasterRCNNDetector
from .utils import *
import numpy as np

import logging
LOG = logging.getLogger(__name__)

class PracticalQualityEvaluator:
    """
    实用的质量评估器
    结合预训练模型的优势和实际需求
    """
    def __init__(self, use_lightweight=True):
        if use_lightweight:
            # 使用YOLOv8 (更快，推荐)
            print("Using YOLOv8 for object detection")
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.pt')
            self.use_yolo = True

        else:
            # 使用Faster R-CNN (更准确)
            print("Using Faster R-CNN for object detection")
            self.detector = FasterRCNNDetector()
            self.use_yolo = False
    
    def evaluate_image_quality(self, image):
        """
        从图像评估感知质量
        
        返回质量指标:
        - confidence_map: 检测置信度分布
        - coverage_map: 覆盖范围
        - object_count: 检测到的物体数量
        """
        if self.use_yolo:
            results = self.detector(image, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            print(f"boxes: {boxes}, scores: {scores}, classes: {classes}")
            plot_detections(image, results.boxes.data.cpu().numpy())
        else:
            detections = self.detector.detect(image)
            boxes = [d['box'] for d in detections]
            scores = [d['score'] for d in detections]
            classes = [d['label'] for d in detections]
            plot_detections(image, detections)
        
        # 创建质量地图
        H, W = image.shape[:2]
        confidence_map = np.zeros((H, W))
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            confidence_map[y1:y2, x1:x2] = np.maximum(
                confidence_map[y1:y2, x1:x2],
                score
            )
        
        return {
            'confidence_map': confidence_map,
            'num_objects': len(boxes),
            'avg_confidence': np.mean(scores) if len(scores) > 0 else 0,
            'detections': {'boxes': boxes, 'scores': scores, 'classes': classes}
        }

# 使用
img_file = '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_BACK_id_2/scene_5_000006.jpg'
import cv2
img_file = '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_BACK_id_2/scene_5_000006.jpg'
image = cv2.imread(img_file)
evaluator = PracticalQualityEvaluator(use_lightweight=True)
quality = evaluator.evaluate_image_quality(image)
print(f"检测到 {quality['num_objects']} 个物体")
print(f"平均置信度: {quality['avg_confidence']:.3f}")