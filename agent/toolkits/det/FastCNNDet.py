import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2

class FasterRCNNDetector:
    def __init__(self, device='cuda', conf_threshold=0.5):
        self.device = device
        self.conf_threshold = conf_threshold
        
        # 加载模型
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(device)
        self.model.eval()
        
        # 自动驾驶相关类别
        self.relevant_classes = {
            1: 'person',
            2: 'bicycle', 
            3: 'car',
            4: 'motorcycle',
            6: 'bus',
            8: 'truck'
        }
    
    def detect(self, image):
        """
        检测图像中的物体
        
        参数:
            image: numpy array (H, W, 3) BGR格式，范围[0, 255]
        
        返回:
            detections: List[Dict]
                - box: (4,) [x1, y1, x2, y2]
                - score: float
                - label: int
                - class_name: str
        """
        # 预处理
        image_tensor = self._preprocess(image)
        
        # 推理
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # 后处理
        detections = self._postprocess(predictions)
        
        return detections
    
    def _preprocess(self, image):
        """图像预处理"""
        # 转换为RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor，归一化到[0, 1]
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)
        
        return image_tensor
    
    def _postprocess(self, predictions):
        """后处理：过滤低置信度和无关类别"""
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        detections = []
        
        for box, score, label in zip(boxes, scores, labels):
            # 过滤低置信度
            if score < self.conf_threshold:
                continue
            
            # 只保留相关类别
            if label not in self.relevant_classes:
                continue
            
            detections.append({
                'box': box,
                'score': score,
                'label': label,
                'class_name': self.relevant_classes[label]
            })
        
        return detections

if __name__ == "__main__":
    detector = FasterRCNNDetector(device='cuda', conf_threshold=0.7)

    import cv2
    img_file = '/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_FRONT_id_1/scene_5_000006.jpg'
    image = cv2.imread(img_file)
    detections = detector.detect(image)

    for det in detections:
        # print(f"{det['class_name']}: {det['score']:.3f} at {det['box']}")
        # 绘制边界框
        x1, y1, x2, y2 = det['box'].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{det['class_name']} {det['score']:.2f}", 
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite('detected.jpg', image)