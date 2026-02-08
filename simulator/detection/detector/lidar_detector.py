from .base_detector import BaseDetector

class LidarDetector(BaseDetector):
    def __init__(self, world, config):
        super().__init__(world, config)