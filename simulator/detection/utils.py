from enum import Enum

from .detector import BaseDetector, CameraDetector, LidarDetector


class DetectorType(Enum):
    """User-defiend data sources"""

    RGB_CAMERA = "camera"
    LIDAR = "lidar"
    # COLLISION = "collision"
    # BIRDEYE = "birdeye"
    # MESSAGE = "message"
    # SPECTATOR = "spectator"


DETECTOR_DICT = {
    # DetectorType.BIRDEYE: BirdeyeHandler,
    # DetectorType.MESSAGE: MessageHandler,
    DetectorType.RGB_CAMERA: CameraDetector,
    DetectorType.LIDAR: LidarDetector,
    # DetectorType.COLLISION: CollisionHandler,
    # DetectorType.SPECTATOR: SpectatorHandler,
}
