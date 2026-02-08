
from scipy.spatial.transform import Rotation as R

def carla_rotation_to_wxyz(carla_rotation):
    r = R.from_euler(
        'xyz',
        [carla_rotation.roll,
         carla_rotation.pitch,
         carla_rotation.yaw],
        degrees=True
    )
    qx, qy, qz, qw = r.as_quat()
    return [qw, qx, qy, qz]   # (w, x, y, z)

g_camera_params = {
    "translation": [
        1.5,
        0.0,
        2.0
    ],
    "rotation": [
        1,
        0,
        0,
        0
    ],
    "camera_intrinsic": [
        [1142.5184053936916, 0.0, 800.0],
        [0.0, 1142.5184053936916, 450.0],
        [0.0, 0.0, 1.0]
    ]
}

# 一般小轿车白名单（基于CARLA 33个可用蓝图）
REGULAR_SEDAN_TYPES = {
    'vehicle.nissan.micra',
    'vehicle.toyota.prius',
    'vehicle.mini.cooper_s',
    'vehicle.mini.cooper_s_2021',
    'vehicle.audi.a2',
    'vehicle.audi.tt',
    'vehicle.citroen.c3',
    'vehicle.seat.leon',
    'vehicle.tesla.model3',
    'vehicle.lincoln.mkz_2017',
    'vehicle.lincoln.mkz_2020',
    'vehicle.chevrolet.impala',
    'vehicle.dodge.charger_2020',
    'vehicle.ford.mustang',
    'vehicle.mercedes.coupe',
    'vehicle.mercedes.coupe_2020',
    'vehicle.bmw.grandtourer',
    'vehicle.audi.etron',
    'vehicle.micro.microlino',
}


def is_regular_sedan(vehicle) -> bool:
    """
    判断vehicle是否为一般小轿车
    
    Args:
        vehicle: CARLA Actor对象
        
    Returns:
        bool: True表示是小轿车，False表示不是
        
    Example:
        >>> if is_regular_sedan(vehicle):
        ...     print("这是一般小轿车")
    """
    return vehicle.type_id in REGULAR_SEDAN_TYPES
