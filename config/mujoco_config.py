# MuJoCo配置文件
MUJOCO_CONFIG = {
    # 仿真参数
    "simulation": {
        "timestep": 0.002,  # 仿真时间步长 (秒)
        "gravity": [0, 0, -9.81],  # 重力加速度
        "integrator": "Euler",  # 积分器类型
    },
    
    # 默认机器人模型
    "robot": {
        "num_joints": 12,
        "xml_path": None,  # 使用内置的默认模型
        "joint_limits": {
            "min": -3.14159,
            "max": 3.14159
        },
        "actuator_limits": {
            "min": -1.0,
            "max": 1.0
        }
    },
    
    # 感知参数
    "perception": {
        "image_size": [224, 224],
        "camera_height": 1.5,
        "camera_distance": 2.0,
        "camera_angle": 45.0
    },
    
    # 环境设置
    "environment": {
        "ground_friction": 0.8,
        "ground_restitution": 0.1,
        "enable_contact": True,
        "enable_constraint": True
    },
    
    # 渲染设置
    "rendering": {
        "enable_viewer": True,
        "window_width": 640,
        "window_height": 480,
        "fps": 60
    }
}