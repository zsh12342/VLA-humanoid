"""
Perception Module - 获取RGB图像和机器人状态
支持MuJoCo仿真和真实机器人
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
import cv2
from pathlib import Path

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


class PerceptionModule:
    """感知模块 - 获取机器人观测数据
    
    支持MuJoCo仿真环境和真实机器人，提供统一的观测接口。
    观测数据包括RGB图像和机器人状态（关节角度、速度、IMU、身体位置和姿态）。
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 num_joints: int = 45,
                 simulation_mode: bool = True,
                 model_path: Optional[str] = None):
        """
        初始化感知模块
        
        Args:
            image_size: 图像尺寸 (width, height)
            num_joints: 机器人关节数量
            simulation_mode: 是否为仿真模式
            model_path: MuJoCo模型文件路径
        """
        self.image_size = image_size
        self.num_joints = num_joints
        self.simulation_mode = simulation_mode
        self.model_path = model_path or str(config.MUJOCO["model_path"])
        
        # 内部状态
        self._sim_time = 0
        self._last_joint_angles = None
        self._last_joint_velocities = None
        
        # MuJoCo环境
        self.model = None
        self.data = None
        self.viewer = None
        
        # 初始化仿真环境
        if simulation_mode and MUJOCO_AVAILABLE:
            self._init_mujoco()
        elif simulation_mode and not MUJOCO_AVAILABLE:
            print("Warning: MuJoCo not available, using fallback simulation")
    
    def _init_mujoco(self):
        """初始化MuJoCo环境"""
        try:
            # 加载Kuavo S45模型或创建默认模型
            if Path(self.model_path).exists():
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
                print(f"Loaded MuJoCo model from: {self.model_path}")
            else:
                # 创建简化的Kuavo S45模型
                xml_content = self._create_kuavo_s45_xml()
                self.model = mujoco.MjModel.from_xml_string(xml_content)
                print("Created default Kuavo S45 model")
            
            self.data = mujoco.MjData(self.model)
            
            # 设置初始状态
            mujoco.mj_resetData(self.model, self.data)
            
            # 设置仿真参数
            self.model.opt.timestep = config.MUJOCO["timestep"]
            self.model.opt.integrator = config.MUJOCO["integrator"]
            
            print(f"MuJoCo initialized successfully")
            print(f"  Joints: {self.model.nq}")
            print(f"  Actuators: {self.model.nu}")
            print(f"  Timestep: {self.model.opt.timestep}")
            
        except Exception as e:
            print(f"MuJoCo initialization failed: {e}")
            self.simulation_mode = False
    
    def _create_kuavo_s45_xml(self) -> str:
        """创建简化的Kuavo S45 XML模型"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<mujoco model="kuavo_s45">
    <worldbody>
        <!-- 地面 -->
        <geom name="ground" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
        
        <!-- 机器人主体 -->
        <body name="torso" pos="0 0 0.8">
            <joint name="root" type="free"/>
            <geom type="box" size="0.3 0.2 0.4" mass="5.0" rgba="0.2 0.2 0.8 1"/>
            
            <!-- 头部 -->
            <body name="head" pos="0 0 0.5">
                <joint name="neck_joint" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom type="sphere" size="0.15" mass="1.0" rgba="0.8 0.2 0.2 1"/>
                
                <!-- 相机 -->
                <camera name="rgb_camera" pos="0.1 0 0.1" quat="0.7071 0 0.7071 0" fovy="60"/>
            </body>
            
            <!-- 左臂 -->
            <body name="left_shoulder" pos="-0.3 0 0.2">
                <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.05 0.15" mass="0.5" rgba="0.2 0.8 0.2 1"/>
                
                <body name="left_elbow" pos="0 0 -0.2">
                    <joint name="left_elbow" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.04 0.12" mass="0.3" rgba="0.2 0.8 0.2 1"/>
                </body>
            </body>
            
            <!-- 右臂 -->
            <body name="right_shoulder" pos="0.3 0 0.2">
                <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.05 0.15" mass="0.5" rgba="0.2 0.8 0.2 1"/>
                
                <body name="right_elbow" pos="0 0 -0.2">
                    <joint name="right_elbow" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.04 0.12" mass="0.3" rgba="0.2 0.8 0.2 1"/>
                </body>
            </body>
            
            <!-- 左腿 -->
            <body name="left_hip" pos="-0.1 0 -0.5">
                <joint name="left_hip_pitch" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.06 0.3" mass="1.0" rgba="0.8 0.8 0.2 1"/>
                
                <body name="left_knee" pos="0 0 -0.4">
                    <joint name="left_knee" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.05 0.25" mass="0.8" rgba="0.8 0.8 0.2 1"/>
                </body>
            </body>
            
            <!-- 右腿 -->
            <body name="right_hip" pos="0.1 0 -0.5">
                <joint name="right_hip_pitch" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.06 0.3" mass="1.0" rgba="0.8 0.8 0.2 1"/>
                
                <body name="right_knee" pos="0 0 -0.4">
                    <joint name="right_knee" type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.05 0.25" mass="0.8" rgba="0.8 0.8 0.2 1"/>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <motor name="root" joint="root" ctrlrange="-1 1" gear="1 0 0 0 1 0"/>
        <motor name="neck" joint="neck_joint" ctrlrange="-1 1"/>
        <motor name="left_shoulder" joint="left_shoulder_pitch" ctrlrange="-1 1"/>
        <motor name="left_elbow" joint="left_elbow" ctrlrange="-1 1"/>
        <motor name="right_shoulder" joint="right_shoulder_pitch" ctrlrange="-1 1"/>
        <motor name="right_elbow" joint="right_elbow" ctrlrange="-1 1"/>
        <motor name="left_hip" joint="left_hip_pitch" ctrlrange="-1 1"/>
        <motor name="left_knee" joint="left_knee" ctrlrange="-1 1"/>
        <motor name="right_hip" joint="right_hip_pitch" ctrlrange="-1 1"/>
        <motor name="right_knee" joint="right_knee" ctrlrange="-1 1"/>
    </actuator>
    
    <sensor>
        <touch name="touch_sensor" site="torso"/>
        <accelerometer name="accelerometer" site="torso"/>
        <gyro name="gyroscope" site="torso"/>
        <framepos name="body_pos" objtype="body" objname="torso"/>
        <framequat name="body_quat" objtype="body" objname="torso"/>
    </sensor>
</mujoco>"""
    
    def get_observation(self) -> Dict[str, Any]:
        """
        获取机器人观测数据
        
        Returns:
            Dict[str, Any]: 包含图像和状态的字典
                - "image": np.array(shape=(height, width, 3), dtype=np.uint8)
                - "state": Dict containing joint_angles, joint_velocities, imu, body_pos, body_quat
        """
        if self.simulation_mode and MUJOCO_AVAILABLE and hasattr(self, 'model'):
            return self._get_mujoco_observation()
        else:
            return self._get_simulated_observation()
    
    def _get_mujoco_observation(self) -> Dict[str, Any]:
        """获取MuJoCo环境中的观测数据"""
        # 执行仿真步骤
        mujoco.mj_step(self.model, self.data)
        
        # 获取RGB图像
        image = self._render_mujoco_image()
        
        # 获取机器人状态
        state = {
            "joint_angles": self.data.qpos[:self.num_joints].copy(),
            "joint_velocities": self.data.qvel[:self.num_joints].copy(),
            "imu": self._get_mujoco_imu_data(),
            "body_pos": self.data.qpos[:3].copy(),
            "body_quat": self.data.qpos[3:7].copy()
        }
        
        return {
            "image": image,
            "state": state
        }
    
    def _get_simulated_observation(self) -> Dict[str, Any]:
        """获取仿真环境中的观测数据（模拟数据）"""
        # 生成模拟RGB图像
        image = self._generate_simulated_image()
        
        # 生成模拟机器人状态
        state = {
            "joint_angles": self._generate_joint_angles(),
            "joint_velocities": self._generate_joint_velocities(),
            "imu": self._generate_imu_data(),
            "body_pos": self._generate_body_pos(),
            "body_quat": self._generate_body_quat()
        }
        
        return {
            "image": image,
            "state": state
        }
    
    def _render_mujoco_image(self) -> np.ndarray:
        """渲染MuJoCo图像"""
        # 创建渲染器
        renderer = mujoco.Renderer(self.model, height=self.image_size[0], width=self.image_size[1])
        
        # 更新场景
        renderer.update_scene(self.data, camera="rgb_camera")
        
        # 渲染图像
        image = renderer.render()
        
        # 转换为RGB格式
        image = (image * 255).astype(np.uint8)
        
        return image
    
    def _generate_simulated_image(self) -> np.ndarray:
        """生成模拟RGB图像"""
        height, width = self.image_size
        
        # 创建基于机器人状态的模拟图像
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 基于时间和状态生成动态效果
        t = self._sim_time
        
        # 生成RGB通道的渐变
        for i in range(height):
            for j in range(width):
                # 基于位置和时间生成颜色
                r = int(128 + 127 * np.sin(i * 0.02 + t))
                g = int(128 + 127 * np.sin(j * 0.02 + t + np.pi/3))
                b = int(128 + 127 * np.sin((i + j) * 0.01 + t + 2*np.pi/3))
                
                image[i, j] = [r, g, b]
        
        self._sim_time += 0.1
        return image
    
    def _get_mujoco_imu_data(self) -> np.ndarray:
        """从MuJoCo获取IMU数据"""
        imu_data = np.zeros(9)
        
        # 从传感器数据获取加速度和角速度
        if hasattr(self.data, 'sensordata') and len(self.data.sensordata) >= 6:
            imu_data[0:3] = self.data.sensordata[0:3]  # 加速度
            imu_data[3:6] = self.data.sensordata[3:6]  # 角速度
        else:
            # 回退到模拟数据
            imu_data[0:3] = np.random.uniform(-2, 2, 3)
            imu_data[3:6] = np.random.uniform(-1, 1, 3)
        
        # 计算姿态角度（从四元数）
        if len(self.data.qpos) >= 7:
            quat = self.data.qpos[3:7]
            imu_data[6:9] = self._quaternion_to_euler(quat)
        else:
            imu_data[6:9] = np.random.uniform(-np.pi/4, np.pi/4, 3)
        
        return imu_data
    
    def _generate_joint_angles(self) -> np.ndarray:
        """生成模拟关节角度"""
        # 生成[-π, π]范围内的随机关节角度
        base_angles = np.random.uniform(-np.pi, np.pi, self.num_joints)
        
        # 添加连续性
        if self._last_joint_angles is not None:
            angles = 0.9 * self._last_joint_angles + 0.1 * base_angles
        else:
            angles = base_angles
        
        self._last_joint_angles = angles
        return angles
    
    def _generate_joint_velocities(self) -> np.ndarray:
        """生成模拟关节速度"""
        # 生成[-1, 1]范围内的随机速度
        base_velocities = np.random.uniform(-1, 1, self.num_joints)
        
        # 添加连续性
        if self._last_joint_velocities is not None:
            velocities = 0.9 * self._last_joint_velocities + 0.1 * base_velocities
        else:
            velocities = base_velocities
        
        self._last_joint_velocities = velocities
        return velocities
    
    def _generate_imu_data(self) -> np.ndarray:
        """生成模拟IMU数据"""
        # 模拟IMU: [ax, ay, az, wx, wy, wz, roll, pitch, yaw]
        imu_data = np.zeros(9)
        
        # 加速度 (m/s²)
        imu_data[0:3] = np.random.uniform(-2, 2, 3)
        
        # 角速度 (rad/s)
        imu_data[3:6] = np.random.uniform(-1, 1, 3)
        
        # 姿态角度 (rad)
        imu_data[6:9] = np.random.uniform(-np.pi/4, np.pi/4, 3)
        
        return imu_data
    
    def _generate_body_pos(self) -> np.ndarray:
        """生成模拟身体位置"""
        return np.array([0.0, 0.0, 0.8]) + np.random.uniform(-0.1, 0.1, 3)
    
    def _generate_body_quat(self) -> np.ndarray:
        """生成模拟身体姿态"""
        # 生成单位四元数
        quat = np.random.randn(4)
        quat = quat / np.linalg.norm(quat)
        return quat
    
    def _quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角"""
        # 确保四元数是单位四元数
        quat = quat / np.linalg.norm(quat)
        
        w, x, y, z = quat
        
        # 转换为欧拉角
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return np.array([roll, pitch, yaw])
    
    def apply_action(self, action: np.ndarray):
        """
        应用动作到MuJoCo环境
        
        Args:
            action: 动作向量
        """
        if MUJOCO_AVAILABLE and hasattr(self, 'model') and hasattr(self, 'data'):
            # 将动作应用到MuJoCo的控制器
            if len(action) >= self.model.nu:
                self.data.ctrl = action[:self.model.nu]
            else:
                # 如果动作维度不足，用零填充
                ctrl = np.zeros(self.model.nu)
                ctrl[:len(action)] = action
                self.data.ctrl = ctrl
            
            # 执行物理仿真步骤
            mujoco.mj_step(self.model, self.data)
    
    def get_state_dim(self) -> int:
        """获取状态空间维度"""
        return config.PERCEPTION["state_dim"]
    
    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        if MUJOCO_AVAILABLE and hasattr(self, 'model'):
            return self.model.nu
        else:
            return self.num_joints
    
    def reset(self):
        """重置感知模块状态"""
        self._sim_time = 0
        self._last_joint_angles = None
        self._last_joint_velocities = None
        
        # 重置MuJoCo环境
        if MUJOCO_AVAILABLE and hasattr(self, 'model'):
            mujoco.mj_resetData(self.model, self.data)
    
    def set_simulation_mode(self, mode: bool):
        """
        设置仿真模式
        
        Args:
            mode: True为仿真模式，False为真实机器人模式
        """
        self.simulation_mode = mode
        self.reset()
    
    def get_camera_image(self) -> np.ndarray:
        """获取相机图像（用于可视化）"""
        if self.simulation_mode and MUJOCO_AVAILABLE and hasattr(self, 'model'):
            return self._render_mujoco_image()
        else:
            return self._generate_simulated_image()
    
    def get_robot_state(self) -> Dict[str, Any]:
        """获取机器人状态（用于调试）"""
        obs = self.get_observation()
        return obs["state"]
    
    def validate_observation(self, obs: Dict[str, Any]) -> bool:
        """验证观测数据格式"""
        if "image" not in obs or "state" not in obs:
            return False
        
        # 检查图像格式
        image = obs["image"]
        if not isinstance(image, np.ndarray) or image.shape != self.image_size + (3,):
            return False
        
        # 检查状态格式
        state = obs["state"]
        required_keys = ["joint_angles", "joint_velocities", "imu", "body_pos", "body_quat"]
        for key in required_keys:
            if key not in state:
                return False
        
        return True
    
    def get_observation_statistics(self) -> Dict[str, Any]:
        """获取观测数据统计信息"""
        obs = self.get_observation()
        state = obs["state"]
        
        return {
            "image_shape": obs["image"].shape,
            "image_dtype": str(obs["image"].dtype),
            "joint_angles_shape": state["joint_angles"].shape,
            "joint_velocities_shape": state["joint_velocities"].shape,
            "imu_shape": state["imu"].shape,
            "body_pos_shape": state["body_pos"].shape,
            "body_quat_shape": state["body_quat"].shape,
            "state_dim": self.get_state_dim(),
            "action_dim": self.get_action_dim(),
            "simulation_mode": self.simulation_mode,
            "mujoco_available": MUJOCO_AVAILABLE
        }