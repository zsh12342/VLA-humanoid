"""
Skill Module - 动作技能模块
定义各种机器人动作技能的基类和具体实现
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


class Skill(ABC):
    """技能基类
    
    定义所有机器人动作技能的通用接口。
    每个技能都可以执行完整动作序列或生成单步动作。
    """
    
    def __init__(self, 
                 name: str, 
                 num_joints: int = 45,
                 max_duration: float = 5.0,
                 frequency: float = 50.0):
        """
        初始化技能基类
        
        Args:
            name: 技能名称
            num_joints: 机器人关节数量
            max_duration: 最大执行时长（秒）
            frequency: 执行频率（Hz）
        """
        self.name = name
        self.num_joints = num_joints
        self.max_duration = max_duration
        self.frequency = frequency
        self.dt = 1.0 / frequency
        
        # 执行状态
        self.is_executing = False
        self.current_step = 0
        self.max_steps = int(max_duration * frequency)
        self.start_time = None
        
        # 技能参数
        self.parameters = {}
        
        # 关节限制
        self.joint_limits = config.ROBOT["joint_limits"]
    
    @abstractmethod
    def execute(self, **kwargs) -> bool:
        """
        执行完整动作序列
        
        Args:
            **kwargs: 技能参数
            
        Returns:
            bool: 执行是否成功
        """
        pass
    
    @abstractmethod
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        根据观测返回单步动作
        
        Args:
            observation: 观测数据，包含图像和状态
            
        Returns:
            np.ndarray: 关节动作向量
        """
        pass
    
    def reset(self):
        """重置技能状态"""
        self.is_executing = False
        self.current_step = 0
        self.start_time = None
        self.parameters = {}
    
    def start_execution(self):
        """开始执行技能"""
        self.reset()
        self.is_executing = True
        self.start_time = time.time()
    
    def is_finished(self) -> bool:
        """
        检查技能是否执行完成
        
        Returns:
            bool: 是否完成
        """
        if not self.is_executing:
            return True
        
        # 检查步数限制
        if self.current_step >= self.max_steps:
            return True
        
        # 检查时间限制
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.max_duration:
                return True
        
        return False
    
    def get_progress(self) -> float:
        """
        获取执行进度
        
        Returns:
            float: 进度百分比 [0, 1]
        """
        if self.max_steps == 0:
            return 0.0
        return min(self.current_step / self.max_steps, 1.0)
    
    def get_elapsed_time(self) -> float:
        """
        获取已执行时间
        
        Returns:
            float: 已执行时间（秒）
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """
        设置技能参数
        
        Args:
            parameters: 技能参数字典
        """
        self.parameters.update(parameters)
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        获取技能参数
        
        Returns:
            Dict[str, Any]: 技能参数
        """
        return self.parameters.copy()
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """
        限制动作在关节限制范围内
        
        Args:
            action: 原始动作向量
            
        Returns:
            np.ndarray: 限制后的动作向量
        """
        # 获取关节限制
        joint_min = self.joint_limits["min"][:self.num_joints]
        joint_max = self.joint_limits["max"][:self.num_joints]
        
        # 限制动作范围
        action = np.clip(action, joint_min, joint_max)
        
        return action
    
    def get_skill_info(self) -> Dict[str, Any]:
        """
        获取技能信息
        
        Returns:
            Dict[str, Any]: 技能信息
        """
        return {
            "name": self.name,
            "num_joints": self.num_joints,
            "max_duration": self.max_duration,
            "frequency": self.frequency,
            "max_steps": self.max_steps,
            "is_executing": self.is_executing,
            "current_step": self.current_step,
            "progress": self.get_progress(),
            "elapsed_time": self.get_elapsed_time(),
            "parameters": self.parameters
        }


class WalkSkill(Skill):
    """行走技能"""
    
    def __init__(self, 
                 num_joints: int = 45,
                 max_duration: float = 10.0,
                 frequency: float = 50.0):
        """
        初始化行走技能
        
        Args:
            num_joints: 机器人关节数量
            max_duration: 最大执行时长
            frequency: 执行频率
        """
        super().__init__("WalkSkill", num_joints, max_duration, frequency)
        
        # 行走参数
        self.stride_length = 0.3
        self.step_height = 0.1
        self.walk_speed = 1.0
        self.direction = "forward"
        
        # 腿部关节索引（假设）
        self.left_leg_joints = [6, 7, 8, 15, 16, 17, 24, 25, 26]  # 左腿关节
        self.right_leg_joints = [9, 10, 11, 18, 19, 20, 27, 28, 29]  # 右腿关节
    
    def execute(self, 
                direction: str = "forward",
                speed: float = 1.0,
                stride_length: float = 0.3,
                **kwargs) -> bool:
        """
        执行行走动作
        
        Args:
            direction: 行走方向 ("forward", "backward", "left", "right")
            speed: 行走速度
            stride_length: 步长
            **kwargs: 其他参数
            
        Returns:
            bool: 执行是否成功
        """
        self.set_parameters({
            "direction": direction,
            "speed": speed,
            "stride_length": stride_length
        })
        
        self.direction = direction
        self.walk_speed = speed
        self.stride_length = stride_length
        
        self.start_execution()
        return True
    
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        生成行走动作
        
        Args:
            observation: 观测数据
            
        Returns:
            np.ndarray: 关节动作向量
        """
        if not self.is_executing:
            return np.zeros(self.num_joints)
        
        action = np.zeros(self.num_joints)
        
        # 获取当前状态
        joint_angles = observation["state"].get("joint_angles", np.zeros(self.num_joints))
        
        # 计算行走相位
        phase = (self.current_step / self.max_steps) * 4 * np.pi  # 两个完整步态周期
        
        # 根据方向调整相位
        direction_factor = self._get_direction_factor()
        
        # 生成腿部动作
        if len(self.left_leg_joints) > 0 and len(self.right_leg_joints) > 0:
            # 左腿动作
            left_action = self._generate_leg_action(
                phase, joint_angles[self.left_leg_joints], "left", direction_factor
            )
            action[self.left_leg_joints] = left_action
            
            # 右腿动作（相位相反）
            right_action = self._generate_leg_action(
                phase + np.pi, joint_angles[self.right_leg_joints], "right", direction_factor
            )
            action[self.right_leg_joints] = right_action
        
        # 生成身体平衡动作
        action = self._add_balance_action(action, observation)
        
        self.current_step += 1
        
        # 检查是否完成
        if self.is_finished():
            self.is_executing = False
        
        return self.clip_action(action)
    
    def _get_direction_factor(self) -> np.ndarray:
        """获取方向因子"""
        direction_map = {
            "forward": np.array([1.0, 0.0]),
            "backward": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]),
            "right": np.array([0.0, -1.0])
        }
        return direction_map.get(self.direction, np.array([1.0, 0.0]))
    
    def _generate_leg_action(self, 
                           phase: float, 
                           current_angles: np.ndarray,
                           leg_side: str,
                           direction_factor: np.ndarray) -> np.ndarray:
        """生成单腿动作"""
        action = np.zeros_like(current_angles)
        
        # 简化的步态模式
        hip_action = self.stride_length * self.walk_speed * np.sin(phase) * direction_factor[0]
        knee_action = self.step_height * self.walk_speed * max(0, np.sin(phase))
        ankle_action = self.stride_length * self.walk_speed * np.cos(phase) * direction_factor[0] * 0.5
        
        # 根据腿部关节数量分配动作
        if len(action) >= 3:
            action[0] = hip_action    # 髋关节
            action[1] = knee_action   # 膝关节
            action[2] = ankle_action  # 踝关节
        
        return action
    
    def _add_balance_action(self, action: np.ndarray, observation: Dict[str, Any]) -> np.ndarray:
        """添加身体平衡动作"""
        # 简单的身体平衡：根据身体姿态调整上半身
        body_quat = observation["state"].get("body_quat", np.array([0, 0, 0, 1]))
        
        # 计算身体倾斜
        roll, pitch, yaw = self._quaternion_to_euler(body_quat)
        
        # 调整上半身关节以保持平衡
        upper_body_joints = [0, 1, 2, 3, 4, 5]  # 假设这些是上半身关节
        for i, joint_idx in enumerate(upper_body_joints):
            if joint_idx < len(action):
                balance_factor = 0.1
                if i < 2:  # 前后平衡
                    action[joint_idx] += balance_factor * pitch
                elif i < 4:  # 左右平衡
                    action[joint_idx] += balance_factor * roll
        
        return action
    
    def _quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """四元数转欧拉角"""
        quat = quat / np.linalg.norm(quat)
        w, x, y, z = quat
        
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(2 * (w * y - z * x))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return np.array([roll, pitch, yaw])


class TurnSkill(Skill):
    """转向技能"""
    
    def __init__(self, 
                 num_joints: int = 45,
                 max_duration: float = 3.0,
                 frequency: float = 50.0):
        """
        初始化转向技能
        
        Args:
            num_joints: 机器人关节数量
            max_duration: 最大执行时长
            frequency: 执行频率
        """
        super().__init__("TurnSkill", num_joints, max_duration, frequency)
        
        # 转向参数
        self.turn_angle = np.pi / 4  # 45度
        self.turn_direction = "left"
        self.turn_speed = 1.0
    
    def execute(self, 
                direction: str = "left",
                angle: float = 45.0,
                speed: float = 1.0,
                **kwargs) -> bool:
        """
        执行转向动作
        
        Args:
            direction: 转向方向 ("left", "right")
            angle: 转向角度（度）
            speed: 转向速度
            **kwargs: 其他参数
            
        Returns:
            bool: 执行是否成功
        """
        self.set_parameters({
            "direction": direction,
            "angle": angle,
            "speed": speed
        })
        
        self.turn_direction = direction
        self.turn_angle = np.radians(angle)
        self.turn_speed = speed
        
        self.start_execution()
        return True
    
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        生成转向动作
        
        Args:
            observation: 观测数据
            
        Returns:
            np.ndarray: 关节动作向量
        """
        if not self.is_executing:
            return np.zeros(self.num_joints)
        
        action = np.zeros(self.num_joints)
        
        # 计算转向进度
        progress = self.current_step / self.max_steps
        turn_factor = np.sin(progress * np.pi)  # 平滑的转向曲线
        
        # 获取转向方向
        direction_multiplier = 1 if self.turn_direction == "left" else -1
        
        # 生成转向动作
        # 假设腿部关节用于转向
        leg_joints = [6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20]
        
        for i, joint_idx in enumerate(leg_joints):
            if joint_idx < len(action):
                # 交替的腿部动作实现转向
                if i % 2 == 0:
                    action[joint_idx] = direction_multiplier * self.turn_angle * turn_factor * self.turn_speed * 0.5
                else:
                    action[joint_idx] = -direction_multiplier * self.turn_angle * turn_factor * self.turn_speed * 0.3
        
        # 添加上半身转向
        upper_body_joints = [0, 1, 2, 3, 4, 5]
        for i, joint_idx in enumerate(upper_body_joints):
            if joint_idx < len(action):
                action[joint_idx] = direction_multiplier * self.turn_angle * turn_factor * self.turn_speed * 0.2
        
        self.current_step += 1
        
        # 检查是否完成
        if self.is_finished():
            self.is_executing = False
        
        return self.clip_action(action)


class CustomSkill(Skill):
    """自定义技能示例"""
    
    def __init__(self, 
                 num_joints: int = 45,
                 max_duration: float = 5.0,
                 frequency: float = 50.0,
                 skill_type: str = "wave"):
        """
        初始化自定义技能
        
        Args:
            num_joints: 机器人关节数量
            max_duration: 最大执行时长
            frequency: 执行频率
            skill_type: 技能类型
        """
        super().__init__("CustomSkill", num_joints, max_duration, frequency)
        self.skill_type = skill_type
        
        # 技能参数
        self.amplitude = 0.5
        self.frequency = 0.1
        
        # 根据技能类型设置关节
        self._setup_skill_joints()
    
    def _setup_skill_joints(self):
        """设置技能相关的关节"""
        if self.skill_type == "wave":
            # 挥手：使用手臂关节
            self.active_joints = [12, 13, 14, 21, 22, 23]  # 手臂关节
        elif self.skill_type == "bow":
            # 鞠躬：使用上半身关节
            self.active_joints = [0, 1, 2, 3, 4, 5]  # 上半身关节
        elif self.skill_type == "stand":
            # 站立：使用所有关节
            self.active_joints = list(range(self.num_joints))
        else:
            # 默认：使用前几个关节
            self.active_joints = list(range(min(10, self.num_joints)))
    
    def execute(self, 
                amplitude: float = 0.5,
                frequency: float = 0.1,
                **kwargs) -> bool:
        """
        执行自定义技能
        
        Args:
            amplitude: 动作幅度
            frequency: 动作频率
            **kwargs: 其他参数
            
        Returns:
            bool: 执行是否成功
        """
        self.set_parameters({
            "amplitude": amplitude,
            "frequency": frequency,
            "skill_type": self.skill_type
        })
        
        self.amplitude = amplitude
        self.frequency = frequency
        
        self.start_execution()
        return True
    
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        生成自定义动作
        
        Args:
            observation: 观测数据
            
        Returns:
            np.ndarray: 关节动作向量
        """
        if not self.is_executing:
            return np.zeros(self.num_joints)
        
        action = np.zeros(self.num_joints)
        
        # 根据技能类型生成动作
        if self.skill_type == "wave":
            action = self._generate_wave_action(action)
        elif self.skill_type == "bow":
            action = self._generate_bow_action(action)
        elif self.skill_type == "stand":
            action = self._generate_stand_action(action)
        else:
            action = self._generate_default_action(action)
        
        self.current_step += 1
        
        # 检查是否完成
        if self.is_finished():
            self.is_executing = False
        
        return self.clip_action(action)
    
    def _generate_wave_action(self, action: np.ndarray) -> np.ndarray:
        """生成挥手动作"""
        t = self.current_step * self.frequency
        
        for i, joint_idx in enumerate(self.active_joints):
            if joint_idx < len(action):
                # 交替的挥手动作
                if i % 2 == 0:
                    action[joint_idx] = self.amplitude * np.sin(2 * np.pi * t)
                else:
                    action[joint_idx] = self.amplitude * np.cos(2 * np.pi * t) * 0.5
        
        return action
    
    def _generate_bow_action(self, action: np.ndarray) -> np.ndarray:
        """生成鞠躬动作"""
        progress = self.get_progress()
        
        # 鞠躬的平滑曲线
        bow_angle = self.amplitude * np.sin(progress * np.pi)
        
        for joint_idx in self.active_joints:
            if joint_idx < len(action):
                # 主要用腰部关节鞠躬
                if joint_idx in [1, 2]:  # 假设这些是腰部关节
                    action[joint_idx] = bow_angle
                else:
                    action[joint_idx] = bow_angle * 0.1  # 其他关节微调
        
        return action
    
    def _generate_stand_action(self, action: np.ndarray) -> np.ndarray:
        """生成站立动作"""
        # 站立动作主要是保持平衡，微调姿势
        progress = self.get_progress()
        
        # 轻微的平衡调整
        balance_adjustment = 0.05 * np.sin(2 * np.pi * progress)
        
        for joint_idx in self.active_joints:
            if joint_idx < len(action):
                action[joint_idx] = balance_adjustment
        
        return action
    
    def _generate_default_action(self, action: np.ndarray) -> np.ndarray:
        """生成默认动作"""
        t = self.current_step * self.frequency
        
        # 简单的正弦波动作
        for i, joint_idx in enumerate(self.active_joints):
            if joint_idx < len(action):
                phase_shift = i * 2 * np.pi / len(self.active_joints)
                action[joint_idx] = self.amplitude * np.sin(2 * np.pi * t + phase_shift)
        
        return action


class SkillManager:
    """技能管理器"""
    
    def __init__(self, num_joints: int = 45):
        """
        初始化技能管理器
        
        Args:
            num_joints: 机器人关节数量
        """
        self.num_joints = num_joints
        self.skills = {}
        self.active_skill = None
        
        # 注册默认技能
        self._register_default_skills()
    
    def _register_default_skills(self):
        """注册默认技能"""
        self.skills["walk"] = WalkSkill(self.num_joints)
        self.skills["turn"] = TurnSkill(self.num_joints)
        self.skills["wave"] = CustomSkill(self.num_joints, skill_type="wave")
        self.skills["bow"] = CustomSkill(self.num_joints, skill_type="bow")
        self.skills["stand"] = CustomSkill(self.num_joints, skill_type="stand")
    
    def register_skill(self, name: str, skill: Skill):
        """
        注册新技能
        
        Args:
            name: 技能名称
            skill: 技能实例
        """
        self.skills[name] = skill
    
    def execute_skill(self, skill_name: str, **kwargs) -> bool:
        """
        执行指定技能
        
        Args:
            skill_name: 技能名称
            **kwargs: 技能参数
            
        Returns:
            bool: 执行是否成功
        """
        if skill_name not in self.skills:
            print(f"Skill '{skill_name}' not found")
            return False
        
        # 停止当前技能
        if self.active_skill is not None:
            self.active_skill.reset()
        
        # 执行新技能
        self.active_skill = self.skills[skill_name]
        return self.active_skill.execute(**kwargs)
    
    def step(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        执行当前技能的单步动作
        
        Args:
            observation: 观测数据
            
        Returns:
            np.ndarray: 动作向量
        """
        if self.active_skill is None:
            return np.zeros(self.num_joints)
        
        action = self.active_skill.step(observation)
        
        # 如果技能完成，清空活跃技能
        if self.active_skill.is_finished():
            self.active_skill = None
        
        return action
    
    def get_active_skill(self) -> Optional[Skill]:
        """获取当前活跃技能"""
        return self.active_skill
    
    def get_skill_names(self) -> List[str]:
        """获取所有技能名称"""
        return list(self.skills.keys())
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """获取技能信息"""
        if skill_name not in self.skills:
            return None
        return self.skills[skill_name].get_skill_info()
    
    def stop_all_skills(self):
        """停止所有技能"""
        if self.active_skill is not None:
            self.active_skill.reset()
            self.active_skill = None
        
        for skill in self.skills.values():
            skill.reset()