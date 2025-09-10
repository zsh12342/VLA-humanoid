#!/usr/bin/env python3
"""
流畅的轨迹生成器推理节点
核心改进：
1. 放宽重新生成阈值
2. 增加轨迹混合，避免卡顿
3. 改进轨迹执行逻辑
"""

import rospy
import json
import time
import math
import numpy as np
import os
import sys
import argparse
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Quaternion, Point
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_srvs.srv import Trigger, TriggerResponse
import pickle
from tqdm import tqdm

# 提前导入手臂控制相关的消息类型
try:
    from kuavo_msgs.msg import armTargetPoses
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    HAS_ARM_MSGS = True
except ImportError:
    rospy.logwarn("无法导入 kuavo_msgs，手臂控制功能将被禁用")
    HAS_ARM_MSGS = False

# 导入模型定义
from true_trajectory_generator import FixedTrajectoryGenerator as TrajectoryGenerator

class SmoothTrajectoryInferenceNode:
    """流畅的轨迹生成器推理节点"""
    
    def __init__(self, model_path: str, config: dict):
        """初始化推理节点"""
        self.model_path = model_path
        self.config = config
        
        # 推理状态
        self.is_running = True
        self.inference_frequency = config.get('inference_frequency', 50.0)
        self.trajectory_length = config.get('trajectory_length', 100)
        self.current_trajectory_step = 0
        self.generated_trajectory = None
        self.trajectory_start_time = 0
        self.trajectory_regeneration_threshold = 5.0  # 大幅放宽重新生成阈值
        
        # 控制参数
        self.instruction = config.get('instruction', '挥手')
        self.control_mode = config.get('control_mode', 'arm')
        self.publish_commands = config.get('publish_commands', True)
        
        # 模型和标准化参数
        self.model = None
        self.norm_stats = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 指令映射
        self.instruction_to_id = {'挥手': 0, '抱拳': 1}
        rospy.loginfo(f"指令映射: {self.instruction_to_id}")
        rospy.loginfo(f"当前指令: {self.instruction} -> ID: {self.instruction_to_id.get(self.instruction, 'unknown')}")
        
        # 机器人状态
        self.current_joint_positions = None
        self.trajectory_start_state = None
        
        # 轨迹混合参数
        self.mixing_enabled = True
        self.mixing_steps = 5  # 混合步数
        
        # 初始化模型
        self._load_model()
        
        # 初始化ROS接口
        self._setup_ros_interfaces()
        
        # 如果是手臂控制模式，设置手臂控制模式
        if self.control_mode == 'arm':
            self._setup_arm_control()
        
        rospy.loginfo("流畅的轨迹生成器推理节点初始化完成")
        rospy.loginfo(f"模型路径: {self.model_path}")
        rospy.loginfo(f"指令: {self.instruction}")
        rospy.loginfo(f"控制模式: {self.control_mode}")
        rospy.loginfo(f"推理频率: {self.inference_frequency}Hz")
        rospy.loginfo(f"轨迹长度: {self.trajectory_length}")
        rospy.loginfo(f"重新生成阈值: {self.trajectory_regeneration_threshold}")
        rospy.loginfo(f"设备: {self.device}")
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            rospy.loginfo(f"加载模型: {self.model_path}")
            
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # 获取配置和标准化参数
            model_config = checkpoint['config']
            self.norm_stats = checkpoint['norm_stats']
            
            rospy.loginfo(f"模型配置: {model_config}")
            rospy.loginfo(f"标准化参数键: {list(self.norm_stats.keys())}")
            
            # 创建模型
            self.model = TrajectoryGenerator(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            rospy.loginfo("模型加载成功")
            rospy.loginfo(f"模型设备: {self.device}")
            
            # 测试模型推理
            self._test_model_inference()
            
        except Exception as e:
            rospy.logerr(f"加载模型失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            raise
    
    def _setup_ros_interfaces(self):
        """设置ROS接口"""
        try:
            # 订阅机器人状态话题
            self.joint_state_sub = rospy.Subscriber(
                '/humanoid_controller/optimizedState_mrt/joint_pos',
                Float64MultiArray,
                self._joint_state_callback
            )
            
            # 动作命令发布器
            if self.control_mode == 'arm':
                if HAS_ARM_MSGS:
                    self.arm_target_pub = rospy.Publisher(
                        '/kuavo_arm_target_poses',
                        armTargetPoses,
                        queue_size=10
                    )
                    rospy.loginfo("手臂控制模式 - 发布手臂目标姿态")
                else:
                    rospy.logerr("无法创建手臂发布器：缺少 kuavo_msgs")
            
            # 推理控制服务
            self.start_service = rospy.Service(
                '/smooth_trajectory_inference/start',
                Trigger,
                self._start_callback
            )
            
            self.stop_service = rospy.Service(
                '/smooth_trajectory_inference/stop',
                Trigger,
                self._stop_callback
            )
            
            rospy.loginfo("ROS接口设置完成")
            rospy.loginfo("控制服务:")
            rospy.loginfo("  开始推理: rosservice call /smooth_trajectory_inference/start")
            rospy.loginfo("  停止推理: rosservice call /smooth_trajectory_inference/stop")
            
        except Exception as e:
            rospy.logerr(f"设置ROS接口失败: {e}")
            raise
    
    def _setup_arm_control(self):
        """设置手臂控制模式"""
        if not HAS_ARM_MSGS:
            rospy.logerr("无法设置手臂控制模式：缺少 kuavo_msgs")
            return
            
        try:
            # 等待手臂控制模式服务
            rospy.wait_for_service('/arm_traj_change_mode', timeout=5.0)
            
            # 创建服务客户端
            change_mode = rospy.ServiceProxy('/arm_traj_change_mode', changeArmCtrlMode)
            
            # 创建请求
            req = changeArmCtrlModeRequest()
            req.control_mode = 2  # EXTERN_CONTROL (外部控制模式)
            
            # 调用服务
            res = change_mode(req)
            
            if res.result:
                rospy.loginfo("手臂控制模式已设置为: EXTERN_CONTROL (外部控制)")
            else:
                rospy.logerr(f"设置手臂控制模式失败: {res.message}")
                
        except Exception as e:
            rospy.logerr(f"设置手臂控制模式时出错: {e}")
            rospy.logwarn("手臂可能不会响应外部控制命令")
    
    def _test_model_inference(self):
        """测试模型推理"""
        try:
            rospy.loginfo("测试模型推理...")
            
            # 创建测试数据
            state_dim = 26
            test_start_state = np.random.randn(state_dim).astype(np.float32)
            test_instruction_id = 0  # 挥手
            
            # 标准化
            test_start_state_norm = (test_start_state - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            
            # 转换为tensor
            start_state_tensor = torch.FloatTensor(test_start_state_norm).unsqueeze(0).to(self.device)
            instruction_id_tensor = torch.LongTensor([test_instruction_id]).to(self.device)
            
            # 进行推理测试
            with torch.no_grad():
                predicted_trajectory_norm = self.model(start_state_tensor, instruction_id_tensor)
            
            # 反标准化
            predicted_trajectory = predicted_trajectory_norm.cpu().numpy()[0] * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            rospy.loginfo(f"测试轨迹形状: {predicted_trajectory.shape}")
            rospy.loginfo(f"测试轨迹范围: [{predicted_trajectory.min():.3f}, {predicted_trajectory.max():.3f}]")
            rospy.loginfo(f"测试轨迹前3步: {predicted_trajectory[:3, :3]}")
            rospy.loginfo(f"测试轨迹变化: {np.std(predicted_trajectory, axis=0)[:3]}")
            
            rospy.loginfo("模型推理测试完成!")
            
        except Exception as e:
            rospy.logerr(f"模型推理测试失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def _joint_state_callback(self, msg: Float64MultiArray):
        """关节状态回调"""
        self.current_joint_positions = np.array(msg.data[:26])
        
        # 如果还没有生成轨迹，生成一个
        if self.generated_trajectory is None and self.is_running:
            rospy.loginfo("初始化轨迹生成...")
            self._generate_new_trajectory()
    
    def _should_regenerate_trajectory(self):
        """判断是否需要重新生成轨迹"""
        if self.trajectory_start_state is None:
            return True
        
        # 计算当前状态与轨迹起始状态的差异
        state_diff = np.linalg.norm(self.current_joint_positions - self.trajectory_start_state)
        
        # 如果状态差异超过阈值，需要重新生成
        if state_diff > self.trajectory_regeneration_threshold:
            rospy.loginfo(f"状态差异过大 ({state_diff:.3f} > {self.trajectory_regeneration_threshold})，重新生成轨迹")
            return True
        
        # 如果轨迹执行完成，需要重新生成
        if self.current_trajectory_step >= len(self.generated_trajectory):
            rospy.loginfo("轨迹执行完成，重新生成")
            return True
        
        return False
    
    def _mix_trajectories(self, old_trajectory, new_trajectory, mix_step):
        """混合新旧轨迹，避免卡顿"""
        if mix_step >= self.mixing_steps:
            return new_trajectory
        
        # 计算混合权重
        alpha = mix_step / self.mixing_steps  # 0到1之间
        
        # 混合轨迹
        mixed_trajectory = (1 - alpha) * old_trajectory + alpha * new_trajectory
        
        return mixed_trajectory
    
    def _generate_new_trajectory(self):
        """生成新的轨迹"""
        try:
            if self.current_joint_positions is None:
                rospy.logwarn("当前关节位置未知，无法生成轨迹")
                return
            
            rospy.loginfo(f"从当前状态生成轨迹 - 指令: {self.instruction}")
            
            # 记录轨迹起始状态
            self.trajectory_start_state = self.current_joint_positions.copy()
            
            # 获取指令ID
            instruction_id = self.instruction_to_id.get(self.instruction, 0)
            
            # 标准化当前状态
            current_state_norm = (self.current_joint_positions - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            
            # 转换为tensor
            start_state_tensor = torch.FloatTensor(current_state_norm).unsqueeze(0).to(self.device)
            instruction_id_tensor = torch.LongTensor([instruction_id]).to(self.device)
            
            # 生成轨迹
            with torch.no_grad():
                predicted_trajectory_norm = self.model(start_state_tensor, instruction_id_tensor)
            
            # 反标准化
            predicted_trajectory = predicted_trajectory_norm.cpu().numpy()[0] * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            # 如果是第一次生成，直接使用
            if self.generated_trajectory is None:
                self.generated_trajectory = predicted_trajectory
                self.mix_step = 0
            else:
                # 否则混合新旧轨迹
                if self.mixing_enabled:
                    self.generated_trajectory = self._mix_trajectories(
                        self.generated_trajectory, predicted_trajectory, self.mix_step
                    )
                    self.mix_step += 1
                else:
                    self.generated_trajectory = predicted_trajectory
            
            self.current_trajectory_step = 0
            self.trajectory_start_time = time.time()
            
            rospy.loginfo(f"生成新轨迹 - 形状: {predicted_trajectory.shape}")
            rospy.loginfo(f"轨迹范围: [{predicted_trajectory.min():.3f}, {predicted_trajectory.max():.3f}]")
            rospy.loginfo(f"起始位置: {predicted_trajectory[0, :3]}")
            rospy.loginfo(f"结束位置: {predicted_trajectory[-1, :3]}")
            rospy.loginfo(f"轨迹变化: {np.std(predicted_trajectory, axis=0)[:3]}")
            
            # 检查轨迹是否有足够的变化
            trajectory_variance = np.var(predicted_trajectory, axis=0)
            if np.mean(trajectory_variance) < 0.001:
                rospy.logwarn("警告：生成的轨迹变化很小，可能存在问题")
            
        except Exception as e:
            rospy.logerr(f"生成轨迹失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def _publish_arm_command(self, action: np.ndarray):
        """发布手臂控制命令"""
        if not HAS_ARM_MSGS:
            return
            
        try:
            # 提取手臂关节数据（左臂12-18，右臂19-25）
            left_arm_joints = action[12:19]  # 左臂7个关节
            right_arm_joints = action[19:26]  # 右臂7个关节
            
            # 转换为度数
            left_arm_degrees = [math.degrees(joint) for joint in left_arm_joints]
            right_arm_degrees = [math.degrees(joint) for joint in right_arm_joints]
            
            # 组合关节数据
            arm_joints = left_arm_degrees + right_arm_degrees
            
            # 创建消息
            arm_msg = armTargetPoses()
            arm_msg.times = [0.0]  # 立即执行
            arm_msg.values = arm_joints
            arm_msg.frame = 2  # local frame
            
            # 发布
            self.arm_target_pub.publish(arm_msg)
            
            rospy.logdebug(f"手臂命令发布: 左臂前3个={left_arm_degrees[:3]}, 右臂前3个={right_arm_degrees[:3]}")
            
        except Exception as e:
            rospy.logerr(f"发布手臂命令失败: {e}")
    
    def _start_callback(self, req):
        """开始推理回调"""
        if self.is_running:
            return TriggerResponse(False, "推理已在进行中")
        
        self.is_running = True
        self.generated_trajectory = None  # 清空旧轨迹
        self.current_trajectory_step = 0
        self.mix_step = 0
        rospy.loginfo("开始流畅的轨迹生成推理")
        return TriggerResponse(True, "推理已开始")
    
    def _stop_callback(self, req):
        """停止推理回调"""
        if not self.is_running:
            return TriggerResponse(False, "推理未在进行中")
        
        self.is_running = False
        self.generated_trajectory = None
        self.current_trajectory_step = 0
        self.mix_step = 0
        rospy.loginfo("停止流畅的轨迹生成推理")
        return TriggerResponse(True, "推理已停止")
    
    def start_inference(self):
        """开始推理"""
        rospy.loginfo("流畅的轨迹生成器推理节点已启动...")
        rospy.loginfo("推理已自动开始！")
        rospy.loginfo("使用以下命令控制推理:")
        rospy.loginfo("  停止推理: rosservice call /smooth_trajectory_inference/stop")
        rospy.loginfo("  重新开始: rosservice call /smooth_trajectory_inference/start")
        rospy.loginfo("按Ctrl+C退出")
        
        # 保持节点运行
        rate = rospy.Rate(self.inference_frequency)
        
        while not rospy.is_shutdown():
            if self.is_running:
                # 检查是否需要重新生成轨迹
                if (self.generated_trajectory is None or 
                    self._should_regenerate_trajectory()):
                    self._generate_new_trajectory()
                
                # 执行当前轨迹
                if self.generated_trajectory is not None:
                    # 获取当前动作
                    if self.current_trajectory_step < len(self.generated_trajectory):
                        current_action = self.generated_trajectory[self.current_trajectory_step]
                        
                        # 发布控制命令
                        if self.publish_commands:
                            self._publish_arm_command(current_action)
                        
                        rospy.logdebug(f"步数: {self.current_trajectory_step}/{len(self.generated_trajectory)}, "
                                     f"动作范围: [{current_action.min():.3f}, {current_action.max():.3f}]")
                        
                        # 更新步数
                        self.current_trajectory_step += 1
                        
                        # 检查轨迹是否完成
                        if self.current_trajectory_step >= len(self.generated_trajectory):
                            rospy.loginfo("轨迹执行完成，停止执行")
                            self.generated_trajectory = None  # 停止，等待新的指令或重新开始
              
            rate.sleep()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='流畅的轨迹生成器推理节点')
    parser.add_argument('--model_path', required=True, help='模型文件路径')
    parser.add_argument('--instruction', default='挥手', help='指令类型 (挥手/抱拳)')
    parser.add_argument('--control_mode', choices=['arm', 'base', 'none'], default='arm',
                       help='控制模式 (arm/base/none)')
    parser.add_argument('--frequency', type=float, default=10.0, help='推理频率Hz')
    parser.add_argument('--trajectory_length', type=int, default=100, help='轨迹长度')
    parser.add_argument('--no_publish', action='store_true', help='不发布控制命令（仅测试）')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 构建配置
    config = {
        'instruction': args.instruction,
        'control_mode': args.control_mode,
        'inference_frequency': args.frequency,
        'trajectory_length': args.trajectory_length,
        'publish_commands': not args.no_publish
    }
    
    # 初始化ROS节点
    rospy.init_node('smooth_trajectory_inference_node', anonymous=True)
    
    try:
        # 创建推理节点
        inference_node = SmoothTrajectoryInferenceNode(args.model_path, config)
        
        # 开始推理
        inference_node.start_inference()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"推理节点运行出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()