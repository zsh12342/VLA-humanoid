#!/usr/bin/env python3
"""
流畅的ACT推理节点 - 恢复到原来的工作版本
核心逻辑：
1. 生成完整轨迹序列
2. 连续执行轨迹
3. 执行完毕后重新生成
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

class KeyJointACTGenerator(nn.Module):
    """关键关节专注的ACT生成器 - 与训练脚本完全一致"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 基础参数
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.num_instructions = config['num_instructions']
        self.hidden_dim = config['hidden_dim']
        self.trajectory_length = config['trajectory_length']
        self.dropout = config['dropout']
        
        # 关键关节数量 - 每个指令重点关注的前N个关节
        self.key_joints_per_instruction = config.get('key_joints_per_instruction', 8)
        
        # 差分预测标志
        self.predict_differences = config.get('predict_differences', False)
        
        # 第一层：指令分类器
        self.instruction_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, self.num_instructions)
        )
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 指令嵌入
        self.instruction_embedding = nn.Embedding(self.num_instructions, 64)
        
        # 时间编码
        self.time_encoding = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4)
        )
        
        # 时序编码器 - 使用Transformer更好地处理时序依赖
        temporal_input_size = self.hidden_dim + 64 + self.hidden_dim // 4  # state + instruction + time
        
        # 关节重要性分析器 - 为每个指令分析关节重要性
        self.joint_importance_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.action_dim),
            nn.Sigmoid()  # 输出每个关节的重要性权重
        )
        
        # 第二层：指令专用的关键关节预测器
        self.key_joint_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(temporal_input_size, 256),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(128, self.key_joints_per_instruction)  # 只预测关键关节
            ) for _ in range(self.num_instructions)
        ])
        
        # 完整关节输出层 - 从关键关节扩展到所有关节
        self.full_joint_expander = nn.Sequential(
            nn.Linear(self.key_joints_per_instruction, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.action_dim)
        )
        
        # 时序编码器
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=temporal_input_size,
                nhead=8,
                dim_feedforward=self.hidden_dim,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 损失权重
        self.classification_weight = 10.0
        self.diversity_weight = 5.0
        
        # 信号放大参数
        self.signal_amplification = config.get('signal_amplification', 1.0)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """权重初始化 - 使用更保守的初始化策略"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 使用更小的初始化范围，防止早期训练不稳定
                if module.weight.dim() >= 2:  # 确保至少是2维张量
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
            elif isinstance(module, nn.TransformerEncoderLayer):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name and param.dim() >= 2:
                        nn.init.xavier_uniform_(param, gain=0.5)
                    elif 'bias' in param_name:
                        nn.init.constant_(param, 0.0)
    
    def forward(self, start_states, instruction_ids, target_actions=None):
        """前向传播"""
        batch_size = start_states.size(0)
        device = start_states.device
        
        # 状态编码
        state_encoded = self.state_encoder(start_states)
        
        # 第一层：指令分类
        instruction_logits = self.instruction_classifier(state_encoded)
        
        # 指令嵌入
        instruction_emb = self.instruction_embedding(instruction_ids)
        
        # 分析关节重要性
        joint_importance_input = torch.cat([state_encoded, instruction_emb], dim=-1)
        joint_importance = self.joint_importance_analyzer(joint_importance_input)
        
        # 时间编码
        time_steps = torch.linspace(0, 1, self.trajectory_length, device=device)
        time_embed = self.time_encoding(time_steps.unsqueeze(-1)).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 准备时序输入
        state_expanded = state_encoded.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        instruction_expanded = instruction_emb.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        
        temporal_input = torch.cat([state_expanded, instruction_expanded, time_embed], dim=-1)
        
        # 时序编码
        temporal_output = self.temporal_encoder(temporal_input)
        
        # 第二层：指令专用的关键关节预测
        key_joint_actions = []
        for i in range(batch_size):
            instruction_id = instruction_ids[i].item()
            predictor = self.key_joint_predictors[instruction_id]
            
            # 预测关键关节 - 处理整个序列
            sequence_output = temporal_output[i]  # [sequence_length, hidden_dim]
            key_action = predictor(sequence_output)  # [sequence_length, key_joints_per_instruction]
            key_joint_actions.append(key_action)
        
        key_joint_actions = torch.stack(key_joint_actions, dim=0)  # [batch_size, sequence_length, key_joints_per_instruction]
        
        # 扩展到完整关节输出
        full_joint_actions = []
        for t in range(self.trajectory_length):
            key_at_t = key_joint_actions[:, t, :]
            full_at_t = self.full_joint_expander(key_at_t)
            full_joint_actions.append(full_at_t)
        
        predicted_actions = torch.stack(full_joint_actions, dim=1)
        
        return predicted_actions, instruction_logits, joint_importance, key_joint_actions

class SmoothACTInferenceNode:
    """流畅的ACT推理节点 - 恢复到原来的工作版本"""
    
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
        
        # 调试计数器
        self.debug_counter = 0
        
        # 动作上下文管理 - 完全信任模型的时序能力
        self.total_steps_executed = 0  # 总执行步数
        self.action_start_time = None  # 动作开始时间
        self.trajectory_history = []  # 轨迹历史用于模型上下文
        self.max_history_length = 32  # 使用模型的时序长度作为上下文
        self.action_context = 'continuous'  # 持续动作，不人为干预
        
        # 关键修复：使用单一长轨迹而非多段拼接
        self.single_trajectory_mode = True  # 单一轨迹模式
        self.trajectory_extension_steps = 16  # 每次扩展16步而非32步
        self.is_action_completed = False  # 动作是否完成
        self.action_cycle_count = 0  # 动作周期计数
        
        # 连续执行参数
        self.execution_steps = 5  # 每次只执行前5步
        self.lookahead_steps = 10  # 提前10步生成下一段轨迹
        self.trajectory_buffer = []  # 轨迹缓冲区
        self.next_trajectory = None  # 下一段轨迹
        self.is_generating = False  # 是否正在生成轨迹
        
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
        
        rospy.loginfo("流畅的ACT推理节点初始化完成")
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
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                model_config = checkpoint['config']
            
            if 'norm_stats' in checkpoint:
                self.norm_stats = checkpoint['norm_stats']
            else:
                # 使用默认的标准化参数
                self.norm_stats = {
                    'state_mean': np.zeros(26),
                    'state_std': np.ones(26),
                    'action_mean': np.zeros(26),
                    'action_std': np.ones(26)
                }
            
            rospy.loginfo(f"模型配置: {model_config}")
            rospy.loginfo(f"标准化参数键: {list(self.norm_stats.keys())}")
            
            # 创建模型
            self.model = KeyJointACTGenerator(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 更新轨迹长度为模型的实际长度
            self.trajectory_length = self.model.trajectory_length
            rospy.loginfo(f"模型轨迹长度: {self.trajectory_length}")
            
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
                '/smooth_act_inference/start',
                Trigger,
                self._start_callback
            )
            
            self.stop_service = rospy.Service(
                '/smooth_act_inference/stop',
                Trigger,
                self._stop_callback
            )
            
            rospy.loginfo("ROS接口设置完成")
            rospy.loginfo("控制服务:")
            rospy.loginfo("  开始推理: rosservice call /smooth_act_inference/start")
            rospy.loginfo("  停止推理: rosservice call /smooth_act_inference/stop")
            
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
                outputs = self.model(start_state_tensor, instruction_id_tensor)
                predicted_actions_norm = outputs[0]  # 第一个输出是预测动作
                instruction_logits = outputs[1]  # 第二个输出是指令分类
                joint_importance = outputs[2]  # 第三个输出是关节重要性
            
            # 反标准化
            predicted_actions = predicted_actions_norm.cpu().numpy()[0] * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            rospy.loginfo(f"测试轨迹形状: {predicted_actions.shape}")
            rospy.loginfo(f"测试轨迹范围: [{predicted_actions.min():.3f}, {predicted_actions.max():.3f}]")
            rospy.loginfo(f"测试轨迹前3步: {predicted_actions[:3, :3]}")
            rospy.loginfo(f"测试轨迹变化: {np.std(predicted_actions, axis=0)[:3]}")
            rospy.loginfo(f"指令分类结果: {torch.argmax(instruction_logits, dim=1).cpu().numpy()}")
            rospy.loginfo(f"关节重要性前5个: {torch.topk(joint_importance[0], 5)[1].cpu().numpy()}")
            
            rospy.loginfo("模型推理测试完成!")
            
        except Exception as e:
            rospy.logerr(f"模型推理测试失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def _joint_state_callback(self, msg: Float64MultiArray):
        """关节状态回调"""
        self.current_joint_positions = np.array(msg.data[:26])
        rospy.logdebug(f"接收到关节位置数据，前3个关节: {self.current_joint_positions[:3]}")
        
        # 如果还没有轨迹缓冲区，记录初始位置
        if len(self.trajectory_buffer) == 0 and self.is_running:
            self.initial_position = self.current_joint_positions.copy()
    
    def _generate_trajectory_sequence(self):
        """生成轨迹序列 - 用于测试的简化版本"""
        if self.current_joint_positions is None:
            rospy.logwarn("当前关节位置未知，无法生成轨迹")
            return None
        
        try:
            rospy.loginfo("生成轨迹序列...")
            
            # 获取指令ID
            instruction_id = self.instruction_to_id.get(self.instruction, 0)
            
            # 标准化当前状态
            current_state_norm = (self.current_joint_positions - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            
            # 转换为tensor
            start_state_tensor = torch.FloatTensor(current_state_norm).unsqueeze(0).to(self.device)
            instruction_id_tensor = torch.LongTensor([instruction_id]).to(self.device)
            
            # 生成轨迹
            with torch.no_grad():
                outputs = self.model(start_state_tensor, instruction_id_tensor)
                predicted_actions_norm = outputs[0]  # 第一个输出是预测动作
            
            # 反标准化
            predicted_actions = predicted_actions_norm.cpu().numpy()[0] * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            rospy.loginfo(f"轨迹生成完成，形状: {predicted_actions.shape}")
            return predicted_actions
            
        except Exception as e:
            rospy.logerr(f"生成轨迹序列失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            return None
    
    def _update_action_context(self):
        """更新动作上下文 - 完全信任模型，只提供历史信息"""
        if self.action_start_time is None:
            self.action_start_time = time.time()
        
        # 更新轨迹历史，提供给模型作为上下文
        if len(self.trajectory_buffer) > 0:
            # 记录最近执行的轨迹
            recently_executed = self.trajectory_buffer[max(0, self.current_trajectory_step - 32):self.current_trajectory_step]
            if recently_executed:
                self.trajectory_history.extend(recently_executed)
                
                # 保持历史记录在模型时序长度内
                if len(self.trajectory_history) > self.max_history_length:
                    self.trajectory_history = self.trajectory_history[-self.max_history_length:]
                
                # 跟踪最大距离
                if hasattr(self, 'initial_position'):
                    current_position = np.array(self.trajectory_history[-1])
                    distance_to_initial = np.linalg.norm(current_position - self.initial_position)
                    if not hasattr(self, 'max_distance_reached') or distance_to_initial > self.max_distance_reached:
                        self.max_distance_reached = distance_to_initial
        
        # 让模型完全自主决定动作的时序和阶段
        # 不设置任何时间限制或阶段判断
        
        # 定期分析轨迹模式（仅用于调试，不影响决策）
        if self.total_steps_executed % 100 == 0 and len(self.trajectory_history) >= 16:
            self._analyze_trajectory_pattern()
    
    def _analyze_trajectory_pattern(self):
        """分析轨迹模式 - 仅用于调试，不影响决策"""
        if len(self.trajectory_history) < 16:
            return
        
        # 分析最近的轨迹模式
        recent_trajectory = np.array(self.trajectory_history[-16:])
        
        # 计算关键指标
        trajectory_variance = np.var(recent_trajectory, axis=0)
        mean_variance = np.mean(trajectory_variance)
        
        # 分析关键关节
        critical_joints = [12, 13, 14]
        critical_variance = np.mean([trajectory_variance[joint] for joint in critical_joints])
        
        # 检测是否在运动中
        is_moving = mean_variance > 0.001
        
        # 检测是否在回到初始位置
        if hasattr(self, 'initial_position'):
            current_position = recent_trajectory[-1]
            distance_to_initial = np.linalg.norm(current_position - self.initial_position)
            is_returning = distance_to_initial < 0.1
        else:
            is_returning = False
        
        # 记录分析结果（仅用于调试）
        self.last_trajectory_analysis = {
            'is_moving': is_moving,
            'mean_variance': mean_variance,
            'critical_variance': critical_variance,
            'is_returning': is_returning
        }
        
        # 每500步输出一次调试信息
        if self.total_steps_executed % 500 == 0:
            rospy.loginfo(f"轨迹分析: 运动中={is_moving}, 方差={mean_variance:.6f}, 关节方差={critical_variance:.6f}, 接近初始位置={is_returning}")
    
    def _should_end_action(self, recent_trajectory, current_position):
        """判断是否应该结束动作 - 简化版本"""
        if not hasattr(self, 'initial_position'):
            return False
        
        # 检查是否接近初始位置
        distance_to_initial = np.linalg.norm(current_position - self.initial_position)
        
        # 简单的结束条件：如果接近初始位置且已经执行了足够长的时间
        min_action_time = 300  # 至少300步才考虑结束
        has_enough_time = self.total_steps_executed > min_action_time
        
        # 如果接近初始位置且已经执行了足够长的时间，结束动作
        if distance_to_initial < 0.05 and has_enough_time:
            rospy.loginfo(f"动作结束条件满足: 距离初始位置={distance_to_initial:.6f}, 执行步数={self.total_steps_executed}")
            return True
        
        # 如果执行时间过长，也考虑结束
        max_action_time = 600  # 最多600步
        if self.total_steps_executed > max_action_time:
            rospy.loginfo(f"达到最大执行时间，结束动作")
            return True
        
        return False
    
    def _generate_ending_trajectory(self, current_position):
        """生成结束轨迹 - 平滑回到初始位置"""
        if not hasattr(self, 'initial_position'):
            return
        
        rospy.loginfo("生成结束轨迹，回到初始位置")
        
        # 生成一个简单的回到初始位置的轨迹
        ending_trajectory = []
        steps_to_return = 20  # 20步回到初始位置
        
        for i in range(steps_to_return):
            alpha = i / (steps_to_return - 1)
            interpolated_position = (1 - alpha) * current_position + alpha * self.initial_position
            ending_trajectory.append(interpolated_position.tolist())
        
        # 添加结束轨迹到缓冲区
        self.trajectory_buffer.extend(ending_trajectory)
        rospy.loginfo(f"结束轨迹生成完成 - 长度: {len(ending_trajectory)}")
        
        # 标记动作完成
        self.is_action_completed = True
        self.action_cycle_count += 1
        
        rospy.loginfo(f"动作周期 {self.action_cycle_count} 完成")
    
    def _select_trajectory_segment(self, predicted_actions, recent_trajectory):
        """智能选择轨迹段 - 确保动作完整展开"""
        # 分析当前动作进度
        if hasattr(self, 'initial_position'):
            current_position = recent_trajectory[-1]
            distance_to_initial = np.linalg.norm(current_position - self.initial_position)
            
            # 分析动作趋势
            trajectory_variance = np.var(recent_trajectory, axis=0)
            mean_variance = np.mean(trajectory_variance)
            
            # 根据动作进度选择合适的轨迹段
            if self.total_steps_executed < 50:
                # 动作初期：选择动作开始部分，让动作展开
                rospy.loginfo("选择动作开始部分 - 让动作展开")
                return predicted_actions[:12].copy()  # 取前12步
            elif distance_to_initial < 0.1 and mean_variance < 0.001 and self.total_steps_executed > 100:
                # 接近初始位置且动作平缓，且已经执行了一段时间：可能需要重新开始
                rospy.loginfo("检测到动作可能需要重新开始")
                return predicted_actions[:8].copy()  # 重新开始
            elif distance_to_initial > 0.3:
                # 远离初始位置：选择中间部分继续动作
                rospy.loginfo("选择动作中间部分 - 继续动作")
                mid_point = len(predicted_actions) // 2
                return predicted_actions[mid_point-6:mid_point+6].copy()  # 取中间12步
            else:
                # 其他情况：选择前半部分保持动作
                rospy.loginfo("选择动作前半部分 - 保持动作")
                return predicted_actions[:10].copy()  # 取前10步
        else:
            # 没有初始位置信息：选择前半部分
            rospy.loginfo("选择默认前半部分")
            return predicted_actions[:12].copy()
    
    def _ensure_trajectory_continuity(self, predicted_actions, start_position):
        """确保轨迹连续性 - 简单处理，让模型自己处理时序"""
        trajectory = predicted_actions.copy()
        
        # 只做最简单的起始位置修正，让模型自己处理时序
        trajectory[0] = start_position
        
        return trajectory
    
    def _should_regenerate_trajectory(self):
        """判断是否需要重新生成轨迹 - 简化版本"""
        # 如果动作已完成，不再生成
        if self.is_action_completed:
            return False
        
        # 如果轨迹缓冲区为空，需要生成
        if len(self.trajectory_buffer) == 0:
            return True
        
        # 如果剩余步数少于阈值，提前生成下一段轨迹
        remaining_steps = len(self.trajectory_buffer) - self.current_trajectory_step
        lookahead_threshold = 20  # 提前20步生成下一段轨迹
        
        if remaining_steps <= lookahead_threshold and not self.is_generating:
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
    
    def _fix_sharp_turning(self, previous_trajectory: np.ndarray, new_trajectory: np.ndarray) -> np.ndarray:
        """简单的位置连续性保证 - 不破坏模型预测内容"""
        try:
            # 只做最简单的处理：确保起始位置连续，保持模型预测完整性
            fixed_trajectory = new_trajectory.copy()
            fixed_trajectory[0] = previous_trajectory[-1]  # 确保位置连续
            
            rospy.loginfo("轨迹连接处理完成（仅保证位置连续）")
            return fixed_trajectory
            
        except Exception as e:
            rospy.logerr(f"轨迹连接处理失败: {e}")
            return new_trajectory  # 如果处理失败，返回原始轨迹
    
    def _generate_new_trajectory(self):
        """生成新的轨迹 - 集成急转弯修复算法"""
        try:
            if self.current_joint_positions is None:
                rospy.logwarn("当前关节位置未知，无法生成轨迹")
                return
            
            if self.is_generating:
                rospy.logdebug("正在生成轨迹，跳过")
                return
            
            self.is_generating = True
            
            # 确定轨迹起始位置
            if len(self.trajectory_buffer) > 0:
                # 使用当前轨迹的实际结束位置作为新轨迹的起始位置
                current_end_position = np.array(self.trajectory_buffer[-1])
                start_position = current_end_position
                
                # 获取上一段轨迹的最后几步用于平滑连接
                recent_steps = min(10, len(self.trajectory_buffer))
                previous_trajectory = np.array(self.trajectory_buffer[-recent_steps:])
            else:
                # 如果没有轨迹，使用当前机器人状态
                start_position = self.current_joint_positions.copy()
                # 记录初始位置
                self.initial_position = start_position.copy()
                previous_trajectory = None
            
            # 获取指令ID
            instruction_id = self.instruction_to_id.get(self.instruction, 0)
            
            # 标准化起始状态
            start_state_norm = (start_position - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            
            # 转换为tensor
            start_state_tensor = torch.FloatTensor(start_state_norm).unsqueeze(0).to(self.device)
            instruction_id_tensor = torch.LongTensor([instruction_id]).to(self.device)
            
            # 生成轨迹
            with torch.no_grad():
                outputs = self.model(start_state_tensor, instruction_id_tensor)
                predicted_actions_norm = outputs[0]  # 第一个输出是预测动作
            
            # 反标准化
            predicted_actions = predicted_actions_norm.cpu().numpy()[0] * self.norm_stats['action_std'] + self.norm_stats['action_mean']
            
            # 应用急转弯修复算法
            if previous_trajectory is not None:
                rospy.loginfo("应用急转弯修复算法...")
                trajectory = self._fix_sharp_turning(previous_trajectory, predicted_actions)
            else:
                # 第一次生成轨迹，只确保起始位置正确
                trajectory = predicted_actions.copy()
                trajectory[0] = start_position
            
            # 如果是第一次生成，直接设置缓冲区
            if len(self.trajectory_buffer) == 0:
                self.trajectory_buffer = trajectory.tolist()
                self.current_trajectory_step = 0
                rospy.loginfo(f"初始化轨迹缓冲区 - 长度: {len(self.trajectory_buffer)}")
            else:
                # 如果已经有轨迹，追加到缓冲区末尾
                self.trajectory_buffer.extend(trajectory.tolist())
                rospy.loginfo(f"追加新轨迹 - 缓冲区长度: {len(self.trajectory_buffer)}, 当前步: {self.current_trajectory_step}")
            
            rospy.loginfo(f"轨迹生成完成 - 本次生成长度: {len(trajectory)}")
            rospy.loginfo(f"轨迹范围: [{trajectory.min():.3f}, {trajectory.max():.3f}]")
            
            self.is_generating = False
            
        except Exception as e:
            rospy.logerr(f"生成轨迹失败: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.is_generating = False
    
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
        self.trajectory_buffer = []  # 清空轨迹缓冲区
        self.current_trajectory_step = 0
        self.next_trajectory = None
        self.is_generating = False
        rospy.loginfo("开始流畅的ACT推理")
        return TriggerResponse(True, "推理已开始")
    
    def _stop_callback(self, req):
        """停止推理回调"""
        if not self.is_running:
            return TriggerResponse(False, "推理未在进行中")
        
        self.is_running = False
        self.trajectory_buffer = []
        self.current_trajectory_step = 0
        self.next_trajectory = None
        self.is_generating = False
        rospy.loginfo("停止流畅的ACT推理")
        return TriggerResponse(True, "推理已停止")
    
    def start_inference(self):
        """开始推理 - 连续轨迹执行模式"""
        rospy.loginfo("流畅ACT推理节点已启动...")
        rospy.loginfo("推理已自动开始！")
        rospy.loginfo("使用以下命令控制推理:")
        rospy.loginfo("  停止推理: rosservice call /smooth_act_inference/stop")
        rospy.loginfo("  重新开始: rosservice call /smooth_act_inference/start")
        rospy.loginfo("按Ctrl+C退出")
        
        # 保持节点运行
        rate = rospy.Rate(self.inference_frequency)
        step_count = 0
        
        while not rospy.is_shutdown():
            if self.is_running:
                step_count += 1
                self.debug_counter += 1
                
                if self.current_joint_positions is not None:
                    # 每100步输出一次调试信息
                    if self.debug_counter % 100 == 0:
                        rospy.loginfo(f"调试信息: 步数={step_count}, 缓冲区长度={len(self.trajectory_buffer)}, 当前步={self.current_trajectory_step}")
                    
                    # 每5步检查一次是否需要重新生成轨迹
                    if step_count % 5 == 0:
                        if self._should_regenerate_trajectory():
                            self._generate_new_trajectory()
                    
                    # 执行当前步
                    if len(self.trajectory_buffer) > 0 and self.current_trajectory_step < len(self.trajectory_buffer):
                        current_action = np.array(self.trajectory_buffer[self.current_trajectory_step])
                        
                        # 发布控制命令
                        if self.publish_commands:
                            self._publish_arm_command(current_action)
                        
                        rospy.logdebug(f"执行步 {self.current_trajectory_step}/{len(self.trajectory_buffer)} - 动作范围: [{current_action.min():.3f}, {current_action.max():.3f}]")
                        
                        self.current_trajectory_step += 1
                        self.total_steps_executed += 1
                        
                        # 更新轨迹历史
                        self.trajectory_history.append(current_action.tolist())
                        if len(self.trajectory_history) > self.max_history_length:
                            self.trajectory_history = self.trajectory_history[-self.max_history_length:]
                        
                        # 检查是否应该结束动作
                        if len(self.trajectory_history) >= 16:
                            recent_trajectory = np.array(self.trajectory_history[-16:])
                            if self._should_end_action(recent_trajectory, current_action):
                                self._generate_ending_trajectory(current_action)
                        
                        # 如果执行完所有轨迹，输出提示
                        if self.current_trajectory_step >= len(self.trajectory_buffer):
                            rospy.loginfo("当前轨迹段执行完毕")
                    else:
                        # 没有有效轨迹时使用当前位置
                        if self.publish_commands:
                            self._publish_arm_command(self.current_joint_positions)
                        rospy.logdebug("没有有效轨迹，使用当前位置")
                else:
                    rospy.logdebug("等待关节位置数据...")
              
            rate.sleep()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='流畅的ACT推理节点')
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
    rospy.init_node('smooth_act_inference_node', anonymous=True)
    
    try:
        # 创建推理节点
        inference_node = SmoothACTInferenceNode(args.model_path, config)
        
        # 开始推理
        inference_node.start_inference()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS节点被中断")
    except Exception as e:
        rospy.logerr(f"推理节点运行出错: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()