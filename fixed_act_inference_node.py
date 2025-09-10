#!/usr/bin/env python3
"""
简化的ACT推理节点 - 使用正确的ROS话题结构
基于原始推理文件修复，使用SimpleACTGenerator模型
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import time
from typing import Dict, List, Tuple
import math
import argparse

# 导入手臂控制相关的消息类型
try:
    from kuavo_msgs.msg import armTargetPoses
    from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest
    HAS_ARM_MSGS = True
except ImportError:
    rospy.logwarn("无法导入 kuavo_msgs，手臂控制功能将被禁用")
    HAS_ARM_MSGS = False

class SimpleACTGenerator(nn.Module):
    """简化版ACT生成器 - 匹配训练模型"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 基础参数
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.num_instructions = config['num_instructions']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.trajectory_length = config['trajectory_length']
        self.dropout = config['dropout']
        
        # 简化的编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        
        # VAE潜在空间
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # 指令嵌入
        self.instruction_embedding = nn.Embedding(self.num_instructions, self.hidden_dim)
        
        # 简化的解码器 - 使用LSTM
        self.decoder = nn.LSTM(
            input_size=self.hidden_dim + self.latent_dim + self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=3,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(self.trajectory_length, self.hidden_dim))
        
        # KL权重
        self.kl_weight = config.get('kl_weight', 1e-4)
        
    def encode(self, states, actions):
        """编码"""
        # 直接使用原始状态和动作，确保训练-推理一致性
        inputs = torch.cat([states, actions], dim=-1)
        
        # 编码
        hidden = self.encoder(inputs)
        
        # VAE参数
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var
    
    def decode(self, latent, instruction_ids, start_states):
        """解码"""
        batch_size = latent.size(0)
        device = latent.device
        
        # 指令嵌入
        instruction_embed = self.instruction_embedding(instruction_ids)  # [batch_size, hidden_dim]
        
        # 扩展潜在向量和指令嵌入到时序
        if latent.dim() == 2:
            latent_expanded = latent.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        else:
            latent_expanded = latent.squeeze(1).expand(-1, self.trajectory_length, -1)
        
        instruction_expanded = instruction_embed.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        
        # 添加位置编码
        pos_enc = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 合并输入
        decoder_input = torch.cat([latent_expanded, instruction_expanded, pos_enc], dim=-1)
        
        # LSTM解码
        output, _ = self.decoder(decoder_input)
        
        # 预测动作
        actions = self.action_head(output)
        
        return actions
    
    def forward(self, start_states, instruction_ids, target_actions=None):
        """前向传播"""
        batch_size = start_states.size(0)
        device = start_states.device
        
        if self.training and target_actions is not None:
            # 训练时
            target_states = start_states.unsqueeze(1).expand(-1, self.trajectory_length, -1)
            
            # 编码
            mu, log_var = self.encode(target_states, target_actions)
            
            # 重参数化
            latent = self.reparameterize(mu, log_var)
            
            # 解码
            predicted_actions = self.decode(latent, instruction_ids, start_states)
            
            return predicted_actions, mu, log_var
        else:
            # 推理时
            latent = torch.randn(batch_size, self.latent_dim, device=device)
            
            # 解码
            predicted_actions = self.decode(latent, instruction_ids, start_states)
            
            return predicted_actions
    
    def reparameterize(self, mu, log_var):
        """重参数化"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

class FixedACTInferenceNode:
    """修复的ACT推理节点 - 连续无缝轨迹执行
    
    核心改进：
    1. 连续轨迹段生成 - 提前20步生成下一段轨迹
    2. 无缝衔接 - 轨迹段之间无停顿，连续执行
    3. 动态扩展 - 实时拼接轨迹段，形成长动作序列
    4. 基于原始推理文件的ROS话题结构
    
    执行逻辑：
    - 生成首段100步轨迹
    - 执行到第80步时，基于预测的第100步状态生成下一段
    - 无缝拼接轨迹段，形成连续的长动作
    - 避免段与段之间的停顿和等待
    """
    
    def __init__(self, model_path: str, instruction: str = '挥手'):
        rospy.init_node('fixed_act_inference_node')
        
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.norm_stats, self.config = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        rospy.loginfo(f"模型加载成功")
        rospy.loginfo(f"设备: {self.device}")
        rospy.loginfo(f"模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 推理参数
        self.trajectory_length = self.config['trajectory_length']
        self.instruction_map = {'挥手': 0, '抱拳': 1}  # 使用中文指令
        self.current_instruction = instruction
        
        rospy.loginfo(f"当前指令: {self.current_instruction}")
        
        # 状态缓冲区
        self.joint_positions = None
        self.state_buffer = []
        self.buffer_size = 16
        
        # 轨迹生成
        self.generated_trajectory = None
        self.current_trajectory_step = 0
        self.trajectory_start_state = None
        
        # ROS话题 - 使用正确的主题
        self.joint_state_sub = rospy.Subscriber(
            '/humanoid_controller/optimizedState_mrt/joint_pos',
            Float64MultiArray,
            self.joint_state_callback
        )
        
        # 手臂控制发布器
        if HAS_ARM_MSGS:
            self.arm_target_pub = rospy.Publisher(
                '/kuavo_arm_target_poses',
                armTargetPoses,
                queue_size=10
            )
            rospy.loginfo("手臂控制模式 - 发布手臂目标姿态")
        else:
            rospy.logerr("无法创建手臂发布器：缺少 kuavo_msgs")
        
        # 设置手臂控制模式
        self._setup_arm_control()
        
        # 控制线程
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        rospy.loginfo("修复的ACT推理节点启动完成 - 流畅执行模式")
        
    def load_model(self, model_path: str):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        config = checkpoint['config']
        norm_stats = checkpoint['norm_stats']
        
        # 重新创建模型
        model = SimpleACTGenerator(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, norm_stats, config
    
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
    
    def joint_state_callback(self, msg: Float64MultiArray):
        """关节数据回调"""
        if len(msg.data) >= 26:
            self.joint_positions = np.array(msg.data[:26])
            
            # 添加到缓冲区
            self.state_buffer.append(self.joint_positions.copy())
            if len(self.state_buffer) > self.buffer_size:
                self.state_buffer.pop(0)
    
    def generate_trajectory(self, start_state: np.ndarray, instruction: str):
        """生成轨迹"""
        try:
            # 获取指令ID
            instruction_id = self.instruction_map[instruction]
            
            # 标准化
            state_mean = self.norm_stats['state_mean']
            state_std = self.norm_stats['state_std']
            
            start_state_norm = (start_state - state_mean) / state_std
            
            # 转换为tensor
            start_tensor = torch.from_numpy(start_state_norm).float().unsqueeze(0).to(self.device)
            instruction_tensor = torch.tensor([instruction_id], dtype=torch.long).to(self.device)
            
            # 生成轨迹
            with torch.no_grad():
                predicted_actions_norm = self.model(start_tensor, instruction_tensor)
                
                # 反标准化
                action_mean = self.norm_stats['action_mean']
                action_std = self.norm_stats['action_std']
                
                predicted_actions = predicted_actions_norm.cpu().numpy()[0]
                predicted_actions = predicted_actions * action_std + action_mean
                
            return predicted_actions
            
        except Exception as e:
            rospy.logerr(f"生成轨迹失败: {e}")
            return None
    
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
    
    def control_loop(self):
        """控制循环 - 连续无缝轨迹执行"""
        rate = rospy.Rate(10)  # 10Hz
        
        while not rospy.is_shutdown():
            try:
                if self.joint_positions is not None and len(self.state_buffer) >= 5:
                    # 检查是否需要生成新轨迹
                    if (self.generated_trajectory is None or 
                        self.current_trajectory_step >= len(self.generated_trajectory) - 20):  # 提前20步生成下一段
                        
                        # 使用缓冲区中的最新状态作为下一段的起始状态
                        if self.generated_trajectory is not None:
                            # 如果当前正在执行轨迹，使用预测的最终状态作为起始点
                            predicted_final_state = self.generated_trajectory[-1]
                            start_state = predicted_final_state
                        else:
                            # 如果是第一次生成，使用当前机器人状态
                            start_state = np.mean(self.state_buffer[-5:], axis=0)
                        
                        if self.generated_trajectory is None:
                            rospy.loginfo(f"开始连续轨迹执行 - 指令: {self.current_instruction}")
                            rospy.loginfo(f"起始状态: [{start_state[12]:.3f}, {start_state[13]:.3f}, {start_state[14]:.3f}]")
                        
                        # 生成下一段轨迹
                        next_trajectory = self.generate_trajectory(start_state, self.current_instruction)
                        
                        if next_trajectory is not None:
                            if self.generated_trajectory is None:
                                # 第一次生成
                                self.generated_trajectory = next_trajectory
                                self.current_trajectory_step = 0
                                self.trajectory_start_state = start_state.copy()
                                rospy.loginfo(f"首段轨迹生成成功，长度: {len(next_trajectory)}")
                            else:
                                # 无缝衔接下一段轨迹
                                self.generated_trajectory = np.concatenate([self.generated_trajectory, next_trajectory], axis=0)
                                rospy.loginfo(f"轨迹段衔接成功，总长度: {len(self.generated_trajectory)}")
                        else:
                            rospy.logerr("轨迹段生成失败")
                            continue
                    
                    # 执行当前步
                    if self.generated_trajectory is not None:
                        current_action = self.generated_trajectory[self.current_trajectory_step]
                        
                        # 发布手臂控制命令
                        if HAS_ARM_MSGS:
                            self._publish_arm_command(current_action)
                        
                        # 更新步数
                        self.current_trajectory_step += 1
                        
                        # 每50步打印一次状态（避免日志过多）
                        if self.current_trajectory_step % 50 == 0:
                            rospy.loginfo(f"连续轨迹执行中 - 步数: {self.current_trajectory_step}")
                            rospy.loginfo(f"当前动作: [{current_action[12]:.3f}, {current_action[13]:.3f}, {current_action[14]:.3f}]")
                
                rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"控制循环错误: {e}")
                time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='修复的ACT推理节点')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--instruction', type=str, default='挥手', choices=['挥手', '抱拳'], help='指令类型')
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"模型文件不存在: {args.model_path}")
        exit(1)
    
    node = FixedACTInferenceNode(args.model_path, args.instruction)
    rospy.spin()