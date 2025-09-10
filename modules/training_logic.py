#!/usr/bin/env python3
"""
Diffusion Policy 训练逻辑
包含训练循环、损失函数和多任务混合训练策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Iterator
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import time
from tqdm import tqdm
import random
from scipy import ndimage
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
from collections import defaultdict

from modules.diffusion_policy import LargeBehaviorModel, create_large_behavior_model
from modules.data_preprocessing import ActionChunkPreprocessor


class EnhancedDiffusionLoss(nn.Module):
    """
    增强版Diffusion损失函数
    包含平滑性约束和物理一致性
    """
    
    def __init__(self, 
                 num_diffusion_steps: int = 1000,
                 beta_schedule: str = 'cosine',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 smoothness_weight: float = 0.02,
                 consistency_weight: float = 0.01,
                 velocity_weight: float = 0.15):
        super().__init__()
        
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_schedule = beta_schedule
        self.smoothness_weight = smoothness_weight
        self.consistency_weight = consistency_weight
        self.velocity_weight = velocity_weight
        
        # 计算噪声调度
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_diffusion_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # 注册为buffer
        self.register_buffer('betas', betas)
        
        # 计算alpha相关参数
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算扩散过程的参数
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        
        # 计算后验参数
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )
        posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        
        # 注册所有张量为buffer
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        
        # 关节限制（弗兰卡机械臂）
        self.joint_limits = torch.tensor([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,  # 关节1-7
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,  # 关节8-14
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,  # 关节15-21
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973                   # 关节22-26
        ])
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        改进的余弦调度 - 更好的收敛性
        """
        steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
        
        # 使用改进的余弦调度，在前几步和后几步有更平稳的过渡
        alpha_bar = torch.cos((steps + s) / (1 + s) * torch.pi * 0.5) ** 2
        
        # 添加线性插值以改善早期和晚期的稳定性
        if timesteps > 10:
            # 前10%使用更温和的调度
            early_phase = int(timesteps * 0.1)
            early_alpha = torch.linspace(1.0, alpha_bar[early_phase], early_phase + 1)
            alpha_bar[:early_phase + 1] = early_alpha
            
            # 后10%也使用温和调度
            late_phase = int(timesteps * 0.9)
            late_alpha = torch.linspace(alpha_bar[late_phase], alpha_bar[-1], timesteps - late_phase + 1)
            alpha_bar[late_phase:] = late_alpha
        
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        
        # 确保beta值在合理范围内
        betas = torch.clamp(betas, 1e-4, 0.999)
        
        return betas
    
    def q_sample(self, 
                 x_start: torch.Tensor, 
                 t: torch.Tensor, 
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向扩散过程
        
        Args:
            x_start: 原始数据 [batch_size, ...]
            t: 时间步 [batch_size]
            noise: 噪声 [batch_size, ...]
            
        Returns:
            带噪声的数据
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 确保索引和缓冲区在同一设备上
        device = x_start.device
        t = t.to(device)
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod.to(device)[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)[t].view(-1, 1, 1)
        
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise
    
    def compute_loss(self, 
                     model: LargeBehaviorModel,
                     x_start: torch.Tensor,
                     instruction_ids: torch.Tensor,
                     state: torch.Tensor,
                     visual_features: Optional[torch.Tensor] = None,
                     noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算增强的扩散损失
        
        Args:
            model: 模型
            x_start: 原始动作序列 [batch_size, chunk_length, action_dim]
            instruction_ids: 指令ID [batch_size]
            state: 状态信息 [batch_size, state_dim]
            visual_features: 视觉特征 [batch_size, visual_feature_dim]
            noise: 噪声 [batch_size, chunk_length, action_dim]
            
        Returns:
            损失字典
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 随机时间步
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        # 生成噪声
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_t = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = model(
            instruction_ids=instruction_ids,
            state=state,
            noisy_actions=x_t,
            timesteps=t,
            visual_features=visual_features
        )
        
        # 计算MSE损失 - 不使用标签平滑
        mse_loss = F.mse_loss(predicted_noise, noise, reduction='mean')
        
        # 计算平滑性损失
        smoothness_loss = self._compute_smoothness_loss(predicted_noise)
        
        # 计算物理一致性损失
        consistency_loss = self._compute_consistency_loss(x_start)
        
        # 计算速度连续性损失
        velocity_loss = self._compute_velocity_loss(x_start)
        
        # 计算时间一致性损失（改进局部特征复现）
        temporal_loss = self._compute_temporal_consistency_loss(x_start, predicted_noise)
        
        # 计算范围损失 - 强制输出在数据集范围内
        range_loss = self._compute_range_loss(x_start)
        
        # 添加曲线保真度损失
        curvature_loss = torch.tensor(0.0, device=x_start.device)
        if x_start.shape[1] > 2:
            # 计算原始数据的二阶差分（曲率特征）
            original_curvature = torch.diff(x_start, dim=1, n=2)
            
            # 使用预测噪声近似重构动作
            alpha_t = 1.0 - (t.float() / self.num_diffusion_steps).view(-1, 1, 1)
            predicted_x = x_start - predicted_noise * (1 - alpha_t).sqrt()
            predicted_curvature = torch.diff(predicted_x, dim=1, n=2)
            
            # 曲线形状保持损失
            curvature_loss = F.mse_loss(predicted_curvature, original_curvature)
        
        # 添加动态性激励 - 鼓励生成有变化的动作
        dynamic_loss = self._compute_dynamic_loss(x_start)
        
        # 总损失 - 极度简化，几乎只使用MSE损失
        total_loss = (mse_loss + 
                     0.001 * smoothness_loss +    # 极度减少平滑性约束
                     0.0005 * consistency_loss + # 极度减少一致性约束
                     0.001 * velocity_loss +      # 极度减少速度约束
                     0.001 * temporal_loss +      # 极度减少时间一致性
                     0.001 * curvature_loss +     # 极度减少曲线保真度
                     0.001 * range_loss +        # 极度减少范围限制
                     0.01 * dynamic_loss)        # 轻微鼓励动态性
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'smoothness_loss': smoothness_loss,
            'consistency_loss': consistency_loss,
            'velocity_loss': velocity_loss,
            'temporal_loss': temporal_loss,
            'curvature_loss': curvature_loss,  # 新增
            'range_loss': range_loss,  # 新增范围损失
            'dynamic_loss': dynamic_loss,  # 新增动态性损失
            'predicted_noise': predicted_noise,
            'target_noise': noise,
            'timesteps': t
        }
    
    def _compute_smoothness_loss(self, predicted_noise: torch.Tensor) -> torch.Tensor:
        """计算平滑性损失 - 极度宽松版本，几乎不限制"""
        # 只惩罚极端的高频变化
        noise_diff = torch.diff(predicted_noise, dim=1)
        
        # 极度宽松的限制
        max_reasonable_change = 10.0  # 允许非常大的变化
        extreme_changes = torch.relu(torch.abs(noise_diff) - max_reasonable_change)
        smoothness_loss = torch.mean(extreme_changes ** 2)
        
        return smoothness_loss
    
    def _compute_consistency_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """计算物理一致性损失 - 极度宽松版本"""
        device = x_start.device
        joint_limits = self.joint_limits.to(device)
        
        # 只惩罚极端的关节限制违反
        violations = torch.relu(torch.abs(x_start) - joint_limits.unsqueeze(0).unsqueeze(0) * 2.0)  # 允许超出2倍
        consistency_loss = torch.mean(violations ** 2) * 0.1  # 极度减少权重
        
        return consistency_loss
    
    def _compute_range_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """计算范围损失 - 极度宽松版本，基本不限制范围"""
        # 使用数据集的实际范围，但允许很大程度的超出
        data_min = -2.9459
        data_max = 1.4640
        
        # 允许超出数据集范围3倍，只防止极端值
        soft_min = data_min - 3.0 * (data_max - data_min)
        soft_max = data_max + 3.0 * (data_max - data_min)
        
        # 计算超出范围的惩罚 - 极度温和
        below_min = torch.relu(soft_min - x_start)
        above_max = torch.relu(x_start - soft_max)
        
        # 使用极小的惩罚权重
        range_loss = 0.01 * (torch.mean(below_min ** 2) + torch.mean(above_max ** 2))
        
        return range_loss
    
    def _compute_velocity_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """计算速度连续性损失 - 改进版本，允许合理运动"""
        if x_start.shape[1] <= 2:
            return torch.tensor(0.0, device=x_start.device)
        
        # 计算速度（一阶差分）
        velocities = torch.diff(x_start, dim=1)  # [batch_size, chunk_length-1, action_dim]
        
        # 计算加速度（二阶差分）
        if x_start.shape[1] > 2:
            accelerations = torch.diff(velocities, dim=1)  # [batch_size, chunk_length-2, action_dim]
            
            # 只惩罚极端的加速度，允许合理的运动
            max_reasonable_accel = 2.0
            extreme_accelerations = torch.relu(torch.abs(accelerations) - max_reasonable_accel)
            acceleration_loss = torch.mean(extreme_accelerations ** 2)
            
            # 轻微惩罚速度突变，但保留动态特征
            velocity_changes = torch.diff(velocities, dim=1)
            max_reasonable_jerk = 3.0
            extreme_jerks = torch.relu(torch.abs(velocity_changes) - max_reasonable_jerk)
            velocity_jerk_loss = torch.mean(extreme_jerks ** 2)
            
            return 0.1 * acceleration_loss + 0.05 * velocity_jerk_loss
        
        return torch.tensor(0.0, device=x_start.device)
    
    def _compute_temporal_consistency_loss(self, x_start: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        """计算时间一致性损失 - 改进局部特征复现"""
        if x_start.shape[1] <= 2:
            return torch.tensor(0.0, device=x_start.device)
        
        # 计算原始动作序列的局部模式
        local_patterns = []
        for i in range(x_start.shape[1] - 1):
            # 相邻帧的差异
            diff = x_start[:, i+1] - x_start[:, i]
            local_patterns.append(diff)
        
        if len(local_patterns) == 0:
            return torch.tensor(0.0, device=x_start.device)
        
        local_patterns = torch.stack(local_patterns, dim=1)  # [batch_size, seq_len-1, action_dim]
        
        # 鼓励预测噪声保持类似的时间结构
        noise_patterns = []
        for i in range(predicted_noise.shape[1] - 1):
            diff = predicted_noise[:, i+1] - predicted_noise[:, i]
            noise_patterns.append(diff)
        
        if len(noise_patterns) == 0:
            return torch.tensor(0.0, device=x_start.device)
        
        noise_patterns = torch.stack(noise_patterns, dim=1)
        
        # 计算模式相似性损失
        pattern_loss = F.mse_loss(noise_patterns, local_patterns, reduction='mean')
        
        # 添加峰值保持损失 - 特别关注重要特征的复现
        if x_start.shape[1] > 3:
            # 找到原始序列中的峰值
            x_std = torch.std(x_start, dim=1, keepdim=True)  # [batch_size, 1, action_dim]
            x_mean = torch.mean(x_start, dim=1, keepdim=True)  # [batch_size, 1, action_dim]
            
            # 识别峰值点（偏离均值超过标准差的点）
            peak_mask = torch.abs(x_start - x_mean) > 1.5 * x_std  # [batch_size, seq_len, action_dim]
            
            if peak_mask.any():
                # 对峰值点施加更强的约束 - 使用预测噪声的自相似性
                peak_predicted_noise = predicted_noise[peak_mask]
                if peak_predicted_noise.numel() > 0:
                    # 计算峰值点的预测噪声应该更接近零（更准确的预测）
                    peak_loss = F.mse_loss(
                        peak_predicted_noise, 
                        torch.zeros_like(peak_predicted_noise), 
                        reduction='mean'
                    )
                    pattern_loss += 0.5 * peak_loss
        
        return pattern_loss
    
    def _compute_dynamic_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """计算动态性损失 - 鼓励合理的动态变化"""
        if x_start.shape[1] <= 1:
            return torch.tensor(0.0, device=x_start.device)
        
        # 计算动作的变化量
        action_changes = torch.diff(x_start, dim=1)
        
        # 鼓励适度的变化，而不是惩罚静止
        change_magnitude = torch.abs(action_changes)
        target_change = 0.01  # 鼓励小幅度变化
        
        # 奖励接近目标变化的动作
        dynamic_reward = torch.exp(-torch.abs(change_magnitude - target_change))
        dynamic_loss = -torch.mean(dynamic_reward)  # 负损失 = 正奖励
        
        return dynamic_loss


class DataAugmentation:
    """
    数据增强类 - 提高数据多样性
    """
    
    def __init__(self, noise_std=0.01, time_warp_prob=0.3, scale_range=(0.9, 1.1)):
        self.noise_std = noise_std
        self.time_warp_prob = time_warp_prob
        self.scale_range = scale_range
    
    def add_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(actions) * self.noise_std
        return actions + noise
    
    def time_warp(self, actions: torch.Tensor) -> torch.Tensor:
        """时间扭曲增强"""
        if random.random() > self.time_warp_prob:
            return actions
            
        batch_size, seq_len, action_dim = actions.shape
        warped_actions = actions.clone()
        
        for i in range(batch_size):
            # 随机选择扭曲程度
            warp_factor = random.uniform(0.8, 1.2)
            
            # 应用时间扭曲
            if warp_factor != 1.0:
                # 使用线性插值进行时间扭曲
                original_indices = torch.arange(seq_len, dtype=torch.float32)
                warped_indices = original_indices * warp_factor
                warped_indices = torch.clamp(warped_indices, 0, seq_len - 1)
                
                for j in range(action_dim):
                    warped_actions[i, :, j] = torch.interp(
                        original_indices, 
                        warped_indices, 
                        actions[i, :, j]
                    )
        
        return warped_actions
    
    def scale_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """动作缩放增强"""
        scale_factor = random.uniform(*self.scale_range)
        return actions * scale_factor
    
    def random_dropout(self, actions: torch.Tensor, dropout_prob=0.1) -> torch.Tensor:
        """随机丢弃一些动作维度"""
        mask = torch.rand_like(actions) > dropout_prob
        return actions * mask.float()
    
    def augment(self, actions: torch.Tensor) -> torch.Tensor:
        """综合数据增强"""
        augmented = actions.clone()
        
        # 随机应用各种增强技术
        if random.random() < 0.8:  # 80%概率添加噪声
            augmented = self.add_noise(augmented)
        
        if random.random() < 0.3:  # 30%概率时间扭曲
            augmented = self.time_warp(augmented)
        
        if random.random() < 0.4:  # 40%概率缩放
            augmented = self.scale_actions(augmented)
        
        if random.random() < 0.2:  # 20%概率随机丢弃
            augmented = self.random_dropout(augmented)
        
        return augmented


class MultiTaskDataset(Dataset):
    """
    多任务数据集 - 支持数据增强
    """
    
    def __init__(self, 
                 processed_data_list: List[Dict[str, Any]],
                 state_dim: int = 60,
                 action_dim: int = 26,
                 instruction_map: Dict[str, int] = None,
                 use_augmentation: bool = True):
        """
        初始化数据集
        
        Args:
            processed_data_list: 处理后的数据列表
            state_dim: 状态维度
            action_dim: 动作维度
            instruction_map: 指令映射
        """
        self.processed_data_list = processed_data_list
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if instruction_map is None:
            instruction_map = {'wave': 0, 'welcome': 1}
        self.instruction_map = instruction_map
        self.instruction_to_name = {v: k for k, v in instruction_map.items()}
        
        # 数据增强
        self.use_augmentation = use_augmentation
        self.augmentation = DataAugmentation() if use_augmentation else None
        
        # 构建数据样本
        self.samples = self._build_samples()
    
    def _build_samples(self) -> List[Dict[str, Any]]:
        """构建数据样本"""
        samples = []
        
        for data in self.processed_data_list:
            task_name = data.get('task_name', 'unknown')
            instruction_id = self.instruction_map.get(task_name, 0)
            
            # 获取动作块
            action_chunks = data['action_chunks']
            
            # 构建状态信息（简化版本）
            if 'processed_observations' in data:
                observations = data['processed_observations']
            else:
                # 使用动作块的统计信息构建虚拟状态
                observations = self._build_virtual_states(action_chunks)
            
            # 为每个动作块创建样本
            for i, chunk in enumerate(action_chunks):
                # 构建状态（使用观测或虚拟状态）
                if i < len(observations):
                    obs = observations[i]
                    state = self._build_state_from_obs(obs)
                else:
                    state = self._build_virtual_state(chunk)
                
                sample = {
                    'task_name': task_name,
                    'instruction_id': instruction_id,
                    'action_chunk': chunk,
                    'state': state,
                    'chunk_index': i,
                    'total_chunks': len(action_chunks)
                }
                samples.append(sample)
        
        return samples
    
    def _build_virtual_states(self, action_chunks: List[List[float]]) -> List[Dict[str, Any]]:
        """构建虚拟状态"""
        virtual_states = []
        
        for chunk in action_chunks:
            # 使用动作块的统计信息构建虚拟状态
            chunk_array = np.array(chunk)
            state = {
                'joint_pos': chunk_array[0].tolist(),  # 使用第一帧作为位置
                'joint_vel': np.zeros(self.action_dim).tolist(),  # 速度设为0
                'joint_torque': np.zeros(self.action_dim).tolist(),  # 力矩设为0
            }
            virtual_states.append(state)
        
        return virtual_states
    
    def _build_state_from_obs(self, obs: Dict[str, Any]) -> List[float]:
        """从观测构建状态"""
        state = np.zeros(self.state_dim)
        
        # 填充关节信息
        if 'joint_pos' in obs:
            pos = np.array(obs['joint_pos'])
            state[:min(len(pos), self.action_dim)] = pos[:min(len(pos), self.action_dim)]
        
        if 'joint_vel' in obs:
            vel = np.array(obs['joint_vel'])
            start_idx = self.action_dim
            end_idx = min(start_idx + len(vel), self.state_dim)
            state[start_idx:end_idx] = vel[:end_idx - start_idx]
        
        if 'joint_torque' in obs:
            torque = np.array(obs['joint_torque'])
            start_idx = self.action_dim * 2
            end_idx = min(start_idx + len(torque), self.state_dim)
            state[start_idx:end_idx] = torque[:end_idx - start_idx]
        
        return state.tolist()
    
    def _build_virtual_state(self, chunk: List[float]) -> List[float]:
        """构建虚拟状态"""
        state = np.zeros(self.state_dim)
        
        # 使用动作块的第一帧作为关节位置
        state[:self.action_dim] = chunk[0]
        
        # 计算速度（简化）
        if len(chunk) > 1:
            velocity = np.array(chunk[1]) - np.array(chunk[0])
            state[self.action_dim:self.action_dim*2] = velocity
        
        return state.tolist()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # 禁用数据增强 - 直接使用原始数据
        action_chunk = torch.tensor(sample['action_chunk'], dtype=torch.float32)
        
        return {
            'instruction_id': torch.tensor(sample['instruction_id'], dtype=torch.long),
            'action_chunk': action_chunk,
            'state': torch.tensor(sample['state'], dtype=torch.float32),
            'task_name': sample['task_name'],
            'chunk_index': sample['chunk_index']
        }


class DiffusionTrainer:
    """
    Diffusion 训练器 - 支持标签平滑
    """
    
    def __init__(self, 
                 model: LargeBehaviorModel,
                 loss_fn: EnhancedDiffusionLoss,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints'):
        """
        初始化训练器
        
        Args:
            model: 模型
            loss_fn: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            save_dir: 保存目录
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 早停相关
        self.patience = config.get('patience', 15)  # 早停耐心值
        self.no_improvement_count = 0
        self.early_stopping = False
        
        # 日志记录
        self.train_losses = []
        self.val_losses = []
        self.task_losses = defaultdict(list)
    
    def train_epoch(self, 
                   dataloader: DataLoader,
                   epoch: int,
                   log_interval: int = 100) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            log_interval: 日志间隔
            
        Returns:
            训练统计
        """
        self.model.train()
        epoch_losses = []
        task_losses = defaultdict(list)
        
        pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据转移到设备
            instruction_ids = batch['instruction_id'].to(self.device)
            action_chunks = batch['action_chunk'].to(self.device)
            states = batch['state'].to(self.device)
            task_names = batch['task_name']
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 计算损失
            loss_dict = self.loss_fn.compute_loss(
                model=self.model,
                x_start=action_chunks,
                instruction_ids=instruction_ids,
                state=states
            )
            
            loss = loss_dict['loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            epoch_losses.append(loss.item())
            
            # 按任务记录损失
            for task_name in task_names:
                task_losses[task_name].append(loss.item())
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 日志记录
            if batch_idx % log_interval == 0:
                avg_loss = np.mean(epoch_losses[-log_interval:])
                pbar.set_postfix({
                    'loss': f'{avg_loss:.6f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Wandb记录
                if HAS_WANDB and wandb.run is not None:
                    wandb.log({
                        'train_loss': avg_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'global_step': self.global_step
                    })
            
            self.global_step += 1
        
        # 计算epoch统计
        epoch_stats = {
            'mean_loss': np.mean(epoch_losses),
            'std_loss': np.std(epoch_losses),
            'min_loss': np.min(epoch_losses),
            'max_loss': np.max(epoch_losses)
        }
        
        # 按任务统计
        for task_name, losses in task_losses.items():
            epoch_stats[f'{task_name}_mean_loss'] = np.mean(losses)
            self.task_losses[task_name].extend(losses)
        
        self.train_losses.extend(epoch_losses)
        
        return epoch_stats
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            验证统计
        """
        self.model.eval()
        val_losses = []
        task_losses = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                instruction_ids = batch['instruction_id'].to(self.device)
                action_chunks = batch['action_chunk'].to(self.device)
                states = batch['state'].to(self.device)
                task_names = batch['task_name']
                
                # 计算损失
                loss_dict = self.loss_fn.compute_loss(
                    model=self.model,
                    x_start=action_chunks,
                    instruction_ids=instruction_ids,
                    state=states
                )
                
                loss = loss_dict['loss'].item()
                val_losses.append(loss)
                
                # 按任务记录损失
                for task_name in task_names:
                    task_losses[task_name].append(loss)
        
        # 计算验证统计
        val_stats = {
            'mean_loss': np.mean(val_losses),
            'std_loss': np.std(val_losses),
            'min_loss': np.min(val_losses),
            'max_loss': np.max(val_losses)
        }
        
        # 按任务统计
        for task_name, losses in task_losses.items():
            val_stats[f'{task_name}_mean_loss'] = np.mean(losses)
        
        self.val_losses.extend(val_losses)
        
        return val_stats
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'task_losses': dict(self.task_losses)
        }
        
        # 保存当前检查点
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_loss = val_loss
            print(f"🎯 New best model saved with loss: {val_loss:.6f}")
    
    def train(self, 
             train_dataloader: DataLoader,
             val_dataloader: DataLoader,
             num_epochs: int,
             save_interval: int = 10,
             log_interval: int = 100,
             use_wandb: bool = False):
        """
        完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            num_epochs: 训练epoch数
            save_interval: 保存间隔
            log_interval: 日志间隔
            use_wandb: 是否使用wandb
        """
        print(f"🚀 开始训练 {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练
            train_stats = self.train_epoch(train_dataloader, epoch, log_interval)
            
            # 验证
            val_stats = self.validate(val_dataloader)
            
            # 打印统计信息
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"  Train Loss: {train_stats['mean_loss']:.6f} ± {train_stats['std_loss']:.6f}")
            print(f"  Val Loss: {val_stats['mean_loss']:.6f} ± {val_stats['std_loss']:.6f}")
            
            # 按任务打印统计
            for key, value in train_stats.items():
                if 'mean_loss' in key and key != 'mean_loss':
                    task_name = key.replace('_mean_loss', '')
                    print(f"  {task_name} Train Loss: {value:.6f}")
            
            # Wandb记录
            if use_wandb and HAS_WANDB:
                wandb.log({
                    'epoch': epoch,
                    'train_mean_loss': train_stats['mean_loss'],
                    'val_mean_loss': val_stats['mean_loss'],
                    'val_std_loss': val_stats['std_loss']
                })
            
            # 保存检查点
            is_best = val_stats['mean_loss'] < self.best_loss
            
            # 早停检查
            if is_best:
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_stats['mean_loss'], is_best)
            
            # 早停逻辑
            if self.no_improvement_count >= self.patience:
                print(f"\n⏹️ 早停触发: {self.patience} 个epoch没有改善")
                print(f"最佳验证损失: {self.best_loss:.6f}")
                self.early_stopping = True
                break
        
        print("🎉 训练完成！")


def create_trainer(config: Dict[str, Any]) -> DiffusionTrainer:
    """
    创建训练器
    
    Args:
        config: 配置字典
        
    Returns:
        训练器实例
    """
    # 创建模型
    model = create_large_behavior_model(config.get('model_config', {}))
    
    # 创建损失函数
    loss_fn = EnhancedDiffusionLoss(
        num_diffusion_steps=config.get('num_diffusion_steps', 1000),
        beta_schedule=config.get('beta_schedule', 'cosine'),
        smoothness_weight=config.get('smoothness_weight', 0.05),  # 减少平滑性约束
        consistency_weight=config.get('consistency_weight', 0.02),  # 减少一致性约束
        velocity_weight=config.get('velocity_weight', 0.01)  # 减少速度约束
    )
    
    # 创建优化器 - 增加L2正则化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 1e-3),  # 增加权重衰减
        betas=(0.9, 0.999)
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('num_epochs', 100),
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # 创建训练器 - 添加早停配置
    trainer_config = {
        'device': config.get('device', 'cuda'),
        'save_dir': config.get('save_dir', './checkpoints'),
        'patience': config.get('patience', 15)  # 早停耐心值
    }
    
    trainer = DiffusionTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        **trainer_config
    )
    
    return trainer


def test_training_components():
    """测试训练组件"""
    print("测试训练组件...")
    
    # 创建虚拟数据
    batch_size = 4
    chunk_length = 16
    action_dim = 26
    state_dim = 60
    
    # 创建模型
    model = create_large_behavior_model({
        'action_dim': action_dim,
        'chunk_length': chunk_length,
        'state_dim': state_dim,
        'diffusion_dim': 256,
        'num_diffusion_steps': 100
    })
    
    # 创建损失函数
    loss_fn = EnhancedDiffusionLoss(num_diffusion_steps=100)
    
    # 测试数据
    instruction_ids = torch.tensor([0, 1, 0, 1])
    state = torch.randn(batch_size, state_dim)
    action_chunks = torch.randn(batch_size, chunk_length, action_dim)
    
    # 测试损失计算
    loss_dict = loss_fn.compute_loss(model, action_chunks, instruction_ids, state)
    print(f"损失形状: {loss_dict['loss'].shape}")
    print(f"损失值: {loss_dict['loss'].item():.6f}")
    
    # 创建训练器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = DiffusionTrainer(model, loss_fn, optimizer, device='cpu')
    
    print("训练组件测试完成！")


if __name__ == "__main__":
    test_training_components()