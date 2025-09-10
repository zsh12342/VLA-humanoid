"""
数据集模块 - 用于加载和处理轨迹数据

支持从JSON格式的轨迹文件加载三元组数据：
- 指令 (instruction)
- 观测 (observation: image + state)
- 动作标签 (skill_id + skill_params)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import json
import numpy as np
from PIL import Image
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys
from collections import defaultdict
import random

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class TrajectoryDataset(Dataset):
    """
    轨迹数据集类 - 增强版本支持时序输入和绝对角度预测
    
    从JSON轨迹文件加载三元组数据，支持数据增强和预处理
    新增功能：
    - 历史观测窗口（3-5帧）
    - 绝对关节角度预测
    - 改进的数据预处理一致性
    """
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 augment: bool = True, max_seq_len: int = 20,
                 sequence_length: int = 10,
                 history_window_size: int = 5,  # 新增：历史窗口大小
                 predict_absolute_angles: bool = True,  # 新增：预测绝对角度
                 trajectories: List[Dict[str, Any]] = None,
                 vocab: Dict[str, int] = None):
        """
        初始化轨迹数据集
        
        Args:
            data_dir: 数据目录路径
            split: 数据分割 ('train', 'val', 'test')
            augment: 是否进行数据增强
            max_seq_len: 最大序列长度
            trajectories: 预加载的轨迹数据（用于数据分割）
            vocab: 预构建的词汇表（用于数据分割）
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.max_seq_len = max_seq_len
        self.sequence_length = sequence_length
        self.history_window_size = history_window_size  # 新增
        self.predict_absolute_angles = predict_absolute_angles  # 新增
        
        # 加载轨迹数据
        if trajectories is not None:
            self.trajectories = trajectories
        else:
            self.trajectories = self._load_trajectories()
        
        # 构建词汇表
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # 图像预处理
        self.image_transform = self._get_image_transform()
        
        # 统计信息
        self.skill_counts = self._count_skills()
        # 新增：数据统计
        self._compute_data_statistics()
        print(f"Loaded {len(self.trajectories)} trajectories for {split} split")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Skill distribution: {self.skill_counts}")
        print(f"History window size: {self.history_window_size}")
        print(f"Predict absolute angles: {self.predict_absolute_angles}")
        if hasattr(self, 'action_stats'):
            print(f"Action stats - Mean: {self.action_stats['mean']:.4f}, Std: {self.action_stats['std']:.4f}")
    
    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """加载轨迹数据"""
        trajectories = []
        
        # 遍历数据目录
        for file_path in self.data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    trajectory = json.load(f)
                    trajectories.append(trajectory)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
        
        return trajectories
    
    def _build_vocab(self) -> Dict[str, int]:
        """构建词汇表"""
        vocab = defaultdict(int)
        vocab_count = defaultdict(int)
        
        # 统计词频
        for trajectory in self.trajectories:
            instruction = trajectory.get('instruction', '')
            words = instruction.split()
            for word in words:
                vocab_count[word] += 1
        
        # 过滤低频词
        min_freq = 2
        filtered_words = [word for word, count in vocab_count.items() if count >= min_freq]
        
        # 构建词汇表
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i, word in enumerate(filtered_words):
            vocab[word] = i + 2
        
        return dict(vocab)
    
    def _get_image_transform(self) -> transforms.Compose:
        """获取图像预处理变换"""
        base_transform = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]
        
        # 训练时添加数据增强
        if self.augment and self.split == 'train':
            augment_transform = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ]
            base_transform = base_transform[:2] + augment_transform + base_transform[2:]
        
        return transforms.Compose(base_transform)
    
    def _count_skills(self) -> Dict[int, int]:
        """统计技能分布"""
        skill_counts = defaultdict(int)
        for trajectory in self.trajectories:
            skill_id = trajectory.get('skill_id', 'unknown')
            skill_id_mapped = self._skill_name_to_id(skill_id)
            skill_counts[skill_id_mapped] += 1
        return dict(skill_counts)
    
    def _skill_name_to_id(self, skill_name: str) -> int:
        """将技能名称转换为ID"""
        skill_map = {
            'walk_forward': 0, 'walk_back': 1, 'turn_left': 2, 'turn_right': 3
        }
        return skill_map.get(skill_name.lower(), 0)
    
    def _tokenize_instruction(self, instruction: str) -> List[int]:
        """将指令转换为token序列"""
        words = instruction.split()
        tokens = []
        for word in words:
            token = self.vocab.get(word, self.vocab['<UNK>'])
            tokens.append(token)
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            tokens.extend([self.vocab['<PAD>']] * (self.max_seq_len - len(tokens)))
        
        return tokens
    
    def _process_observation(self, obs: Dict[str, Any], t_norm: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """处理观测数据 - 增强版本，添加时间戳和滑动窗口支持"""
        # 处理图像 - 如果图像为null，创建空白图像
        if 'image' in obs and obs['image'] is not None:
            if isinstance(obs['image'], list):
                # 从列表重建图像
                image_array = np.array(obs['image'], dtype=np.uint8)
                if 'image_shape' in obs:
                    image_array = image_array.reshape(obs['image_shape'])
            else:
                image_array = obs['image'].astype(np.uint8)
            
            # 确保图像是3通道的
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif image_array.shape[-1] == 1:
                image_array = np.repeat(image_array, 3, axis=-1)
            
            image_tensor = self.image_transform(image_array)
        else:
            # 创建空白图像
            image_tensor = torch.zeros(3, 224, 224)
        
        # 处理状态向量 - 改进的数据一致性
        joint_angles = np.array(obs.get('joint_pos', []))
        joint_velocities = np.array(obs.get('joint_vel', []))
        root_pos = np.array(obs.get('root_pos', []))
        root_orien = np.array(obs.get('root_orien', []))
        
        # 改进：确保数据类型一致性
        joint_angles = joint_angles.astype(np.float32)
        joint_velocities = joint_velocities.astype(np.float32)
        root_pos = root_pos.astype(np.float32)
        root_orien = root_orien.astype(np.float32)
        
        # 改进：处理缺失数据
        if len(joint_angles) == 0:
            joint_angles = np.zeros(45, dtype=np.float32)
        if len(joint_velocities) == 0:
            joint_velocities = np.zeros(45, dtype=np.float32)
        if len(root_pos) == 0:
            root_pos = np.zeros(3, dtype=np.float32)
        if len(root_orien) == 0:
            root_orien = np.array([0, 0, 0, 1], dtype=np.float32)  # 单位四元数
        
        # 创建模拟IMU数据（如果不存在）
        imu = np.zeros(9, dtype=np.float32)  # 简化的IMU数据
        
        # 拼接状态向量 - 新增：添加归一化时间戳
        state_vector = np.concatenate([
            joint_angles.flatten(),
            joint_velocities.flatten(),
            imu.flatten(),
            root_pos.flatten(),
            root_orien.flatten(),
            [t_norm]  # 新增：归一化时间戳
        ])
        
        # 确保状态向量维度正确（新增时间戳维度）
        target_dim = config.PERCEPTION['state_dim'] + 1  # 增加时间戳维度
        if len(state_vector) < target_dim:
            state_vector = np.pad(state_vector, (0, target_dim - len(state_vector)))
        elif len(state_vector) > target_dim:
            state_vector = state_vector[:target_dim]
        
        state_tensor = torch.tensor(state_vector, dtype=torch.float32)
        
        return image_tensor, state_tensor
    
    def _compute_data_statistics(self):
        """计算数据统计信息，用于改进数据预处理一致性"""
        all_actions = []
        all_states = []
        
        for trajectory in self.trajectories:
            actions = trajectory.get('actions', [])
            observations = trajectory.get('observations', [])
            
            # 收集动作数据
            for action in actions:
                if action:
                    all_actions.append(np.array(action, dtype=np.float32))
            
            # 收集状态数据
            for obs in observations:
                joint_pos = obs.get('joint_pos', [])
                if joint_pos:
                    all_states.append(np.array(joint_pos, dtype=np.float32))
        
        # 计算统计信息
        if all_actions:
            all_actions = np.array(all_actions)
            self.action_stats = {
                'mean': float(np.mean(all_actions)),
                'std': float(np.std(all_actions)),
                'min': float(np.min(all_actions)),
                'max': float(np.max(all_actions))
            }
        else:
            self.action_stats = {'mean': 0.0, 'std': 1.0, 'min': -1.0, 'max': 1.0}
        
        if all_states:
            all_states = np.array(all_states)
            self.state_stats = {
                'mean': float(np.mean(all_states)),
                'std': float(np.std(all_states)),
                'min': float(np.min(all_states)),
                'max': float(np.max(all_states))
            }
        else:
            self.state_stats = {'mean': 0.0, 'std': 1.0, 'min': -1.0, 'max': 1.0}
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """标准化动作数据 - 改进一致性"""
        if hasattr(self, 'action_stats'):
            # 使用统计信息进行标准化
            if self.action_stats['std'] > 0:
                action = (action - self.action_stats['mean']) / self.action_stats['std']
            else:
                action = action - self.action_stats['mean']
        
        # 确保在合理范围内
        action = np.clip(action, -5.0, 5.0)  # 防止极端值
        
        return action
    
    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """反标准化动作数据"""
        if hasattr(self, 'action_stats'):
            if self.action_stats['std'] > 0:
                action = action * self.action_stats['std'] + self.action_stats['mean']
            else:
                action = action + self.action_stats['mean']
        
        return action
    
    def _extract_skill_params_from_sequence(self, trajectory: Dict[str, Any], start_idx: int, seq_length: int) -> torch.Tensor:
        """从序列数据中提取技能参数"""
        skill_id = trajectory.get('skill_id', 'unknown')
        observations = trajectory.get('observations', [])
        actions = trajectory.get('actions', [])
        
        # 基础参数
        params = [
            float(seq_length),  # 序列长度
            float(len(observations)),  # 总帧数
            float(start_idx),  # 起始位置
        ]
        
        # 从实际数据中提取运动特征
        if observations and start_idx < len(observations):
            # 计算运动特征
            joint_positions = []
            root_positions = []
            
            for i in range(min(seq_length, len(observations) - start_idx)):
                obs = observations[start_idx + i]
                joint_pos = obs.get('joint_pos', [])
                root_pos = obs.get('root_pos', [])
                
                if joint_pos:
                    joint_positions.append(joint_pos)
                if root_pos:
                    root_positions.append(root_pos)
            
            # 计算运动统计特征
            if joint_positions:
                joint_positions = np.array(joint_positions)
                # 关节运动范围
                joint_range = np.max(joint_positions, axis=0) - np.min(joint_positions, axis=0)
                params.extend([
                    np.mean(joint_range),  # 平均关节运动范围
                    np.std(joint_range),   # 关节运动变化
                ])
            else:
                params.extend([0.0, 0.0])
            
            # 计算位移特征
            if root_positions and len(root_positions) > 1:
                root_positions = np.array(root_positions)
                displacement = np.linalg.norm(root_positions[-1] - root_positions[0])
                params.extend([
                    float(displacement),  # 总位移
                    float(displacement / seq_length),  # 平均速度
                ])
            else:
                params.extend([0.0, 0.0])
        else:
            params.extend([0.0, 0.0, 0.0, 0.0])
        
        # 技能特定参数
        if skill_id == 'walk_forward':
            params.extend([1.0, 0.0])  # 前进标识
        elif skill_id == 'walk_back':
            params.extend([0.0, 1.0])  # 后退标识
        elif skill_id == 'turn_left':
            params.extend([1.0, 0.0])  # 左转标识
        elif skill_id == 'turn_right':
            params.extend([0.0, 1.0])  # 右转标识
        else:
            params.extend([0.0, 0.0])
        
        # 确保参数维度正确
        target_dim = config.ALGORITHM['planner']['param_dim']
        if len(params) < target_dim:
            params.extend([0.0] * (target_dim - len(params)))
        elif len(params) > target_dim:
            params = params[:target_dim]
        
        return torch.tensor(params, dtype=torch.float32)
    
    def _extract_skill_params(self, trajectory: Dict[str, Any]) -> torch.Tensor:
        """提取技能参数"""
        # 从轨迹中提取参数
        skill_id = trajectory.get('skill_id', 'unknown')
        metadata = trajectory.get('metadata', {})
        num_observations = len(trajectory.get('observations', []))
        
        # 基础参数
        params = [
            metadata.get('duration', 5.0),  # 持续时间
            metadata.get('frequency', 50.0),  # 频率
            num_observations,  # 步数
        ]
        
        # 技能特定参数 - 从metadata中提取实际参数
        if skill_id == 'walk_forward':
            params.extend([
                metadata.get('stride_length', 0.3),  # 步长
                metadata.get('walk_speed', 1.0),  # 速度
                1.0  # 前进方向标识
            ])
        elif skill_id == 'walk_back':
            params.extend([
                metadata.get('stride_length', 0.3),  # 步长
                metadata.get('walk_speed', 1.0),  # 速度
                0.0  # 后退方向标识
            ])
        elif skill_id == 'turn_left':
            params.extend([
                metadata.get('turn_angle', 45.0),  # 转向角度
                metadata.get('turn_speed', 1.0),  # 速度
                1.0  # 左转标识
            ])
        elif skill_id == 'turn_right':
            params.extend([
                metadata.get('turn_angle', 45.0),  # 转向角度
                metadata.get('turn_speed', 1.0),  # 速度
                0.0  # 右转标识
            ])
        else:
            # 默认参数
            params.extend([0.0, 0.0, 0.0])
        
        # 确保参数维度正确
        target_dim = config.ALGORITHM['planner']['param_dim']
        if len(params) < target_dim:
            params.extend([0.0] * (target_dim - len(params)))
        elif len(params) > target_dim:
            params = params[:target_dim]
        
        return torch.tensor(params, dtype=torch.float32)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        # 每条轨迹生成多个序列样本
        return len(self.trajectories) * 10  # 每条轨迹生成10个序列
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取序列数据样本 - 增强版本支持历史窗口和绝对角度
        
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典：
                - instruction_ids: 指令ID（简化版）
                - state_sequences: 状态序列 [batch_size, seq_len, state_dim]
                - action_sequences: 动作序列 [batch_size, seq_len, action_dim]
                - past_actions: 过去动作 [batch_size, history_len, action_dim]（新增）
                - skill_id: 技能ID
                - skill_params: 技能参数
        """
        # 选择轨迹和序列起始位置
        trajectory_idx = idx // 10
        sequence_idx = idx % 10
        
        trajectory = self.trajectories[trajectory_idx]
        observations = trajectory.get('observations', [])
        actions = trajectory.get('actions', [])
        
        # 处理指令 - 简化为ID
        instruction = trajectory.get('instruction', '')
        skill_id = trajectory.get('skill_id', 'unknown')
        instruction_id = self._instruction_to_id(instruction, skill_id)
        instruction_tensor = torch.tensor(instruction_id, dtype=torch.int32)
        
        # 计算序列起始位置
        total_frames = len(observations)
        if total_frames <= self.sequence_length:
            # 如果轨迹不够长，从头开始
            start_idx = 0
            seq_length = total_frames
        else:
            # 随机选择序列起始位置
            max_start = total_frames - self.sequence_length
            start_idx = sequence_idx * (max_start // 10)
            seq_length = self.sequence_length
        
        # 处理观测序列 - 增强版本
        image_sequence = []
        state_sequence = []
        action_sequence = []
        past_actions = []  # 新增：历史动作
        
        for i in range(seq_length):
            obs_idx = start_idx + i
            if obs_idx < len(observations):
                obs = observations[obs_idx]
                # 新增：计算归一化时间戳
                t_norm = obs_idx / max(len(observations) - 1, 1)  # 归一化到 [0, 1]
                
                image_tensor, state_tensor = self._process_observation(obs, t_norm)
                image_sequence.append(image_tensor)
                state_sequence.append(state_tensor)
                
                # 处理对应的动作 - 改进：支持绝对角度预测
                if obs_idx < len(actions):
                    action = actions[obs_idx]
                    if self.predict_absolute_angles:
                        # 预测绝对关节角度
                        action_tensor = torch.tensor(action, dtype=torch.float32)
                    else:
                        # 预测关节差值
                        if obs_idx > 0:
                            prev_action = actions[obs_idx - 1]
                            action_diff = np.array(action) - np.array(prev_action)
                            action_tensor = torch.tensor(action_diff, dtype=torch.float32)
                        else:
                            action_tensor = torch.tensor(action, dtype=torch.float32)
                    
                    # 应用标准化
                    if isinstance(action, np.ndarray):
                        action_normalized = self._normalize_action(action)
                    else:
                        action_normalized = self._normalize_action(np.array(action))
                    action_tensor = torch.tensor(action_normalized, dtype=torch.float32)
                    action_sequence.append(action_tensor)
        
        # 如果序列不够长，进行填充
        while len(image_sequence) < self.sequence_length:
            image_sequence.append(torch.zeros(3, 224, 224))
            # 填充时使用t_norm=1.0表示结束
            _, padded_state = self._process_observation({}, 1.0)
            state_sequence.append(padded_state)
            action_sequence.append(torch.zeros(len(actions[0]) if actions else 26))
        
        # 新增：构建历史动作窗口
        history_start = max(0, start_idx - self.history_window_size)
        for i in range(history_start, start_idx):
            if i < len(actions):
                action = actions[i]
                if self.predict_absolute_angles:
                    past_action = torch.tensor(action, dtype=torch.float32)
                else:
                    if i > 0:
                        prev_action = actions[i - 1]
                        action_diff = np.array(action) - np.array(prev_action)
                        past_action = torch.tensor(action_diff, dtype=torch.float32)
                    else:
                        past_action = torch.tensor(action, dtype=torch.float32)
                
                # 应用标准化
                past_action = torch.tensor(self._normalize_action(past_action.numpy()), dtype=torch.float32)
                past_actions.append(past_action)
        
        # 填充历史动作
        while len(past_actions) < self.history_window_size:
            past_actions.append(torch.zeros(len(actions[0]) if actions else 26))
        
        # 新增：构建滑动窗口历史观测
        history_observations = []
        for i in range(max(0, start_idx - self.history_window_size), start_idx):
            if i < len(observations):
                obs = observations[i]
                t_norm = i / max(len(observations) - 1, 1)
                _, state_tensor = self._process_observation(obs, t_norm)
                history_observations.append(state_tensor)
        
        # 填充历史观测
        while len(history_observations) < self.history_window_size:
            _, padded_state = self._process_observation({}, 0.0)
            history_observations.append(padded_state)
        
        # 堆叠序列
        image_sequence = torch.stack(image_sequence, dim=0)  # [seq_len, 3, 224, 224]
        state_sequence = torch.stack(state_sequence, dim=0)  # [seq_len, state_dim+1]
        action_sequence = torch.stack(action_sequence, dim=0)  # [seq_len, action_dim]
        past_actions = torch.stack(past_actions, dim=0)  # [history_len, action_dim]
        history_observations = torch.stack(history_observations, dim=0)  # [history_len, state_dim+1]
        
        # 处理技能标签
        skill_id_mapped = self._skill_name_to_id(skill_id)
        skill_id_tensor = torch.tensor(skill_id_mapped, dtype=torch.int32)
        
        # 处理技能参数 - 从轨迹数据中提取实际参数
        skill_params_tensor = self._extract_skill_params_from_sequence(trajectory, start_idx, seq_length)
        
        return {
            'instruction_ids': instruction_tensor,  # 改为简化的ID
            'image_sequence': image_sequence,
            'state_sequence': state_sequence,  # 现在包含时间戳
            'action_sequences': action_sequence,  # 改为复数形式
            'past_actions': past_actions,  # 新增
            'history_observations': history_observations,  # 新增：滑动窗口历史观测
            'skill_id': skill_id_tensor,
            'skill_params': skill_params_tensor
        }
    
    def _instruction_to_id(self, instruction: str, skill_id: str) -> int:
        """将指令转换为ID - 简化版本"""
        # 基于技能ID和指令内容生成简单的ID
        skill_map = {
            'walk_forward': 0,
            'walk_back': 1,
            'turn_left': 2,
            'turn_right': 3,
            'wave': 4,
            'bow': 5,
            'stand': 6,
            'custom': 7
        }
        
        base_id = skill_map.get(skill_id.lower(), 0)
        
        # 根据指令内容微调
        if '快' in instruction or '快速' in instruction:
            base_id += 8
        elif '慢' in instruction or '慢慢' in instruction:
            base_id += 16
        
        return min(base_id, 31)  # 限制在合理范围内
    
    def get_vocab(self) -> Dict[str, int]:
        """获取词汇表"""
        return self.vocab.copy()
    
    def get_skill_distribution(self) -> Dict[int, int]:
        """获取技能分布"""
        return self.skill_counts.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'num_trajectories': len(self.trajectories),
            'vocab_size': self.vocab_size,
            'skill_distribution': self.skill_counts,
            'split': self.split,
            'augment': self.augment,
            'max_seq_len': self.max_seq_len
        }


def create_data_loaders(data_dir: str, batch_size: int = 8, 
                       num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        Dict[str, DataLoader]: 包含train/val/test数据加载器的字典
    """
    # 首先加载所有轨迹数据
    temp_dataset = TrajectoryDataset(data_dir, split='all', augment=False)
    all_trajectories = temp_dataset.trajectories
    vocab = temp_dataset.get_vocab()
    
    # 按技能类别进行分层分割
    train_trajectories = []
    val_trajectories = []
    test_trajectories = []
    
    # 按技能分组
    skill_groups = {}
    for trajectory in all_trajectories:
        skill_name = trajectory.get('metadata', {}).get('skill_name', 'unknown')
        if skill_name not in skill_groups:
            skill_groups[skill_name] = []
        skill_groups[skill_name].append(trajectory)
    
    # 对每个技能组进行分割
    train_split = config.DATA['train_split']
    val_split = config.DATA['val_split']
    test_split = config.DATA['test_split']
    
    for skill_name, trajectories in skill_groups.items():
        n_total = len(trajectories)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val
        
        # 随机打乱
        random.shuffle(trajectories)
        
        # 分割数据
        train_trajectories.extend(trajectories[:n_train])
        val_trajectories.extend(trajectories[n_train:n_train + n_val])
        test_trajectories.extend(trajectories[n_train + n_val:])
    
    print(f"Data split summary:")
    print(f"  Train: {len(train_trajectories)} samples")
    print(f"  Val: {len(val_trajectories)} samples")
    print(f"  Test: {len(test_trajectories)} samples")
    
    # 创建数据集
    train_dataset = TrajectoryDataset(data_dir, split='train', augment=True, 
                                     trajectories=train_trajectories, vocab=vocab)
    val_dataset = TrajectoryDataset(data_dir, split='val', augment=False,
                                   trajectories=val_trajectories, vocab=vocab)
    test_dataset = TrajectoryDataset(data_dir, split='test', augment=False,
                                    trajectories=test_trajectories, vocab=vocab)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'vocab': vocab
    }


def create_demo_data(data_dir: str, num_trajectories: int = 100):
    """
    创建演示数据
    
    Args:
        data_dir: 数据目录路径
        num_trajectories: 要创建的轨迹数量
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 技能和指令模板
    skill_templates = {
        'walk': ['向前走', '向后走', '快速前进', '慢慢走'],
        'turn': ['向左转', '向右转', '转个弯', '改变方向'],
        'wave': ['挥手', '打招呼', '摆摆手', '说再见'],
        'bow': ['鞠躬', '行礼', '点头致意', '弯腰'],
        'stand': ['站立', '站直', '保持站立', '站稳'],
        'custom': ['跳起来', '蹲下', '坐下', '站起来']
    }
    
    # 生成轨迹数据
    for i in range(num_trajectories):
        skill_name = random.choice(list(skill_templates.keys()))
        instruction = random.choice(skill_templates[skill_name])
        
        # 创建轨迹
        trajectory = {
            'instruction': instruction,
            'timestamp': 1234567890.0 + i,
            'frequency': 50.0,
            'observations': [],
            'actions': [],
            'metadata': {
                'skill_name': skill_name,
                'num_joints': 45,
                'image_size': [224, 224],
                'total_steps': random.randint(20, 100),
                'duration': random.uniform(2.0, 8.0),
                'stride_length': random.uniform(0.2, 0.5) if skill_name == 'walk' else 0.3,
                'walk_speed': random.uniform(0.5, 2.0) if skill_name == 'walk' else 1.0,
                'turn_angle': random.uniform(15.0, 90.0) if skill_name == 'turn' else 45.0,
                'turn_speed': random.uniform(0.5, 2.0) if skill_name == 'turn' else 1.0,
                'direction': random.choice(['forward', 'backward']) if skill_name == 'walk' else random.choice(['left', 'right']) if skill_name == 'turn' else 'forward'
            }
        }
        
        # 生成观测数据
        num_steps = trajectory['metadata']['total_steps']
        for step in range(num_steps):
            # 生成模拟图像数据
            image_data = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            
            # 生成模拟状态数据
            joint_angles = np.random.uniform(-np.pi, np.pi, 45)
            joint_velocities = np.random.uniform(-1, 1, 45)
            imu = np.random.uniform(-2, 2, 9)
            body_pos = np.random.uniform(-0.1, 0.1, 3)
            body_quat = np.random.randn(4)
            body_quat = body_quat / np.linalg.norm(body_quat)
            
            obs = {
                'image': image_data.tolist(),
                'image_shape': [224, 224, 3],
                'image_dtype': 'uint8',
                'state': {
                    'joint_angles': joint_angles.tolist(),
                    'joint_velocities': joint_velocities.tolist(),
                    'imu': imu.tolist(),
                    'body_pos': body_pos.tolist(),
                    'body_quat': body_quat.tolist()
                }
            }
            
            trajectory['observations'].append(obs)
            
            # 生成动作数据
            action = np.random.uniform(-1, 1, 45)
            trajectory['actions'].append(action.tolist())
        
        # 保存轨迹
        file_path = data_dir / f"trajectory_{i:04d}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
    
    print(f"Created {num_trajectories} demo trajectories in {data_dir}")


if __name__ == "__main__":
    # 测试数据集
    print("Testing TrajectoryDataset...")
    
    # 创建演示数据
    demo_data_dir = "./demo_trajectories"
    create_demo_data(demo_data_dir, num_trajectories=10)
    
    # 创建数据集
    dataset = TrajectoryDataset(demo_data_dir, split='train', augment=True)
    
    # 测试数据加载
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Instruction tokens shape: {sample['instruction_tokens'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"State vector shape: {sample['state_vector'].shape}")
    print(f"Skill ID: {sample['skill_id']}")
    print(f"Skill params shape: {sample['skill_params'].shape}")
    
    # 创建数据加载器
    data_loaders = create_data_loaders(demo_data_dir, batch_size=4, num_workers=2)
    
    # 测试数据加载器
    for batch in data_loaders['train']:
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Batch instruction tokens: {batch['instruction_tokens'].shape}")
        print(f"Batch images: {batch['image'].shape}")
        print(f"Batch state vectors: {batch['state_vector'].shape}")
        print(f"Batch skill IDs: {batch['skill_id'].shape}")
        print(f"Batch skill params: {batch['skill_params'].shape}")
        break
    
    print("TrajectoryDataset test completed successfully!")