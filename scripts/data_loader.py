#!/usr/bin/env python3
"""
ROS1 采集数据加载器

将采集的JSON数据转换为适合训练的格式
支持System-2 → System-1网络训练
"""

import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.model_selection import train_test_split


class TrajectoryDataset(Dataset):
    """轨迹数据集类"""
    
    def __init__(self, 
                 data_dir: str, 
                 skill_id: str = None,
                 transform=None,
                 max_frames: int = 35):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录
            skill_id: 特定技能ID，如果为None则加载所有技能
            transform: 数据变换
            max_frames: 最大帧数
        """
        self.data_dir = Path(data_dir)
        self.skill_id = skill_id
        self.transform = transform
        self.max_frames = max_frames
        
        # 加载数据
        self.trajectories = self._load_trajectories()
        
        # 构建词汇表
        self.skill_vocab = self._build_skill_vocab()
        self.instruction_vocab = self._build_instruction_vocab()
        
        print(f"加载了 {len(self.trajectories)} 条轨迹")
        print(f"技能词汇表: {self.skill_vocab}")
        print(f"指令词汇表: {len(self.instruction_vocab)} 条指令")
    
    def _load_trajectories(self) -> List[Dict[str, Any]]:
        """加载轨迹数据"""
        trajectories = []
        
        # 遍历数据目录
        for skill_dir in self.data_dir.iterdir():
            if not skill_dir.is_dir():
                continue
                
            # 如果指定了技能ID，跳过其他技能
            if self.skill_id and skill_dir.name != self.skill_id:
                continue
            
            # 加载该技能的所有轨迹文件
            for json_file in skill_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        trajectory_data = json.load(f)
                    
                    # 处理轨迹数据
                    processed_trajectory = self._process_trajectory(trajectory_data)
                    trajectories.append(processed_trajectory)
                    
                except Exception as e:
                    print(f"加载轨迹文件 {json_file} 失败: {e}")
        
        return trajectories
    
    def _process_trajectory(self, trajectory_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理单条轨迹数据"""
        # 提取基本信息
        skill_id = trajectory_data[0]['skill_id']
        instruction = trajectory_data[0]['instruction']
        params = trajectory_data[0]['params']
        
        # 提取观测和动作序列
        observations = []
        for frame in trajectory_data:
            obs = frame['observation']
            observations.append(obs)
        
        # 如果轨迹过长，截断；如果过短，填充
        if len(observations) > self.max_frames:
            observations = observations[:self.max_frames]
        elif len(observations) < self.max_frames:
            # 用最后一帧填充
            last_obs = observations[-1]
            while len(observations) < self.max_frames:
                observations.append(last_obs.copy())
        
        return {
            'skill_id': skill_id,
            'instruction': instruction,
            'params': params,
            'observations': observations,
            'length': len(trajectory_data),
            'original_length': len(trajectory_data)
        }
    
    def _build_skill_vocab(self) -> Dict[str, int]:
        """构建技能词汇表"""
        skills = set()
        for traj in self.trajectories:
            skills.add(traj['skill_id'])
        
        return {skill: idx for idx, skill in enumerate(sorted(skills))}
    
    def _build_instruction_vocab(self) -> Dict[str, int]:
        """构建指令词汇表"""
        instructions = set()
        for traj in self.trajectories:
            instructions.add(traj['instruction'])
        
        return {instr: idx for idx, instr in enumerate(sorted(instructions))}
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取数据样本"""
        trajectory = self.trajectories[idx]
        
        # 编码技能ID
        skill_id = self.skill_vocab[trajectory['skill_id']]
        
        # 编码指令
        instruction = self._encode_instruction(trajectory['instruction'])
        
        # 处理观测数据
        observations = trajectory['observations']
        joint_positions = []
        body_positions = []
        body_quaternions = []
        images = []
        
        for obs in observations:
            # 关节位置
            joint_pos = np.array(obs['joint_pos'], dtype=np.float32)
            joint_positions.append(joint_pos)
            
            # 身体位置
            body_pos = np.array(obs['body_pos'], dtype=np.float32)
            body_positions.append(body_pos)
            
            # 身体四元数
            body_quat = np.array(obs['body_quat'], dtype=np.float32)
            body_quaternions.append(body_quat)
            
            # 图像（如果有）
            if 'rgb_image' in obs and obs['rgb_image']:
                image = self._load_image(obs['rgb_image'])
                images.append(image)
            else:
                # 创建空白图像
                images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # 转换为张量
        joint_positions = torch.FloatTensor(np.array(joint_positions))
        body_positions = torch.FloatTensor(np.array(body_positions))
        body_quaternions = torch.FloatTensor(np.array(body_quaternions))
        images = torch.FloatTensor(np.array(images))
        
        # 重新排列图像维度 (T, H, W, C) -> (T, C, H, W)
        images = images.permute(0, 3, 1, 2)
        
        return {
            'skill_id': torch.LongTensor([skill_id]),
            'instruction': torch.LongTensor(instruction),
            'joint_positions': joint_positions,
            'body_positions': body_positions,
            'body_quaternions': body_quaternions,
            'images': images,
            'length': torch.LongTensor([trajectory['length']]),
            'params': trajectory['params']
        }
    
    def _encode_instruction(self, instruction: str, max_length: int = 20) -> List[int]:
        """编码指令为索引序列"""
        # 简单的字符级编码
        chars = list(instruction)
        encoded = []
        
        for char in chars[:max_length]:
            encoded.append(ord(char) % 1000)  # 限制在0-999范围内
        
        # 填充到固定长度
        while len(encoded) < max_length:
            encoded.append(0)  # 填充0
        
        return encoded
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            else:
                return np.zeros((224, 224, 3), dtype=np.uint8)
        except Exception as e:
            print(f"加载图像 {image_path} 失败: {e}")
            return np.zeros((224, 224, 3), dtype=np.uint8)


class DataManager:
    """数据管理器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
        # 数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def prepare_data(self, 
                    skill_id: str = None,
                    batch_size: int = 32,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    random_seed: int = 42):
        """准备数据"""
        # 加载完整数据集
        full_dataset = TrajectoryDataset(
            data_dir=self.data_dir,
            skill_id=skill_id
        )
        
        # 分割数据集
        indices = list(range(len(full_dataset)))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=random_seed
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_ratio/(train_ratio+val_ratio), random_state=random_seed
        )
        
        # 创建子数据集
        self.train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4
        )
        
        print(f"数据集分割完成:")
        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        print(f"  测试集: {len(self.test_dataset)} 样本")
        
        return {
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'test_loader': self.test_loader,
            'skill_vocab': full_dataset.skill_vocab,
            'instruction_vocab': full_dataset.instruction_vocab
        }
    
    def get_data_stats(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {
            'data_dir': str(self.data_dir),
            'train_size': len(self.train_dataset) if self.train_dataset else 0,
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'skills': [],
            'instructions': []
        }
        
        # 统计技能和指令
        if self.train_dataset:
            full_dataset = self.train_dataset.dataset
            stats['skills'] = list(full_dataset.skill_vocab.keys())
            stats['instructions'] = list(full_dataset.instruction_vocab.keys())
        
        return stats


def demo_data_loading():
    """演示数据加载"""
    print("数据加载演示")
    print("=" * 50)
    
    # 创建数据管理器
    data_dir = "/tmp/vla_trajectories"
    data_manager = DataManager(data_dir)
    
    # 准备数据
    try:
        data_info = data_manager.prepare_data(
            skill_id=None,  # 加载所有技能
            batch_size=16,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # 显示统计信息
        stats = data_manager.get_data_stats()
        print("\n数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 演示数据加载
        print("\n演示数据加载:")
        for batch_idx, batch in enumerate(data_info['train_loader']):
            print(f"Batch {batch_idx}:")
            print(f"  Skill IDs: {batch['skill_id'].shape}")
            print(f"  Instructions: {batch['instruction'].shape}")
            print(f"  Joint positions: {batch['joint_positions'].shape}")
            print(f"  Body positions: {batch['body_positions'].shape}")
            print(f"  Body quaternions: {batch['body_quaternions'].shape}")
            print(f"  Images: {batch['images'].shape}")
            print(f"  Lengths: {batch['length'].shape}")
            
            if batch_idx >= 2:  # 只显示前3个batch
                break
        
        print("\n数据加载演示完成!")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保已经采集了数据并保存在指定目录")


if __name__ == '__main__':
    demo_data_loading()