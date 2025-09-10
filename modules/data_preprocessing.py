#!/usr/bin/env python3
"""
高频数据动作序列预处理模块
实现动态下采样和chunking策略
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt


class ActionChunkPreprocessor:
    """动作块预处理器"""
    
    def __init__(self, 
                 chunk_length: int = 16,
                 min_action_variance: float = 0.001,
                 max_downsample_factor: int = 8,
                 overlap_ratio: float = 0.5,  # 新增重叠比例
                 normalize_data: bool = True):  # 新增数据标准化
        """
        初始化预处理器
        
        Args:
            chunk_length: 动作块长度
            min_action_variance: 最小动作方差，用于判断是否需要下采样
            max_downsample_factor: 最大下采样因子
            overlap_ratio: 重叠比例
            normalize_data: 是否标准化数据
        """
        self.chunk_length = chunk_length
        self.min_action_variance = min_action_variance
        self.max_downsample_factor = max_downsample_factor
        self.overlap_ratio = overlap_ratio
        self.normalize_data = normalize_data
        
        # 数据标准化参数
        self.data_min = -2.9459
        self.data_max = 1.4640
        self.data_mean = (self.data_min + self.data_max) / 2.0
        self.data_std = (self.data_max - self.data_min) / 4.0  # 覆盖99.7%的数据
    
    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """标准化动作数据到[-1, 1]范围"""
        if not self.normalize_data:
            return actions
        
        # 标准化到[-1, 1]
        normalized = (actions - self.data_mean) / self.data_std
        # 确保在[-1, 1]范围内
        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized
    
    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        """反标准化动作数据"""
        if not self.normalize_data:
            return normalized_actions
        
        # 从[-1, 1]反标准化
        actions = normalized_actions * self.data_std + self.data_mean
        return actions
    
    def analyze_task_data(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析任务数据特征
        
        Args:
            trajectory_data: 轨迹数据
            
        Returns:
            数据分析结果
        """
        actions = np.array(trajectory_data.get('actions', []))
        
        if len(actions) == 0:
            return {'error': 'No actions found'}
        
        # 计算基本统计信息
        total_frames = len(actions)
        action_dim = actions.shape[1] if len(actions.shape) > 1 else 1
        
        # 计算动作变化统计
        if total_frames > 1:
            action_diffs = np.diff(actions, axis=0)
            action_variances = np.var(action_diffs, axis=0)
            mean_variance = np.mean(action_variances)
            max_variance = np.max(action_variances)
            min_variance = np.min(action_variances)
        else:
            mean_variance = 0.0
            max_variance = 0.0
            min_variance = 0.0
        
        # 计算每个关节的平均变化量
        joint_changes = np.mean(np.abs(action_diffs), axis=0) if total_frames > 1 else np.zeros(action_dim)
        
        return {
            'total_frames': total_frames,
            'action_dim': action_dim,
            'mean_action_variance': mean_variance,
            'max_action_variance': max_variance,
            'min_action_variance': min_variance,
            'mean_joint_changes': joint_changes.tolist(),
            'total_action_range': float(actions.max() - actions.min()),
            'action_std': float(np.std(actions)),
            'action_mean': float(np.mean(actions))
        }
    
    def determine_downsample_factor(self, analysis: Dict[str, Any]) -> int:
        """
        确定下采样因子
        
        Args:
            analysis: 数据分析结果
            
        Returns:
            下采样因子
        """
        mean_variance = analysis['mean_action_variance']
        
        # 如果动作变化太小，需要下采样
        if mean_variance < self.min_action_variance:
            # 计算需要的下采样因子
            target_variance = self.min_action_variance * 2
            if mean_variance > 0:
                factor = int(np.sqrt(target_variance / mean_variance))
                factor = min(factor, self.max_downsample_factor)
                factor = max(factor, 1)
            else:
                factor = self.max_downsample_factor
        else:
            factor = 1
        
        return factor
    
    def downsample_actions(self, actions: np.ndarray, factor: int) -> np.ndarray:
        """
        下采样动作序列 - 使用固定间隔采样以保持时间一致性
        
        Args:
            actions: 原始动作序列
            factor: 下采样因子
            
        Returns:
            下采样后的动作序列
        """
        if factor <= 1:
            return actions
        
        # 使用固定间隔采样，保持时间一致性
        n_frames = len(actions)
        n_downsampled = n_frames // factor
        
        if n_downsampled == 0:
            return actions[:1]  # 至少保留一帧
        
        # 固定间隔采样，而不是滑动窗口平均
        indices = np.arange(0, n_frames, factor)[:n_downsampled]
        downsampled = actions[indices]
        
        return downsampled
    
    def create_action_chunks(self, actions: np.ndarray) -> List[np.ndarray]:
        """
        创建动作块
        
        Args:
            actions: 动作序列
            
        Returns:
            动作块列表
        """
        n_frames = len(actions)
        chunks = []
        
        # 在动作块创建部分修改步长：
        step_size = max(1, int(self.chunk_length * (1 - self.overlap_ratio)))  # 50%重叠
        
        for start_idx in range(0, n_frames - self.chunk_length + 1, step_size):
            end_idx = start_idx + self.chunk_length
            chunk = actions[start_idx:end_idx]
            chunks.append(chunk)
        
        # 处理剩余的帧
        remaining_frames = n_frames - (len(chunks) * step_size)
        if remaining_frames > 0:
            # 使用填充创建最后一个块
            last_chunk = actions[-remaining_frames:]
            if len(last_chunk) < self.chunk_length:
                # 用最后一帧填充到完整长度
                padding = np.tile(last_chunk[-1:], (self.chunk_length - len(last_chunk), 1))
                padded_chunk = np.vstack([last_chunk, padding])
                chunks.append(padded_chunk)
            else:
                chunks.append(last_chunk)
        
        return chunks
    
    def process_trajectory(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个轨迹
        
        Args:
            trajectory_data: 轨迹数据
            
        Returns:
            处理后的轨迹数据
        """
        print("📊 分析轨迹数据...")
        analysis = self.analyze_task_data(trajectory_data)
        
        if 'error' in analysis:
            return {'error': analysis['error']}
        
        print(f"   总帧数: {analysis['total_frames']}")
        print(f"   动作维度: {analysis['action_dim']}")
        print(f"   平均动作方差: {analysis['mean_action_variance']:.6f}")
        print(f"   动作标准差: {analysis['action_std']:.6f}")
        
        # 确定下采样因子
        downsample_factor = self.determine_downsample_factor(analysis)
        print(f"   下采样因子: {downsample_factor}")
        
        # 从joint_pos计算真实的actions（相邻帧差值）
        joint_positions = np.array([obs['joint_pos'] for obs in trajectory_data['observations']])
        # 计算actions：后一帧减去前一帧
        actions = np.diff(joint_positions, axis=0)
        
        if downsample_factor > 1:
            downsampled_actions = self.downsample_actions(actions, downsample_factor)
            print(f"   下采样后帧数: {len(downsampled_actions)}")
        else:
            downsampled_actions = actions
        
        # 标准化动作数据
        if self.normalize_data:
            normalized_actions = self.normalize_actions(downsampled_actions)
            print(f"   数据标准化完成，范围: [{normalized_actions.min():.4f}, {normalized_actions.max():.4f}]")
        else:
            normalized_actions = downsampled_actions
        
        # 创建动作块
        chunks = self.create_action_chunks(normalized_actions)
        print(f"   创建动作块数: {len(chunks)}")
        
        # 处理其他数据（observations等）
        processed_data = {
            'task_name': trajectory_data.get('task_name', 'unknown'),
            'original_analysis': analysis,
            'downsample_factor': downsample_factor,
            'original_frames': analysis['total_frames'],
            'downsampled_frames': len(downsampled_actions),
            'chunk_length': self.chunk_length,
            'num_chunks': len(chunks),
            'action_chunks': [chunk.tolist() for chunk in chunks],
            'chunk_statistics': self._compute_chunk_statistics(chunks),
            'normalized': self.normalize_data,
            'normalization_params': {
                'data_min': self.data_min,
                'data_max': self.data_max,
                'data_mean': self.data_mean,
                'data_std': self.data_std
            }
        }
        
        # 如果有observations，也进行相应的处理
        if 'observations' in trajectory_data:
            processed_observations = self._process_observations(
                trajectory_data['observations'], 
                downsample_factor
            )
            processed_data['processed_observations'] = processed_observations
        
        return processed_data
    
    def _compute_chunk_statistics(self, chunks: List[np.ndarray]) -> Dict[str, Any]:
        """
        计算块统计信息
        
        Args:
            chunks: 动作块列表
            
        Returns:
            统计信息
        """
        all_chunks = np.vstack(chunks)
        
        return {
            'mean_action': float(np.mean(all_chunks)),
            'std_action': float(np.std(all_chunks)),
            'min_action': float(np.min(all_chunks)),
            'max_action': float(np.max(all_chunks)),
            'mean_chunk_variance': float(np.mean([np.var(chunk, axis=0).mean() for chunk in chunks])),
            'max_chunk_variance': float(np.max([np.var(chunk, axis=0).max() for chunk in chunks]))
        }
    
    def _process_observations(self, observations: List[Dict[str, Any]], downsample_factor: int) -> List[Dict[str, Any]]:
        """
        处理观测数据 - 使用固定间隔采样以保持时间一致性
        
        Args:
            observations: 观测数据列表
            downsample_factor: 下采样因子
            
        Returns:
            处理后的观测数据
        """
        if downsample_factor <= 1:
            return observations
        
        # 使用固定间隔采样，保持时间一致性
        processed = []
        for i in range(0, len(observations), downsample_factor):
            processed.append(observations[i])
        
        return processed
    
    def visualize_preprocessing(self, original_data: Dict[str, Any], processed_data: Dict[str, Any]):
        """
        可视化预处理结果
        
        Args:
            original_data: 原始数据
            processed_data: 处理后数据
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始动作序列
        original_actions = np.array(original_data['actions'])
        axes[0, 0].plot(original_actions[:, 0], label='Joint 0')
        axes[0, 0].plot(original_actions[:, 1], label='Joint 1')
        axes[0, 0].plot(original_actions[:, 2], label='Joint 2')
        axes[0, 0].set_title('Original Action Sequence')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Action Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 下采样后的动作序列
        downsampled_actions = np.array(processed_data['action_chunks']).reshape(-1, original_actions.shape[1])
        downsampled_frames = np.arange(len(downsampled_actions)) * processed_data['downsample_factor']
        axes[0, 1].plot(downsampled_frames, downsampled_actions[:, 0], 'o-', label='Joint 0')
        axes[0, 1].plot(downsampled_frames, downsampled_actions[:, 1], 's-', label='Joint 1')
        axes[0, 1].plot(downsampled_frames, downsampled_actions[:, 2], '^-', label='Joint 2')
        axes[0, 1].set_title('Downsampled Action Sequence')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Action Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 动作块可视化
        chunk_actions = np.array(processed_data['action_chunks'])
        chunk_idx = 0
        for i, chunk in enumerate(chunk_actions[:5]):  # 只显示前5个块
            chunk_frames = np.arange(len(chunk)) + i * self.chunk_length
            axes[1, 0].plot(chunk_frames, chunk[:, 0], alpha=0.7, label=f'Chunk {i}' if i < 3 else '')
        axes[1, 0].set_title('Action Chunks (First 5 chunks)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Action Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 统计信息
        analysis = processed_data['original_analysis']
        stats_text = f"""
        Original Frames: {analysis['total_frames']}
        Downsampled Frames: {processed_data['downsampled_frames']}
        Downsample Factor: {processed_data['downsample_factor']}
        Action Chunks: {processed_data['num_chunks']}
        Chunk Length: {processed_data['chunk_length']}
        
        Mean Action Variance: {analysis['mean_action_variance']:.6f}
        Action Std: {analysis['action_std']:.6f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('Preprocessing Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def preprocess_trajectory_file(trajectory_path: str, output_path: str = None, show_visualization: bool = False) -> Dict[str, Any]:
    """
    预处理轨迹文件
    
    Args:
        trajectory_path: 轨迹文件路径
        output_path: 输出文件路径
        show_visualization: 是否显示可视化图表
        
    Returns:
        处理后的数据
    """
    # 加载轨迹数据
    with open(trajectory_path, 'r', encoding='utf-8') as f:
        trajectory_data = json.load(f)
    
    # 创建预处理器
    preprocessor = ActionChunkPreprocessor(
        chunk_length=16,
        min_action_variance=0.001,
        max_downsample_factor=8
    )
    
    # 处理轨迹
    processed_data = preprocessor.process_trajectory(trajectory_data)
    
    # 保存处理后的数据
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"处理后的数据已保存到: {output_path}")
    
    # 可视化结果（仅在需要时显示）
    if show_visualization:
        preprocessor.visualize_preprocessing(trajectory_data, processed_data)
    
    return processed_data


if __name__ == "__main__":
    # 测试预处理模块
    test_trajectory_path = "/root/kuavo_ws/src/vla/trajectories/wave_001.json"
    output_path = "/root/kuavo_ws/src/vla/trajectories/wave_001_processed.json"
    
    processed_data = preprocess_trajectory_file(test_trajectory_path, output_path)
    print("预处理完成！")