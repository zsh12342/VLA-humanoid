#!/usr/bin/env python3
"""
Diffusion Policy 推理逻辑
包含动作序列生成、滑动窗口推理和多任务推理
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import deque

from modules.diffusion_policy import LargeBehaviorModel, create_large_behavior_model
from modules.data_preprocessing import ActionChunkPreprocessor


class DiffusionInferenceEngine:
    """
    Diffusion 推理引擎
    """
    
    def __init__(self, 
                 model: LargeBehaviorModel,
                 device: str = 'cuda',
                 chunk_length: int = 16,
                 sliding_window_overlap: int = 4):
        """
        初始化推理引擎
        
        Args:
            model: 训练好的模型
            device: 设备
            chunk_length: 动作块长度
            sliding_window_overlap: 滑动窗口重叠长度
        """
        self.model = model.to(device)
        self.device = device
        self.chunk_length = chunk_length
        self.sliding_window_overlap = sliding_window_overlap
        self.model.eval()
        
        # 状态缓存
        self.state_history = deque(maxlen=10)
        self.action_history = deque(maxlen=50)
        
        # 指令映射
        self.instruction_map = {'wave': 0, 'welcome': 1}
        self.instruction_to_name = {v: k for k, v in self.instruction_map.items()}
    
    def generate_action_chunk(self, 
                            instruction: str,
                            state: np.ndarray,
                            visual_features: Optional[np.ndarray] = None,
                            num_sampling_steps: int = 100,
                            guidance_scale: float = 1.5,
                            temperature: float = 1.0) -> np.ndarray:
        """
        生成动作块
        
        Args:
            instruction: 指令字符串
            state: 当前状态 [state_dim]
            visual_features: 视觉特征 [visual_feature_dim]
            num_sampling_steps: 采样步数
            guidance_scale: 引导强度
            temperature: 温度参数
            
        Returns:
            生成的动作块 [chunk_length, action_dim]
        """
        # 转换为tensor
        instruction_id = torch.tensor([self.instruction_map[instruction]], dtype=torch.long).to(self.device)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 处理视觉特征
        visual_tensor = None
        if visual_features is not None:
            visual_tensor = torch.tensor(visual_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 生成动作序列
        with torch.no_grad():
            action_chunk = self.model.generate_action_sequence(
                instruction_ids=instruction_id,
                state=state_tensor,
                visual_features=visual_tensor,
                num_steps=num_sampling_steps,
                guidance_scale=guidance_scale
            )
        
        # 应用温度
        if temperature != 1.0:
            action_chunk = action_chunk * temperature
        
        # 转换为numpy
        return action_chunk.squeeze(0).cpu().numpy()
    
    def sliding_window_inference(self, 
                               instruction: str,
                               initial_state: np.ndarray,
                               total_steps: int,
                               visual_features_sequence: Optional[List[np.ndarray]] = None,
                               num_sampling_steps: int = 100,
                               guidance_scale: float = 1.5,
                               temperature: float = 1.0) -> List[np.ndarray]:
        """
        滑动窗口推理
        
        Args:
            instruction: 指令字符串
            initial_state: 初始状态 [state_dim]
            total_steps: 总步数
            visual_features_sequence: 视觉特征序列
            num_sampling_steps: 采样步数
            guidance_scale: 引导强度
            temperature: 温度参数
            
        Returns:
            完整动作序列
        """
        all_actions = []
        current_state = initial_state.copy()
        
        # 计算需要生成的块数
        num_chunks = (total_steps + self.chunk_length - 1) // self.chunk_length
        
        for chunk_idx in range(num_chunks):
            print(f"生成第 {chunk_idx + 1}/{num_chunks} 个动作块...")
            
            # 获取当前视觉特征
            current_visual = None
            if visual_features_sequence and chunk_idx < len(visual_features_sequence):
                current_visual = visual_features_sequence[chunk_idx]
            
            # 生成当前块
            action_chunk = self.generate_action_chunk(
                instruction=instruction,
                state=current_state,
                visual_features=current_visual,
                num_sampling_steps=num_sampling_steps,
                guidance_scale=guidance_scale,
                temperature=temperature
            )
            
            # 添加到动作序列
            all_actions.append(action_chunk)
            
            # 更新状态（简化版本，使用最后一个动作作为新状态）
            last_action = action_chunk[-1]
            current_state = self._update_state(current_state, last_action)
            
            # 更新历史
            self.state_history.append(current_state.copy())
            self.action_history.extend(action_chunk)
        
        # 处理最后一步，确保总步数正确
        if len(all_actions) * self.chunk_length > total_steps:
            # 截断多余的步骤
            final_actions = []
            remaining_steps = total_steps
            
            for chunk in all_actions:
                if remaining_steps <= 0:
                    break
                take_steps = min(len(chunk), remaining_steps)
                final_actions.append(chunk[:take_steps])
                remaining_steps -= take_steps
            
            all_actions = final_actions
        
        return all_actions
    
    def _update_state(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        更新状态（简化版本）
        
        Args:
            current_state: 当前状态
            action: 执行的动作
            
        Returns:
            更新后的状态
        """
        new_state = current_state.copy()
        
        # 简单的状态更新：将动作添加到位置
        if len(action) >= len(new_state):
            new_state = action[:len(new_state)]
        else:
            new_state[:len(action)] = action
        
        # 添加一些噪声使其更真实
        noise = np.random.normal(0, 0.01, len(new_state))
        new_state += noise
        
        return new_state
    
    def multi_task_inference(self, 
                           task_sequence: List[Tuple[str, int]],
                           initial_state: np.ndarray,
                           visual_features_dict: Optional[Dict[str, List[np.ndarray]]] = None,
                           num_sampling_steps: int = 50,
                           guidance_scale: float = 1.0) -> Dict[str, List[np.ndarray]]:
        """
        多任务推理
        
        Args:
            task_sequence: 任务序列 [(instruction, num_steps), ...]
            initial_state: 初始状态
            visual_features_dict: 视觉特征字典
            num_sampling_steps: 采样步数
            guidance_scale: 引导强度
            
        Returns:
            每个任务的动作序列
        """
        task_results = {}
        current_state = initial_state.copy()
        
        for instruction, num_steps in task_sequence:
            print(f"\n🎯 执行任务: {instruction} ({num_steps} 步)")
            
            # 获取视觉特征
            visual_sequence = None
            if visual_features_dict and instruction in visual_features_dict:
                visual_sequence = visual_features_dict[instruction]
            
            # 生成动作序列
            task_actions = self.sliding_window_inference(
                instruction=instruction,
                initial_state=current_state,
                total_steps=num_steps,
                visual_features_sequence=visual_sequence,
                num_sampling_steps=num_sampling_steps,
                guidance_scale=guidance_scale
            )
            
            task_results[instruction] = task_actions
            
            # 更新状态为最后一个动作
            if task_actions:
                last_action = task_actions[-1][-1]
                current_state = self._update_state(current_state, last_action)
        
        return task_results
    
    def evaluate_action_quality(self, 
                               generated_actions: List[np.ndarray],
                               reference_actions: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """
        评估动作质量
        
        Args:
            generated_actions: 生成的动作序列
            reference_actions: 参考动作序列（可选）
            
        Returns:
            质量指标
        """
        # 展平动作序列
        flat_actions = np.vstack(generated_actions)
        
        # 基本统计
        metrics = {
            'mean_action': float(np.mean(flat_actions)),
            'std_action': float(np.std(flat_actions)),
            'min_action': float(np.min(flat_actions)),
            'max_action': float(np.max(flat_actions)),
            'total_steps': len(flat_actions)
        }
        
        # 动作变化分析
        if len(flat_actions) > 1:
            action_diffs = np.diff(flat_actions, axis=0)
            metrics['mean_change'] = float(np.mean(np.abs(action_diffs)))
            metrics['max_change'] = float(np.max(np.abs(action_diffs)))
            metrics['action_variance'] = float(np.var(action_diffs))
        
        # 振荡分析
        if len(flat_actions) > 2:
            velocity = np.diff(flat_actions, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # 计算符号变化（振荡指标）
            sign_changes = 0
            for i in range(len(velocity) - 1):
                sign_changes += np.sum(np.sign(velocity[i]) != np.sign(velocity[i+1]))
            
            total_comparisons = len(velocity) - 1
            metrics['oscillation_rate'] = sign_changes / (total_comparisons * flat_actions.shape[1]) if total_comparisons > 0 else 0
            metrics['mean_acceleration'] = float(np.mean(np.abs(acceleration)))
        
        # 如果有参考动作，计算相似度
        if reference_actions is not None:
            flat_reference = np.vstack(reference_actions)
            
            # 确保长度一致
            min_len = min(len(flat_actions), len(flat_reference))
            flat_actions_aligned = flat_actions[:min_len]
            flat_reference_aligned = flat_reference[:min_len]
            
            # 计算相似度
            similarity = self._calculate_similarity(flat_actions_aligned, flat_reference_aligned)
            metrics['similarity_to_reference'] = similarity
            
            # 计算MSE
            mse = np.mean((flat_actions_aligned - flat_reference_aligned) ** 2)
            metrics['mse_to_reference'] = float(mse)
        
        return metrics
    
    def _calculate_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        计算序列相似度
        """
        if seq1.shape != seq2.shape:
            return 0.0
        
        seq1_flat = seq1.flatten()
        seq2_flat = seq2.flatten()
        
        dot_product = np.dot(seq1_flat, seq2_flat)
        norm1 = np.linalg.norm(seq1_flat)
        norm2 = np.linalg.norm(seq2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def visualize_generation(self, 
                           generated_actions: List[np.ndarray],
                           reference_actions: Optional[List[np.ndarray]] = None,
                           save_path: Optional[str] = None):
        """
        可视化生成的动作序列
        
        Args:
            generated_actions: 生成的动作序列
            reference_actions: 参考动作序列
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 展平动作序列
        flat_generated = np.vstack(generated_actions)
        
        # 绘制生成的动作序列
        time_steps = np.arange(len(flat_generated))
        
        # 选择前6个关节进行绘制
        important_joints = [0, 1, 2, 3, 4, 5]
        
        for i, joint_idx in enumerate(important_joints[:3]):
            axes[0, 0].plot(time_steps, flat_generated[:, joint_idx], 
                           label=f'Joint {joint_idx}', linewidth=2)
        
        axes[0, 0].set_title('Generated Action Sequence')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Action Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 绘制动作变化
        if len(flat_generated) > 1:
            action_diffs = np.diff(flat_generated, axis=0)
            diff_steps = np.arange(len(action_diffs))
            
            for i, joint_idx in enumerate(important_joints[:3]):
                axes[0, 1].plot(diff_steps, action_diffs[:, joint_idx], 
                               label=f'Joint {joint_idx}', linewidth=2)
        
        axes[0, 1].set_title('Action Changes')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Action Change')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 绘制关节轨迹
        for i, joint_idx in enumerate(important_joints[:3]):
            axes[1, 0].plot(flat_generated[:, joint_idx], flat_generated[:, joint_idx+1], 
                           'o-', markersize=3, label=f'Joint {joint_idx} vs {joint_idx+1}')
        
        axes[1, 0].set_title('Joint Trajectories')
        axes[1, 0].set_xlabel(f'Joint {important_joints[0]}')
        axes[1, 0].set_ylabel(f'Joint {important_joints[1]}')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 绘制统计信息
        metrics = self.evaluate_action_quality(generated_actions, reference_actions)
        
        stats_text = f"""
        Generation Statistics:
        Total Steps: {metrics['total_steps']}
        Mean Action: {metrics['mean_action']:.4f}
        Std Action: {metrics['std_action']:.4f}
        Mean Change: {metrics['mean_change']:.4f}
        Oscillation Rate: {metrics['oscillation_rate']:.3f}
        
        {'Similarity: {:.4f}'.format(metrics.get('similarity_to_reference', 0)) if 'similarity_to_reference' in metrics else ''}
        {'MSE: {:.6f}'.format(metrics.get('mse_to_reference', 0)) if 'mse_to_reference' in metrics else ''}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('Generation Statistics')
        axes[1, 1].axis('off')
        
        # 如果有参考动作，绘制对比
        if reference_actions is not None:
            flat_reference = np.vstack(reference_actions)
            min_len = min(len(flat_generated), len(flat_reference))
            
            # 创建新的对比图
            fig2, axes2 = plt.subplots(1, 1, figsize=(12, 6))
            
            time_steps_aligned = np.arange(min_len)
            
            for i, joint_idx in enumerate(important_joints[:3]):
                axes2.plot(time_steps_aligned, flat_generated[:min_len, joint_idx], 
                          label=f'Generated Joint {joint_idx}', linewidth=2)
                axes2.plot(time_steps_aligned, flat_reference[:min_len, joint_idx], 
                          label=f'Reference Joint {joint_idx}', linewidth=2, linestyle='--')
            
            axes2.set_title('Generated vs Reference Actions')
            axes2.set_xlabel('Time Step')
            axes2.set_ylabel('Action Value')
            axes2.legend()
            axes2.grid(True)
            
            if save_path:
                comparison_path = save_path.replace('.png', '_comparison.png')
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                print(f"对比图已保存到: {comparison_path}")
            
            # 训练期间不显示图表
            if not save_path:
                plt.show()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"生成图已保存到: {save_path}")
        
        # 训练期间不显示图表
        if not save_path:
            plt.show()
    
    def benchmark_inference(self, 
                          instruction: str,
                          state: np.ndarray,
                          num_runs: int = 10) -> Dict[str, float]:
        """
        基准测试推理性能
        
        Args:
            instruction: 指令字符串
            state: 状态
            num_runs: 运行次数
            
        Returns:
            性能指标
        """
        print(f"🚀 开始基准测试 ({num_runs} 次)...")
        
        times = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            # 生成动作块
            action_chunk = self.generate_action_chunk(
                instruction=instruction,
                state=state,
                num_sampling_steps=100,
                guidance_scale=1.5
            )
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        # 计算统计信息
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # 计算FPS
        fps = self.chunk_length / mean_time
        
        return {
            'mean_time': mean_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'fps': fps,
            'total_params': sum(p.numel() for p in self.model.parameters())
        }


def create_inference_engine(model_path: str, device: str = 'cuda') -> DiffusionInferenceEngine:
    """
    创建推理引擎
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        推理引擎实例
    """
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型
    model = create_large_behavior_model()
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 创建推理引擎
    engine = DiffusionInferenceEngine(
        model=model,
        device=device,
        chunk_length=16,
        sliding_window_overlap=4
    )
    
    return engine


def test_inference_engine():
    """测试推理引擎"""
    print("测试推理引擎...")
    
    # 创建虚拟模型
    model = create_large_behavior_model({
        'action_dim': 26,
        'chunk_length': 16,
        'state_dim': 60,
        'diffusion_dim': 256,
        'num_diffusion_steps': 100
    })
    
    # 创建推理引擎
    engine = DiffusionInferenceEngine(model, device='cpu')
    
    # 测试数据
    instruction = 'wave'
    state = np.random.randn(60)
    
    # 测试动作生成
    action_chunk = engine.generate_action_chunk(instruction, state, num_sampling_steps=100)
    print(f"生成的动作块形状: {action_chunk.shape}")
    
    # 测试滑动窗口推理
    actions = engine.sliding_window_inference(instruction, state, total_steps=32)
    print(f"生成的动作序列块数: {len(actions)}")
    
    # 测试质量评估
    metrics = engine.evaluate_action_quality(actions)
    print(f"动作质量指标: {metrics}")
    
    print("推理引擎测试完成！")


if __name__ == "__main__":
    test_inference_engine()