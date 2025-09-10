#!/usr/bin/env python3
"""
修复的轨迹生成器 - 解决起始位置问题
核心修复：
1. 正确的起始状态处理
2. 渐进式轨迹变化
3. 真正的从任意位置开始
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
import pickle
from tqdm import tqdm
import math

class FixedTrajectoryGenerator(nn.Module):
    """修复的轨迹生成器"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # 基础参数
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.instruction_vocab_size = config['instruction_vocab_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.trajectory_length = config['trajectory_length']
        self.dropout = config['dropout']
        
        # 增强的指令编码器 - 确保指令差异
        self.instruction_embedding = nn.Embedding(self.instruction_vocab_size, self.hidden_dim * 4)
        self.instruction_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 8, self.hidden_dim * 8),
        )
        
        # 指令差异损失
        self.instruction_diff_weight = 0.05
        
        # 起始状态编码器
        self.start_state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        )
        
        # 进度编码器 - 简单的时间编码
        self.progress_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
        )
        
        # 轨迹解码器
        self.trajectory_decoder = nn.LSTM(
            input_size=self.hidden_dim * 8 + self.hidden_dim * 2 + self.hidden_dim + self.action_dim,
            hidden_size=self.hidden_dim * 4,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        
        # 时序增强层
        self.temporal_enhancer = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        # 动作预测头
        self.action_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.action_dim)
        )
        
        # 指令特定的动作模式（可学习） - 增强初始化
        self.instruction_patterns = nn.Parameter(
            torch.randn(self.instruction_vocab_size, self.trajectory_length, self.action_dim) * 0.5
        )
        
                
    def forward(self, start_state, instruction_id, target_actions=None):
        """
        start_state: [batch_size, state_dim] - 起始状态
        instruction_id: [batch_size] - 指令ID
        target_actions: [batch_size, trajectory_length, action_dim] (用于训练)
        """
        batch_size = start_state.size(0)
        
        # 1. 编码指令，生成轨迹特征
        instruction_embedded = self.instruction_embedding(instruction_id)  # [batch_size, hidden_dim * 4]
        instruction_features = self.instruction_encoder(instruction_embedded)  # [batch_size, hidden_dim * 8]
        instruction_features = instruction_features.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        
        # 2. 编码起始状态
        start_state_features = self.start_state_encoder(start_state)  # [batch_size, hidden_dim * 2]
        start_state_features = start_state_features.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        
        # 3. 生成进度编码
        progress = torch.linspace(0, 1, self.trajectory_length, device=start_state.device)
        progress = progress.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [batch_size, trajectory_length, 1]
        progress_features = self.progress_encoder(progress)  # [batch_size, trajectory_length, hidden_dim]
        
        # 4. 自回归生成轨迹
        if self.training:
            # 训练时使用teacher forcing
            actions = []
            hidden = None
            
            for t in range(self.trajectory_length):
                # 合并所有特征
                if t == 0:
                    # 第一步使用零动作
                    prev_action = torch.zeros(batch_size, 1, self.action_dim, device=start_state.device)
                else:
                    if target_actions is not None:
                        prev_action = target_actions[:, t-1:t, :]
                    else:
                        prev_action = actions[-1]  # 推理时使用上一步预测的动作
                
                combined_features = torch.cat([
                    instruction_features[:, t:t+1, :],
                    start_state_features[:, t:t+1, :],
                    progress_features[:, t:t+1, :],
                    prev_action
                ], dim=-1)
                
                # LSTM解码
                output, hidden = self.trajectory_decoder(combined_features, hidden)
                
                # 时序增强
                enhanced_output = self.temporal_enhancer(output)
                
                # 预测动作
                action = self.action_predictor(enhanced_output)
                actions.append(action)
            
            predicted_actions = torch.cat(actions, dim=1)  # [batch_size, trajectory_length, action_dim]
            
        else:
            # 推理时自回归生成
            actions = []
            hidden = None
            
            for t in range(self.trajectory_length):
                # 合并所有特征
                if t == 0:
                    prev_action = torch.zeros(batch_size, 1, self.action_dim, device=start_state.device)
                else:
                    prev_action = actions[-1]
                
                combined_features = torch.cat([
                    instruction_features[:, t:t+1, :],
                    start_state_features[:, t:t+1, :],
                    progress_features[:, t:t+1, :],
                    prev_action
                ], dim=-1)
                
                # LSTM解码
                output, hidden = self.trajectory_decoder(combined_features, hidden)
                
                # 时序增强
                enhanced_output = self.temporal_enhancer(output)
                
                # 预测动作
                action = self.action_predictor(enhanced_output)
                actions.append(action)
            
            predicted_actions = torch.cat(actions, dim=1)  # [batch_size, trajectory_length, action_dim]
        
        # 5. 应用正确的渐进式约束
        constrained_actions = self._apply_progressive_constraints(
            predicted_actions, start_state, instruction_id
        )
        
        return constrained_actions
    
    def _apply_progressive_constraints(self, actions, start_state, instruction_id):
        """应用渐进式约束 - 从起始状态开始，逐渐变化到目标动作"""
        batch_size = actions.size(0)
        
        # 正确扩展起始状态到轨迹长度
        start_state_expanded = start_state.unsqueeze(1).expand(-1, self.trajectory_length, -1)
        
        # 生成进度权重
        progress_weights = torch.linspace(0, 1, self.trajectory_length, device=start_state.device)
        progress_weights = progress_weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        
        # 获取指令特定的动作模式
        instruction_patterns = self.instruction_patterns[instruction_id]  # [batch_size, trajectory_length, action_dim]
        
        # 渐进式约束：从起始状态逐渐过渡到目标动作
        target_actions = start_state_expanded + instruction_patterns
        constrained_actions = start_state_expanded + progress_weights * (target_actions - start_state_expanded)
        
        return constrained_actions
    
    def compute_instruction_diversity_loss(self, instruction_ids):
        """计算指令差异损失"""
        if len(torch.unique(instruction_ids)) < 2:
            return torch.tensor(0.0, device=instruction_ids.device)
        
        # 计算指令嵌入之间的差异
        unique_ids = torch.unique(instruction_ids)
        if len(unique_ids) >= 2:
            embeddings = []
            for id in unique_ids:
                embed = self.instruction_embedding(torch.tensor([id], device=instruction_ids.device))
                embeddings.append(embed)
            
            embeddings = torch.cat(embeddings, dim=0)  # [num_unique, hidden_dim * 4]
            
            # 计算嵌入之间的平均距离
            total_diff = 0
            count = 0
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    diff = torch.norm(embeddings[i] - embeddings[j], p=2)
                    total_diff += diff
                    count += 1
            
            # 负号表示最小化负差异（即最大化差异）
            return -total_diff / count if count > 0 else torch.tensor(0.0, device=instruction_ids.device)
        
        return torch.tensor(0.0, device=instruction_ids.device)

class FixedTrajectoryDataset(Dataset):
    """修复的轨迹数据集"""
    
    def __init__(self, data_dir: str, file_names: List[str], trajectory_length: int = 100, 
                 norm_stats: Dict = None, augment: bool = True):
        self.data_dir = data_dir
        self.file_names = file_names
        self.trajectory_length = trajectory_length
        self.norm_stats = norm_stats
        self.augment = augment
        
        # 指令到ID的映射
        self.instruction_to_id = {'挥手': 0, '抱拳': 1}
        
    def __len__(self):
        return len(self.file_names) * 10  # 每个文件生成10个不同的起始点
    
    def __getitem__(self, idx):
        file_idx = idx // 10
        start_offset = (idx % 10) * 10  # 每10帧一个起始点
        
        file_name = self.file_names[file_idx]
        
        # 加载JSON轨迹数据
        trajectory_path = os.path.join(self.data_dir, file_name)
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
        
        # 提取观测数据
        observations = data['observations']
        
        # 提取关节数据
        joint_positions = np.array([obs['joint_pos'] for obs in observations])
        
        # 随机选择起始点
        if self.augment and len(joint_positions) > self.trajectory_length + start_offset:
            start_idx = start_offset
            # 确保有足够的长度
            max_start = len(joint_positions) - self.trajectory_length
            start_idx = min(start_idx, max_start)
        else:
            start_idx = 0
        
        # 提取轨迹
        if len(joint_positions) >= start_idx + self.trajectory_length:
            trajectory = joint_positions[start_idx:start_idx + self.trajectory_length]
        else:
            # 如果不够长，重复数据
            trajectory = joint_positions[start_idx:]
            repeat_times = (self.trajectory_length + len(trajectory) - 1) // len(trajectory)
            trajectory = np.tile(trajectory, (repeat_times, 1))[:self.trajectory_length]
        
        # 获取起始状态和完整轨迹
        start_state = trajectory[0]  # [state_dim]
        target_actions = trajectory  # [trajectory_length, state_dim]
        
        # 获取指令ID
        instruction = data.get('instruction', '挥手')
        instruction_id = self.instruction_to_id.get(instruction, 0)
        
        return {
            'start_state': start_state.astype(np.float32),
            'target_actions': target_actions.astype(np.float32),
            'instruction_id': np.array(instruction_id, dtype=np.int64),
            'file_name': file_name,
            'start_idx': start_idx
        }

def compute_fixed_dataset_stats(data_dir: str, file_names: List[str]):
    """计算修复数据集统计信息"""
    print("计算修复数据集统计信息...")
    
    all_trajectories = []
    
    for file_name in tqdm(file_names):
        trajectory_path = os.path.join(data_dir, file_name)
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
        
        observations = data['observations']
        joint_positions = np.array([obs['joint_pos'] for obs in observations])
        all_trajectories.append(joint_positions)
    
    all_trajectories = np.vstack(all_trajectories)
    
    # 计算统计信息
    state_mean = np.mean(all_trajectories, axis=0)
    state_std = np.std(all_trajectories, axis=0)
    state_std = np.clip(state_std, 0.1, np.inf)  # 恢复到原来的设置
    
    action_mean = state_mean.copy()
    action_std = state_std.copy()
    
    stats = {
        'state_mean': state_mean,
        'state_std': state_std,
        'action_mean': action_mean,
        'action_std': action_std,
        'state_min': np.min(all_trajectories, axis=0),
        'state_max': np.max(all_trajectories, axis=0),
    }
    
    print(f"状态数据: 均值范围[{state_mean.min():.3f}, {state_mean.max():.3f}], "
          f"标准差范围[{state_std.min():.3f}, {state_std.max():.3f}]")
    
    return stats

def train_fixed_trajectory_generator():
    """训练修复的轨迹生成器"""
    print("开始训练修复的轨迹生成器...")
    
    # 配置参数
    config = {
        'state_dim': 26,
        'action_dim': 26,
        'instruction_vocab_size': 2,
        'hidden_dim': 256,
        'num_layers': 4,
        'trajectory_length': 100,  # 生成100步的完整轨迹
        'dropout': 0.1
    }
    
    # 数据准备
    data_dir = './trajectories'
    file_names = [f'wave_{i:03d}.json' for i in range(1, 9)] + [f'welcome_{i:03d}.json' for i in range(1, 9)]
    
    print(f"使用数据文件: {len(file_names)} 个")
    
    # 计算数据集统计
    norm_stats = compute_fixed_dataset_stats(data_dir, file_names)
    
    # 创建数据集
    train_dataset = FixedTrajectoryDataset(data_dir, file_names[:12], config['trajectory_length'], norm_stats, augment=True)
    val_dataset = FixedTrajectoryDataset(data_dir, file_names[12:], config['trajectory_length'], norm_stats, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FixedTrajectoryGenerator(config).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    
    # 损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    num_epochs = 1000
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    # 创建保存目录
    save_dir = './checkpoints/fixed_trajectory_generator'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始训练 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            start_states = batch['start_state'].to(device).float()
            target_actions = batch['target_actions'].to(device).float()
            instruction_ids = batch['instruction_id'].to(device).long()
            
            # 标准化
            state_mean = torch.from_numpy(norm_stats['state_mean']).to(device).float()
            state_std = torch.from_numpy(norm_stats['state_std']).to(device).float()
            action_mean = torch.from_numpy(norm_stats['action_mean']).to(device).float()
            action_std = torch.from_numpy(norm_stats['action_std']).to(device).float()
            
            start_states_norm = (start_states - state_mean) / state_std
            target_actions_norm = (target_actions - action_mean) / action_std
            
            # 前向传播
            predicted_actions_norm = model(start_states_norm, instruction_ids, target_actions_norm)
            
            # 计算损失
            mse_loss = criterion(predicted_actions_norm, target_actions_norm)
            diversity_loss = model.compute_instruction_diversity_loss(instruction_ids)
            loss = mse_loss + model.instruction_diff_weight * diversity_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # 检查梯度
            if batch_idx == 0:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f'  梯度范数: {total_norm:.6f}, MSE损失: {mse_loss.item():.6f}, Diversity损失: {diversity_loss.item():.6f}, 总损失: {loss.item():.6f}')
        
        # 验证阶段
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                start_states = batch['start_state'].to(device).float()
                target_actions = batch['target_actions'].to(device).float()
                instruction_ids = batch['instruction_id'].to(device).long()
                
                # 标准化
                state_mean = torch.from_numpy(norm_stats['state_mean']).to(device).float()
                state_std = torch.from_numpy(norm_stats['state_std']).to(device).float()
                action_mean = torch.from_numpy(norm_stats['action_mean']).to(device).float()
                action_std = torch.from_numpy(norm_stats['action_std']).to(device).float()
                
                start_states_norm = (start_states - state_mean) / state_std
                target_actions_norm = (target_actions - action_mean) / action_std
                
                # 前向传播
                predicted_actions_norm = model(start_states_norm, instruction_ids)
                
                # 计算损失
                loss = criterion(predicted_actions_norm, target_actions_norm)
                val_losses.append(loss.item())
        
        # 计算平均损失
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # 学习率调度
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'norm_stats': norm_stats,
                'best_val_loss': best_val_loss
            }
            
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print(f"  保存最佳模型 - Val Loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print(f"早停: {patience} 轮验证损失没有改善")
            break
    
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return model, norm_stats, config

if __name__ == "__main__":
    model, norm_stats, config = train_fixed_trajectory_generator()