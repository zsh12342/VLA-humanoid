#!/usr/bin/env python3
"""
方向优先的动作预测器模块
确保方向正确性为基本要求
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

class SimpleDirectionActionPredictor(nn.Module):
    """方向优先的动作预测器 - 方向正确性为基本要求"""
    
    def __init__(self, base_model, action_dim=26, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 动作序列回归器 - 更深层的网络
        self.action_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, action_dim * 16)  # 输出16个时间步的动作变化量
        )
        
        self.condition_dim = None
        
    def forward(self, 
                instruction_ids: torch.Tensor,
                state: torch.Tensor,
                visual_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """方向优先的前向传播"""
        
        # 获取基础模型的条件编码
        with torch.no_grad():
            base_outputs = self.base_model(
                instruction_ids=instruction_ids,
                state=state,
                visual_features=visual_features,
                target_actions=None
            )
            condition = base_outputs['condition']  # [batch_size, condition_dim]
        
        # 初始化条件维度
        if self.condition_dim is None:
            self.condition_dim = condition.size(1)
        
        # 预测所有关节的变化量
        action_changes_flat = self.action_regressor(condition)  # [batch_size, action_dim * 16]
        
        # 重塑为变化量序列
        batch_size = condition.size(0)
        predicted_changes = action_changes_flat.view(batch_size, 16, self.action_dim)  # [batch_size, seq_len, action_dim]
        
        return {
            'predicted_changes': predicted_changes,
            'condition': condition
        }

def create_simple_direction_model(base_model, action_dim=26, hidden_dim=256):
    """创建方向优先的动作预测器"""
    return SimpleDirectionActionPredictor(
        base_model=base_model,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )