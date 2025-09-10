"""
Instruction Module - 处理自然语言指令
"""
import random
from typing import List, Dict, Any, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import config


class InstructionModule:
    """自然语言指令处理模块
    
    负责管理和生成自然语言指令，支持指令的增删改查，
    并提供指令编码功能用于神经网络输入。
    """
    
    def __init__(self, 
                 instruction_list: Optional[List[str]] = None,
                 vocab_size: int = 1000,
                 max_length: int = 20):
        """
        初始化指令模块
        
        Args:
            instruction_list: 可选的指令列表，如果为None则使用默认指令
            vocab_size: 词汇表大小
            max_length: 指令最大长度
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        if instruction_list is None:
            self.instruction_list = config.INSTRUCTIONS.copy()
        else:
            self.instruction_list = instruction_list.copy()
        
        # 构建词汇表
        self.vocab = self._build_vocab()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # 指令到技能ID的映射
        self.instruction_to_skill = self._build_instruction_skill_mapping()
    
    def _build_vocab(self) -> List[str]:
        """构建词汇表"""
        # 特殊token
        special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>"]
        
        # 从指令列表中提取字符
        all_chars = set()
        for instruction in self.instruction_list:
            all_chars.update(list(instruction))
        
        # 构建词汇表
        vocab = special_tokens + sorted(list(all_chars))
        
        # 限制词汇表大小
        if len(vocab) > self.vocab_size:
            vocab = vocab[:self.vocab_size]
        
        return vocab
    
    def _build_instruction_skill_mapping(self) -> Dict[str, int]:
        """构建指令到技能ID的映射"""
        mapping = {
            # 行走相关
            "向前走": 0, "向后走": 0, "前进": 0, "后退": 0,
            # 转向相关
            "向左转": 1, "向右转": 1, "左转": 1, "右转": 1,
            # 手臂动作
            "抬起左臂": 2, "抬起右臂": 2, "挥手": 2, "打招呼": 2,
            # 身体动作
            "鞠躬": 3, "坐下": 3, "蹲下": 3,
            # 静止
            "停止": 4, "站起来": 4, "站立": 4,
            # 腿部动作
            "抬左腿": 5, "抬右腿": 5, "跳跃": 5,
        }
        
        # 为其他指令设置默认值
        for instruction in self.instruction_list:
            if instruction not in mapping:
                mapping[instruction] = 0  # 默认为行走技能
        
        return mapping
    
    def get_instruction(self) -> str:
        """
        获取一个随机指令
        
        Returns:
            str: 自然语言指令
        """
        return random.choice(self.instruction_list)
    
    def get_all_instructions(self) -> List[str]:
        """
        获取所有可用指令
        
        Returns:
            List[str]: 所有指令列表
        """
        return self.instruction_list.copy()
    
    def add_instruction(self, instruction: str):
        """
        添加新指令
        
        Args:
            instruction: 要添加的指令
        """
        if instruction not in self.instruction_list:
            self.instruction_list.append(instruction)
            # 更新词汇表
            self._update_vocab(instruction)
    
    def remove_instruction(self, instruction: str):
        """
        移除指令
        
        Args:
            instruction: 要移除的指令
        """
        if instruction in self.instruction_list:
            self.instruction_list.remove(instruction)
    
    def encode_instruction(self, instruction: str) -> np.ndarray:
        """
        编码指令为数字序列
        
        Args:
            instruction: 自然语言指令
            
        Returns:
            np.ndarray: 编码后的指令序列 [max_length]
        """
        # 添加开始和结束标记
        tokens = ["<START>"] + list(instruction) + ["<END>"]
        
        # 转换为索引
        encoded = []
        for token in tokens:
            if token in self.word_to_idx:
                encoded.append(self.word_to_idx[token])
            else:
                encoded.append(self.word_to_idx["<UNK>"])
        
        # 填充到固定长度
        while len(encoded) < self.max_length:
            encoded.append(self.word_to_idx["<PAD>"])
        
        # 截断到最大长度
        encoded = encoded[:self.max_length]
        
        return np.array(encoded, dtype=np.int64)
    
    def decode_instruction(self, encoded: np.ndarray) -> str:
        """
        解码数字序列为指令
        
        Args:
            encoded: 编码的指令序列
            
        Returns:
            str: 解码后的指令
        """
        tokens = []
        for idx in encoded:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                if token in ["<PAD>", "<START>", "<END>"]:
                    continue
                tokens.append(token)
        
        return "".join(tokens)
    
    def get_skill_id(self, instruction: str) -> int:
        """
        获取指令对应的技能ID
        
        Args:
            instruction: 自然语言指令
            
        Returns:
            int: 技能ID
        """
        return self.instruction_to_skill.get(instruction, 0)
    
    def get_skill_name(self, skill_id: int) -> str:
        """
        根据技能ID获取技能名称
        
        Args:
            skill_id: 技能ID
            
        Returns:
            str: 技能名称
        """
        skill_names = ["walk", "turn", "wave", "body", "stand", "leg"]
        return skill_names[skill_id] if 0 <= skill_id < len(skill_names) else "custom"
    
    def get_instruction_embedding(self, instruction: str, embedding_dim: int = 128) -> np.ndarray:
        """
        获取指令的嵌入表示（简化版本）
        
        Args:
            instruction: 自然语言指令
            embedding_dim: 嵌入维度
            
        Returns:
            np.ndarray: 指令嵌入向量
        """
        # 简单的字符级嵌入
        encoded = self.encode_instruction(instruction)
        
        # 创建随机嵌入矩阵（实际应用中应该使用预训练的嵌入）
        np.random.seed(42)
        embedding_matrix = np.random.randn(len(self.vocab), embedding_dim) * 0.1
        
        # 获取嵌入
        embedding = np.zeros(embedding_dim)
        count = 0
        for idx in encoded:
            if idx < len(embedding_matrix):
                embedding += embedding_matrix[idx]
                count += 1
        
        if count > 0:
            embedding /= count
        
        return embedding.astype(np.float32)
    
    def get_batch_instructions(self, batch_size: int) -> List[str]:
        """
        获取批量指令
        
        Args:
            batch_size: 批次大小
            
        Returns:
            List[str]: 指令列表
        """
        return [self.get_instruction() for _ in range(batch_size)]
    
    def get_batch_encodings(self, instructions: List[str]) -> np.ndarray:
        """
        获取批量编码
        
        Args:
            instructions: 指令列表
            
        Returns:
            np.ndarray: 批量编码 [batch_size, max_length]
        """
        return np.stack([self.encode_instruction(inst) for inst in instructions])
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取指令模块统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_instructions": len(self.instruction_list),
            "vocab_size": len(self.vocab),
            "max_length": self.max_length,
            "skill_distribution": self._get_skill_distribution(),
            "instructions": self.instruction_list
        }
    
    def _get_skill_distribution(self) -> Dict[str, int]:
        """获取技能分布"""
        distribution = {}
        for instruction in self.instruction_list:
            skill_id = self.get_skill_id(instruction)
            skill_name = self.get_skill_name(skill_id)
            distribution[skill_name] = distribution.get(skill_name, 0) + 1
        return distribution
    
    def _update_vocab(self, new_instruction: str):
        """更新词汇表"""
        for char in new_instruction:
            if char not in self.word_to_idx and len(self.vocab) < self.vocab_size:
                self.vocab.append(char)
                self.word_to_idx[char] = len(self.vocab) - 1
                self.idx_to_word[len(self.vocab) - 1] = char
    
    def save_vocab(self, filepath: str):
        """保存词汇表"""
        import json
        
        vocab_data = {
            "vocab": self.vocab,
            "word_to_idx": self.word_to_idx,
            "instruction_to_skill": self.instruction_to_skill,
            "instructions": self.instruction_list
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, filepath: str):
        """加载词汇表"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data["vocab"]
        self.word_to_idx = vocab_data["word_to_idx"]
        self.instruction_to_skill = vocab_data["instruction_to_skill"]
        self.instruction_list = vocab_data["instructions"]
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}