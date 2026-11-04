import torch
import torch.nn as nn
import numpy as np
from typing import List

class ActorNetwork(nn.Module):
    """Actor网络：输出连续动作概率 [0,1]"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        """
        初始化Actor网络
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
        """
        super(ActorNetwork, self).__init__()
        
        # 构建MLP层
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ])
        
        self.shared_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 初始化参数
        self.apply(self._init_weights)
        # 特别处理最后一层
        nn.init.uniform_(self.action_head.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.action_head.bias.data, -3e-3, 3e-3)
        
    def _init_weights(self, module):
        """Xavier初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态张量，shape=(batch_size, state_dim)
        Returns:
            action_probs: 动作概率，shape=(batch_size, action_dim)，范围[0,1]
        """
        features = self.shared_net(state)
        action_logits = self.action_head(features)
        return torch.sigmoid(action_logits)


class CriticNetwork(nn.Module):
    """Critic网络：评估状态-动作对的Q值"""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int, hidden_dims: List[int]):
        """
        初始化Critic网络
        Args:
            state_dim: 单个智能体状态维度
            action_dim: 单个智能体动作维度
            num_agents: 智能体数量
            hidden_dims: 隐藏层维度列表
        """
        super(CriticNetwork, self).__init__()
        
        # 输入维度 = 所有智能体的状态和动作
        input_dim = num_agents * (state_dim + action_dim)
        
        # 构建MLP层
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU()
            ])
        
        self.shared_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化参数
        self.apply(self._init_weights)
        nn.init.uniform_(self.q_head.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.q_head.bias.data, -3e-3, 3e-3)
        
    def _init_weights(self, module):
        """Xavier初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight.data)
            nn.init.constant_(module.bias.data, 0)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            states: 所有智能体状态张量，shape=(batch_size, num_agents, state_dim)
            actions: 所有智能体动作张量，shape=(batch_size, num_agents, action_dim)
        Returns:
            q_value: Q值，shape=(batch_size, 1)
        """
        batch_size = states.shape[0]
        
        # 展平状态和动作
        states_flat = states.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        
        # 拼接输入
        x = torch.cat([states_flat, actions_flat], dim=1)
        
        # 前向传播
        features = self.shared_net(x)
        q_value = self.q_head(features)
        
        return q_value