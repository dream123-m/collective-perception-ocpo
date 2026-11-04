import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from torch.distributions.bernoulli import Bernoulli

class ActorNetwork(nn.Module):
    """Actor网络：输出动作概率"""

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
                nn.Tanh()
            ])
        
        self.shared_net = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # 初始化参数
        self.apply(self._init_weights)
        # 特别处理最后一层，使其输出接近0
        nn.init.orthogonal_(self.action_head.weight.data, gain=0.01)
        nn.init.constant_(self.action_head.bias.data, 0.0)
        
    def _init_weights(self, module):
        """正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=1.0)
            nn.init.constant_(module.bias.data, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            state: 状态张量，shape=(batch_size, state_dim)
        Returns:
            action_probs: 动作概率，shape=(batch_size, action_dim)
        """
        features = self.shared_net(state)
        action_logits = self.action_head(features)
        return torch.sigmoid(action_logits)
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor, 
                  deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取动作和对数概率
        Args:
            state: 状态张量
            action_mask: 动作掩码
            deterministic: 是否使用确定性策略
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        action_probs = self.forward(state)
        
        # 应用动作掩码
        action_probs = action_probs * action_mask.float()
        
        # 处理全零掩码情况
        if torch.all(action_mask == 0):
            action_probs = torch.ones_like(action_probs) * 1e-6
        
        # 创建伯努利分布
        dist = Bernoulli(probs=action_probs)
        
        if deterministic:
            action = (action_probs > 0.5).float()
        else:
            action = dist.sample()
        
        # 计算对数概率
        log_prob = dist.log_prob(action)
        
        return action, log_prob.sum(dim=-1)

class CriticNetwork(nn.Module):
    """Critic网络：关注全局广播分布和共享率"""

    def __init__(self, num_targets: int, hidden_dims: List[int]):
        """
        初始化Critic网络
        Args:
            num_targets: 目标车辆总数
            hidden_dims: 隐藏层维度列表
        """
        super(CriticNetwork, self).__init__()
        
        # 输入维度 = 2*目标数量 + 1 (广播标志 + 广播计数 + 共享率)
        input_dim = 2 * num_targets + 1
        
        # 构建MLP层
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.Tanh()
            ])
        
        self.shared_net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化参数
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data, gain=1.0)
            nn.init.constant_(module.bias.data, 0)
    
    def forward(self, broadcast_state: torch.Tensor, coverage_rate: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            broadcast_state: 广播状态张量，shape=(batch_size, 2*num_targets)
                           包含每个目标的广播标志和广播计数
            coverage_rate: 当前覆盖率，shape=(batch_size, 1)
        Returns:
            value: 状态值，shape=(batch_size, 1)
        """
        # 拼接输入
        x = torch.cat([broadcast_state, coverage_rate], dim=1)
        
        # 前向传播
        features = self.shared_net(x)
        value = self.value_head(features)
        
        return value 