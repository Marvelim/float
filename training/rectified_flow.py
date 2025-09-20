"""
Rectified Flow implementation for Flow Matching Training
基于 Rectified Flow 方法的流匹配训练实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional

class RectifiedFlow(nn.Module):
    """
    Rectified Flow 损失函数模块
    """
    
    def __init__(self, opt):
        """
        初始化损失函数
        
        Args:
            sigma_min: 最小噪声水平
            sigma_max: 最大噪声水平
            loss_type: 损失类型 ('mse', 'l1', 'huber')
        """
        super().__init__()
        self.opt = opt

    def loss(self, 
               model,
               x1: torch.Tensor,
               conditions: Dict[str, torch.Tensor],
               prev_conditions: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播计算损失
        
        Args:
            model: FMT 模型
            x1: 目标数据（真实动作序列）
            conditions: 条件信息 {'wa': audio, 'wr': reference, 'we': emotion}
            prev_conditions: 前一帧条件信息（用于自回归）
            
        Returns:
            包含损失和其他信息的字典
        """
        batch_size = x1.shape[0]
        device = x1.device
        
        # 采样时间步
        t = torch.rand(batch_size, device=device)
        
        # 采样噪声
        x0 = torch.randn_like(x1)
        
        # 将 t 扩展到与 x1 相同的形状进行广播
        t_expanded = t.view(batch_size, *([1] * (x1.dim() - 1)))
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        v_t = x1 - x0
        
        wa, wr, we = conditions['wa'], conditions['wr'], conditions['we']
        prev_x, prev_wa = prev_conditions['prev_x'], prev_conditions['prev_wa']
        
        model_output = model(
            t=t,
            x=x_t,
            wa=wa,
            wr=wr,
            we=we,
            prev_x=prev_x,
            prev_wa=prev_wa,
            train=True
        )

        out_prev = model_output[:, :self.opt.num_prev_frames]
        out_current = model_output[:, self.opt.num_prev_frames:]

        loss_prev = F.l1_loss(out_prev, prev_x)
        loss_current = F.l1_loss(out_current, v_t)
        loss = loss_prev + loss_current

        return {
            'loss': loss,
            'loss_prev': loss_prev,
            'loss_current': loss_current
        }