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

    def do_cfg(self, x, dropout_prob):
        """
        实现 classifier-free guidance 的 dropout
        
        Args:
            x: 输入张量 (batch_size, ...)
            dropout_prob: dropout 概率
            
        Returns:
            应用了 CFG dropout 的张量
        """
        if dropout_prob == 0.0:
            return x
            
        batch_size = x.shape[0]
        # 对整个 batch 进行 mask（每个样本要么全部被 mask，要么全部不被 mask）
        cfg_mask = torch.rand(batch_size, device=x.device) < dropout_prob
        
        # 将 mask 扩展到与 x 相同的形状
        cfg_mask = cfg_mask.view(batch_size, *([1] * (x.dim() - 1)))
        
        # 创建无条件输入（通常是零向量）
        uncond = torch.zeros_like(x)
        
        # 应用 mask：如果 cfg_mask 为 True，使用无条件输入；否则使用原始输入
        return torch.where(cfg_mask, uncond, x)

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
        
        wa = self.do_cfg(wa, self.opt.audio_dropout_prob)
        wr = self.do_cfg(wr, self.opt.ref_dropout_prob)
        we = self.do_cfg(we, self.opt.emotion_dropout_prob)
        prev_x = self.do_cfg(prev_x, self.opt.prev_frame_dropout_prob)
        prev_wa = self.do_cfg(prev_wa, self.opt.prev_frame_dropout_prob)
        
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