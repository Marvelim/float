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
        uncond = torch.zeros_like(x)
        cfg_mask = torch.rand(x) < dropout_prob
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
        t = torch.randn(batch_size, device)
        
        # 采样噪声
        x0 = torch.randn_like(x1)
        x_t = (1 - t) * x0 + t * x1
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

        out_prev = model_output[:, :self.num_prev_frames]
        out_current = model_output[:, self.num_prev_frames:]

        loss_prev = F.l1_loss(out_prev, prev_x)
        loss_current = F.l1_loss(out_current, v_t)
        loss = loss_prev + loss_current

        return {
            'loss': loss,
            'loss_prev': loss_prev,
            'loss_current': loss_current
        }


class FlowMatchingTrainer:
    """
    Flow Matching 训练器
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: RectifiedFlow.loss,
                 device: torch.device,
                 gradient_clip_val: float = 1.0):
        """
        初始化训练器
        
        Args:
            model: FMT 模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 设备
            gradient_clip_val: 梯度裁剪值
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.gradient_clip_val = gradient_clip_val
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            包含损失信息的字典
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 准备数据
        x1 = batch['motion_latent'].to(self.device)  # 目标动作潜在表示
        
        conditions = {
            'wa': batch['audio_features'].to(self.device),
            'wr': batch['reference_motion'].to(self.device),
            'we': batch['emotion_features'].to(self.device)
        }
        
        # 准备前一帧条件（如果存在）
        prev_conditions = None
        if 'prev_motion' in batch:
            prev_conditions = {
                'prev_x': batch['prev_motion'].to(self.device),
                'prev_wa': batch['prev_audio'].to(self.device)
            }
        
        # 前向传播
        loss_dict = self.loss_fn(self.model, x1, conditions, prev_conditions)
        loss = loss_dict['loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        
        # 更新参数
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'grad_norm': torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf')).item()
        }
    
    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行验证步骤
        
        Args:
            batch: 批次数据
            
        Returns:
            包含验证损失的字典
        """
        self.model.eval()
        
        # 准备数据
        x1 = batch['motion_latent'].to(self.device)
        
        conditions = {
            'wa': batch['audio_features'].to(self.device),
            'wr': batch['reference_motion'].to(self.device),
            'we': batch['emotion_features'].to(self.device)
        }
        
        prev_conditions = None
        if 'prev_motion' in batch:
            prev_conditions = {
                'prev_x': batch['prev_motion'].to(self.device),
                'prev_wa': batch['prev_audio'].to(self.device)
            }
        
        # 前向传播
        loss_dict = self.loss_fn(self.model, x1, conditions, prev_conditions)
        
        return {
            'val_loss': loss_dict['loss'].item()
        }