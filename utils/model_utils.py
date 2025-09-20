#!/usr/bin/env python3
"""
模型创建和初始化工具函数
Model creation and initialization utility functions for FLOAT training
"""

import torch
import torch.optim as optim
from models.float.FMT import FlowMatchingTransformer
from models.float.FLOAT import FLOAT
from .checkpoint_utils import load_weight


def create_model(opt, device):
    """创建和初始化模型"""
    print("创建 FLOAT 模型...")
    
    # 设置设备到 opt 中
    opt.rank = device
    
    model = FLOAT(opt).to(device)
    load_weight(model, opt.ckpt_path, device)

    model.fmt = FlowMatchingTransformer(opt).to(device)
    # 确保所有子模块都在GPU上
    model.audio_encoder = model.audio_encoder.to(device)
    model.emotion_encoder = model.emotion_encoder.to(device)
    model.motion_autoencoder = model.motion_autoencoder.to(device)
    
    # 冻结不需要训练的参数
    model.audio_encoder.requires_grad_(False)
    model.emotion_encoder.requires_grad_(False)
    model.motion_autoencoder.requires_grad_(False)

    model.fmt.train()

    # 验证所有模型组件都在正确的设备上
    print("验证模型设备分配...")
    print(f"主模型设备: {next(model.parameters()).device}")
    print(f"音频编码器设备: {next(model.audio_encoder.parameters()).device}")
    print(f"情感编码器设备: {next(model.emotion_encoder.parameters()).device}")
    print(f"运动自编码器设备: {next(model.motion_autoencoder.parameters()).device}")
    print(f"FMT设备: {next(model.fmt.parameters()).device}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fmt_params = sum(p.numel() for p in model.fmt.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"FMT 参数数量: {fmt_params:,}")
    
    return model


def create_optimizer(model, opt):
    """创建优化器"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    return optimizer


def create_scheduler(optimizer, opt):
    """创建学习率调度器"""
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=opt.steps,
        eta_min=opt.lr * 0.01
    )
    return scheduler