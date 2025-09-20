#!/usr/bin/env python3
"""
检查点管理工具函数
Checkpoint management utility functions for FLOAT training
"""

import time
import torch


def save_checkpoint(model, optimizer, scheduler, step, loss, save_path):
    """保存检查点"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, save_path)
    print(f"检查点已保存: {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """加载检查点"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    print(f"从步骤 {step} 恢复训练，损失: {loss:.6f}")
    return step, loss


def load_weight(model, checkpoint_path: str, device: torch.device) -> None:
    """加载模型权重"""
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for model_name, model_param in model.named_parameters():
            if model_name in state_dict:
                model_param.copy_(state_dict[model_name].to(device))
                print("successfully loaded", model_name)
            elif "wav2vec2" in model_name: 
                pass
            else:
                print(f"! Warning; {model_name} not found in state_dict.")

    del state_dict
