"""
Training utilities for FLOAT model
FLOAT 模型训练工具函数
"""

import os
import random
import logging
import numpy as np
from typing import Dict, Any, Optional, Union
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, LinearLR, 
    SequentialLR, CosineAnnealingWarmRestarts
)


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重现性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保 CUDA 操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: int = logging.INFO, 
                 log_file: Optional[str] = None) -> logging.Logger:
    """
    设置日志系统
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger("FLOAT")
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   lr_scheduler: Optional[Any],
                   epoch: int,
                   global_step: int,
                   best_val_loss: float,
                   config: Any,
                   filepath: str):
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        epoch: 当前轮次
        global_step: 全局步数
        best_val_loss: 最佳验证损失
        config: 配置对象
        filepath: 保存路径
    """
    # 获取模型状态字典
    if hasattr(model, 'module'):
        # 分布式训练
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config,
    }
    
    # 保存学习率调度器状态
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 保存检查点
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, 
                   device: torch.device) -> Dict[str, Any]:
    """
    加载训练检查点
    
    Args:
        filepath: 检查点文件路径
        device: 设备
        
    Returns:
        检查点字典
    """
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def get_lr_scheduler(optimizer: optim.Optimizer,
                    scheduler_type: str = "cosine",
                    num_epochs: int = 100,
                    warmup_epochs: int = 5,
                    min_lr: float = 1e-6,
                    step_size: int = 30,
                    gamma: float = 0.1) -> Optional[Any]:
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        num_epochs: 总轮次
        warmup_epochs: 预热轮次
        min_lr: 最小学习率
        step_size: 步长（用于 StepLR）
        gamma: 衰减因子（用于 StepLR）
        
    Returns:
        学习率调度器
    """
    if scheduler_type.lower() == "cosine":
        if warmup_epochs > 0:
            # 带预热的余弦调度
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=min_lr
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=min_lr
            )
    
    elif scheduler_type.lower() == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type.lower() == "linear":
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.param_groups[0]['lr'],
            total_iters=num_epochs
        )
    
    elif scheduler_type.lower() == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_epochs // 4,
            T_mult=2,
            eta_min=min_lr
        )
    
    elif scheduler_type.lower() == "none":
        scheduler = None
    
    else:
        raise ValueError(f"不支持的学习率调度器: {scheduler_type}")
    
    return scheduler


class AverageMeter:
    """平均值计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置计数器"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: Union[float, Dict[str, float]], n: int = 1):
        """
        更新计数器
        
        Args:
            val: 值或值字典
            n: 样本数量
        """
        if isinstance(val, dict):
            # 如果是字典，更新所有键值
            if not hasattr(self, '_dict_mode'):
                self._dict_mode = True
                self._dict_vals = {}
                self._dict_avgs = {}
                self._dict_sums = {}
                self._dict_counts = {}
            
            for key, value in val.items():
                if key not in self._dict_vals:
                    self._dict_vals[key] = 0
                    self._dict_avgs[key] = 0
                    self._dict_sums[key] = 0
                    self._dict_counts[key] = 0
                
                self._dict_vals[key] = value
                self._dict_sums[key] += value * n
                self._dict_counts[key] += n
                self._dict_avgs[key] = self._dict_sums[key] / self._dict_counts[key]
        else:
            # 单个值
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    @property
    def avg(self):
        """获取平均值"""
        if hasattr(self, '_dict_mode') and self._dict_mode:
            return self._dict_avgs
        else:
            return self._avg
    
    @avg.setter
    def avg(self, value):
        """设置平均值"""
        self._avg = value


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """
    获取模型大小（MB）
    
    Args:
        model: 模型
        
    Returns:
        模型大小（MB）
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def setup_wandb(config: Any, project_name: Optional[str] = None):
    """
    设置 Weights & Biases 日志记录
    
    Args:
        config: 配置对象
        project_name: 项目名称
    """
    try:
        import wandb
        
        wandb.init(
            project=project_name or config.training.wandb_project,
            entity=config.training.wandb_entity,
            name=config.experiment_name,
            config=config.to_dict() if hasattr(config, 'to_dict') else config,
            tags=["rectified_flow", "float", "flow_matching"]
        )
    except ImportError:
        print("警告: wandb 未安装，跳过 wandb 初始化")
    except Exception as e:
        print(f"警告: wandb 初始化失败: {e}")


def cleanup_checkpoints(checkpoint_dir: str, max_keep: int = 5):
    """
    清理旧的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        max_keep: 最大保留数量
    """
    checkpoint_files = []
    
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_step_") and file.endswith(".pth"):
            filepath = os.path.join(checkpoint_dir, file)
            mtime = os.path.getmtime(filepath)
            checkpoint_files.append((mtime, filepath))
    
    # 按修改时间排序
    checkpoint_files.sort(reverse=True)
    
    # 删除多余的检查点
    for _, filepath in checkpoint_files[max_keep:]:
        try:
            os.remove(filepath)
            print(f"删除旧检查点: {filepath}")
        except OSError as e:
            print(f"删除检查点失败 {filepath}: {e}")


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def get_gpu_memory_info() -> Dict[str, float]:
    """
    获取 GPU 内存信息
    
    Returns:
        包含内存信息的字典
    """
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)   # GB
        max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
        
        return {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved,
            'max_gb': max_memory,
            'utilization': memory_allocated / max_memory
        }
    else:
        return {
            'allocated_gb': 0,
            'reserved_gb': 0,
            'max_gb': 0,
            'utilization': 0
        }


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """
    保存指标到文件
    
    Args:
        metrics: 指标字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    从文件加载指标
    
    Args:
        filepath: 文件路径
        
    Returns:
        指标字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


class EarlyStopping:
    """早停类"""
    
    def __init__(self, 
                 patience: int = 7,
                 min_delta: float = 0,
                 restore_best_weights: bool = True):
        """
        初始化早停
        
        Args:
            patience: 耐心值（多少个 epoch 没有改善就停止）
            min_delta: 最小改善幅度
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证损失
            model: 模型
            
        Returns:
            是否应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
