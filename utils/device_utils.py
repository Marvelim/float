#!/usr/bin/env python3
"""
设备管理工具函数
Device management utility functions for FLOAT training
"""

import torch


def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    
    return device


def get_device_info():
    """获取设备信息"""
    if torch.cuda.is_available():
        return {
            'device_type': 'cuda',
            'device_name': torch.cuda.get_device_name(),
            'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'device_count': torch.cuda.device_count()
        }
    else:
        return {
            'device_type': 'cpu',
            'device_name': 'CPU',
            'total_memory_gb': None,
            'device_count': 1
        }
