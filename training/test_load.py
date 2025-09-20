#!/usr/bin/env python3
"""
测试 load 函数
"""

import os
import sys
import torch

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.base_options import BaseOptions
from training.dataset import create_dataloader
from models.float.FLOAT import FLOAT
from training.train import load


def test_load_function():
    """测试 load 函数"""
    print("测试 load 函数...")
    
    # 解析选项
    opt = BaseOptions().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.rank = device
    
    # 创建模型
    print("创建模型...")
    model = FLOAT(opt).to(device)
    model.audio_encoder.requires_grad_(False)
    model.emotion_encoder.requires_grad_(False)
    model.motion_autoencoder.requires_grad_(False)
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_dataloader(
        data_root="../datasets/ravdess_processed",
        batch_size=2,  # 小批次
        num_workers=0,  # 避免多进程问题
        train=True,
        opt=opt
    )
    
    # 获取一个批次
    print("获取批次数据...")
    batch_data = next(iter(dataloader))
    print(f"原始批次数据类型: {type(batch_data)}")
    print(f"原始批次大小: {len(batch_data)}")
    
    if len(batch_data) > 0:
        print(f"第一个数据项: {batch_data[0]}")
    
    # 测试 load 函数
    print("测试 load 函数...")
    try:
        processed_data = load(batch_data, model, device, opt)
        print("✅ load 函数执行成功！")
        print(f"处理后的数据键: {list(processed_data.keys())}")
        
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
                
    except Exception as e:
        print(f"❌ load 函数执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_load_function()
