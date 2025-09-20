#!/usr/bin/env python3
"""
FLOAT 训练脚本 - 基于 Rectified Flow 的流匹配训练
FLOAT Training Script - Flow Matching Training with Rectified Flow

基于 rectified_flow.py 和 dataset.py 的简化训练实现
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入工具函数（使用绝对导入避免与 training/utils.py 冲突）
import utils.wandb_utils as wandb_utils
import utils.memory_utils as memory_utils
import utils.device_utils as device_utils
import utils.data_utils as data_utils
import utils.video_utils as video_utils
import utils.model_utils as model_utils

# 导入具体函数
from utils.wandb_utils import init_wandb
from utils.memory_utils import clear_memory, clear_cache
from utils.device_utils import setup_device
from utils.data_utils import load, get_batch_sample
from utils.video_utils import ensure
from utils.model_utils import create_model, create_optimizer, create_scheduler

from options.base_options import BaseOptions
from training.dataset import create_dataloader
from training.rectified_flow import RectifiedFlow

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FLOAT 训练脚本')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--data-root', type=str, default='../datasets/ravdess_processed', help='数据根目录')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--steps', type=int, default=200000, help='训练步数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=1, help='数据加载器工作进程数')
    parser.add_argument('--log-step', type=int, default=500, help='日志记录间隔')
    parser.add_argument('--sample-step', type=int, default=2000, help='样本生成间隔')
    parser.add_argument('--save-step', type=int, default=10000, help='检查点保存间隔')
    parser.add_argument('--ckpt_path',
				default="../checkpoints/float.pth", type=str, help='checkpoint path')
    
    # wandb 相关参数
    parser.add_argument('--use-wandb', action='store_true', help='是否使用 wandb 进行实验跟踪')
    parser.add_argument('--wandb-project', type=str, default='float-training', help='wandb 项目名称')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb 实体名称')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb 运行名称')
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=[], help='wandb 标签')
    
    args = parser.parse_args()
    
    # 解析选项
    opt = BaseOptions().parse()
    
    # 更新选项
    opt.data_root = args.data_root
    opt.batch_size = args.batch_size
    opt.steps = args.steps
    opt.lr = args.lr
    opt.num_workers = args.num_workers
    opt.log_step = args.log_step
    opt.sample_step = args.sample_step
    opt.save_step = args.save_step
    opt.ckpt_path = args.ckpt_path
    
    # 更新 wandb 选项
    opt.use_wandb = args.use_wandb
    opt.wandb_project = args.wandb_project
    opt.wandb_entity = args.wandb_entity
    opt.wandb_run_name = args.wandb_run_name
    opt.wandb_tags = args.wandb_tags
    
    print("=" * 50)
    print("FLOAT 训练配置")
    print("=" * 50)
    print(f"数据根目录: {opt.data_root}")
    print(f"批次大小: {opt.batch_size}")
    print(f"训练步数: {opt.steps}")
    print(f"学习率: {opt.lr}")
    print(f"工作进程数: {opt.num_workers}")
    print(f"使用 wandb: {opt.use_wandb}")
    if opt.use_wandb:
        print(f"wandb 项目: {opt.wandb_project}")
        print(f"wandb 实体: {opt.wandb_entity}")
        print(f"wandb 运行名称: {opt.wandb_run_name}")
        print(f"wandb 标签: {opt.wandb_tags}")
    print("=" * 50)
    
    # 设置设备
    device = setup_device()
    
    # 创建模型
    model = create_model(opt, device)
    
    # 创建损失函数
    rectified_flow = RectifiedFlow(opt)
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, opt)
    scheduler = create_scheduler(optimizer, opt)
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_dataloader(
        data_root=opt.data_root,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        train=True,
        opt=opt
    )
    
    cnt = 0
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"批次索引: {batch_idx}")
        print(f"批次数据: {batch_data}")
        batch_real = load(batch_data, model, device, opt)
        cnt = cnt + 1
        new_batch = get_batch_sample(batch_real, opt, model)
        ensure(model, new_batch, opt)
        break


if __name__ == "__main__":
    main()
