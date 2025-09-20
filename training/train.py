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
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 导入工具函数
from utils.wandb_utils import init_wandb, log_to_wandb, log_sample_to_wandb, finish_wandb
from utils.memory_utils import clear_memory, clear_cache
from utils.device_utils import setup_device
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, load_weight
from utils.data_utils import load, get_batch_sample, get_audio_preprocessor
from utils.video_utils import ensure, generate_sample, save_generated_video, save_data_out_as_video
from utils.model_utils import create_model, create_optimizer, create_scheduler

from options.base_options import BaseOptions
from training.dataset import create_dataloader
from training.rectified_flow import RectifiedFlow

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'


def train_step(model, rectified_flow, batch_data, optimizer, device, accumulation_steps=1):
    """执行一个训练步骤（支持梯度累积）"""
    # 将数据移动到设备
    for key in batch_data:
        if isinstance(batch_data[key], torch.Tensor):
            batch_data[key] = batch_data[key].to(device)
    
    
    # 准备条件数据
    conditions = {
        'wa': batch_data['audio_latent_cur'],  # 当前音频特征
        'wr': batch_data['reference_motion'],  # 参考运动
        'we': batch_data['emotion_features']   # 情感特征
    }
    
    # 准备前一帧条件
    prev_conditions = {
        'prev_x': batch_data['motion_latent_prev'],  # 前一帧运动
        'prev_wa': batch_data['audio_latent_prev']   # 前一帧音频
    }
    
    # 目标数据（当前帧运动）
    x1 = batch_data['motion_latent_cur']
    
    # 计算损失
    loss_dict = rectified_flow.loss(
        model=model.fmt,
        x1=x1,
        conditions=conditions,
        prev_conditions=prev_conditions
    )
    
    loss = loss_dict['loss'] / accumulation_steps  # 缩放损失
    
    # 反向传播
    loss.backward()
    
    return loss_dict


def train(model, dataloader, rectified_flow, optimizer, scheduler, opt, device, start_step=0):
    """主训练循环"""
    print("开始训练...")
    
    # 初始化 wandb
    wandb_initialized = init_wandb(opt)
    
    # 创建输出目录
    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    # 训练循环
    model.train()
    step = start_step
    
    # 用于记录训练统计
    running_loss = 0.0
    running_loss_prev = 0.0
    running_loss_current = 0.0
    
    start_time = time.time()
    
    while step < opt.steps:
        for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Data Loader"):
            if step >= opt.steps:
                break
            
            real_data = load(batch_data, model, device, opt)
            for j in tqdm(range(opt.batch_step), total=opt.batch_step, desc="Batch Step"):
                # print("debug: batch_idx = {}, j = {}".format(batch_idx, j))
                new_batch = get_batch_sample(real_data, opt, model)

                # ensure(model, new_batch, opt)
                # 训练步骤
                loss_dict = train_step(model, rectified_flow, new_batch, optimizer, device)
                
                # 更新学习率
                scheduler.step()
                
                # 记录统计信息
                running_loss += loss_dict['loss'].item()
                running_loss_prev += loss_dict['loss_prev'].item()
                running_loss_current += loss_dict['loss_current'].item()
                
                # 定期清理内存
                if step % 2 == 0:
                    clear_memory()
                
                step += 1
                
                # 打印训练信息
                if step % opt.log_step == 0:
                    elapsed_time = time.time() - start_time
                    avg_loss = running_loss / opt.log_step
                    avg_loss_prev = running_loss_prev / opt.log_step
                    avg_loss_current = running_loss_current / opt.log_step
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # 准备损失字典用于 wandb 记录
                    loss_dict_for_log = {
                        'loss': torch.tensor(avg_loss),
                        'loss_prev': torch.tensor(avg_loss_prev),
                        'loss_current': torch.tensor(avg_loss_current)
                    }
                    
                    # 记录到 wandb
                    log_to_wandb(step, loss_dict_for_log, current_lr, elapsed_time, opt)
                    
                    print(f"步骤 {step:6d}/{opt.steps} | "
                        f"损失: {avg_loss:.6f} | "
                        f"前一帧损失: {avg_loss_prev:.6f} | "
                        f"当前帧损失: {avg_loss_current:.6f} | "
                        f"学习率: {current_lr:.2e} | "
                        f"时间: {elapsed_time:.1f}s")
                    
                    # 重置统计
                    running_loss = 0.0
                    running_loss_prev = 0.0
                    running_loss_current = 0.0
                    start_time = time.time()
                
                # 生成样本
                if step % opt.sample_step == 0:
                    print(f"生成样本 (步骤 {step})...")
                    sample_video_path = generate_sample(model, device, opt, step)
                    # 记录样本到 wandb
                    if sample_video_path:
                        log_sample_to_wandb(step, sample_video_path, opt)
                    # 清理内存
                    clear_memory()
                
                # 保存检查点
                if step % opt.save_step == 0:
                    checkpoint_path = output_dir / f"checkpoint_step_{step}.pt"
                    save_checkpoint(model, optimizer, scheduler, step, loss_dict['loss'].item(), checkpoint_path)
    
    # 保存最终模型
    final_path = output_dir / f"final_step_{step}.pt"
    save_checkpoint(model, optimizer, scheduler, step, loss_dict['loss'].item(), final_path)
    
    # 完成 wandb 运行
    if wandb_initialized:
        finish_wandb()
    
    print("训练完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FLOAT 训练脚本')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--data-root', type=str, default='../datasets/ravdess_processed', help='数据根目录')
    parser.add_argument('--batch-size', type=int, default=1, help='批次大小')
    parser.add_argument('--steps', type=int, default=200000, help='训练步数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=1, help='数据加载器工作进程数')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='梯度累积步数')
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
    opt.gradient_accumulation_steps = args.gradient_accumulation_steps
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
    print(f"数据加载器创建完成，批次数量: {len(dataloader)}")
    
    # 恢复训练（如果指定）
    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(model, optimizer, scheduler, args.resume, device)
    
    # 开始训练
    train(model, dataloader, rectified_flow, optimizer, scheduler, opt, device, start_step)


if __name__ == "__main__":
    main()
