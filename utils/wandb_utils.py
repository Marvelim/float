#!/usr/bin/env python3
"""
Wandb 工具函数
Wandb utility functions for FLOAT training
"""

import os
import time

# wandb 导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb 未安装，将跳过 wandb 日志记录")


def init_wandb(opt):
    """初始化 wandb"""
    if not WANDB_AVAILABLE or not opt.use_wandb:
        return False
    
    # 生成运行名称
    run_name = opt.wandb_run_name
    if run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"float_training_{timestamp}"
    
    # 初始化 wandb
    wandb.init(
        project=opt.wandb_project,
        entity=opt.wandb_entity,
        name=run_name,
        tags=opt.wandb_tags,
        config=vars(opt)
    )
    
    print(f"wandb 初始化完成: {wandb.run.url}")
    return True


def log_to_wandb(step, loss_dict, lr, elapsed_time, opt):
    """记录训练指标到 wandb"""
    if not WANDB_AVAILABLE or not opt.use_wandb or not wandb.run:
        return
    
    # 记录基本指标
    wandb.log({
        "step": step,
        "loss/total": loss_dict['loss'].item(),
        "loss/prev": loss_dict['loss_prev'].item(),
        "loss/current": loss_dict['loss_current'].item(),
        "learning_rate": lr,
        "time/elapsed": elapsed_time,
        "time/step": elapsed_time / opt.log_step
    }, step=step)


def log_sample_to_wandb(step, video_path, opt):
    """记录样本视频到 wandb"""
    if not WANDB_AVAILABLE or not opt.use_wandb or not wandb.run:
        return
    
    if video_path and os.path.exists(video_path):
        wandb.log({
            "samples/video": wandb.Video(video_path, fps=opt.fps, format="mp4")
        }, step=step)


def finish_wandb():
    """完成 wandb 运行"""
    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()
