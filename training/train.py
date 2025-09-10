"""
Main training script for FLOAT model with Rectified Flow
使用 Rectified Flow 的 FLOAT 模型主训练脚本
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.float.FLOAT import FLOAT
from training.dataset import create_dataloader, collate_fn
from training.rectified_flow import RectifiedFlow, FlowMatchingTrainer
from training.utils import (
    setup_logging, set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, get_lr_scheduler, setup_wandb
)


class FLOATTrainer:
    """FLOAT 训练器主类"""
    
    def __init__(self, opt: dict):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.opt = opt
        self.device = torch.device(opt.device)
        self.rank = opt.rank
        
        # 设置随机种子
        set_seed(opt.seed)
        
        # 设置日志
        self.logger = setup_logging(
            log_level=logging.INFO,
            log_file=os.path.join(opt.output_dir, "train.log")
        )
        
        # 初始化模型
        self.model = self._build_model()
        
        # 初始化损失函数
        self.loss_fn = RectifiedFlow.loss (
            sigma_min=opt.training.sigma_min,
            sigma_max=opt.training.sigma_max,
            loss_type=opt.training.loss_type
        )
        
        # 初始化优化器
        self.optimizer = self._build_optimizer()
        
        # 初始化学习率调度器
        self.lr_scheduler = self._build_lr_scheduler(opt)
        
        # 初始化数据加载器
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # 初始化混合精度训练
        self.scaler = GradScaler() if opt.training.use_amp else None
        # 初始化训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # 初始化指标记录
        self.train_metrics = AverageMeter()
        self.val_metrics = AverageMeter()
        
        # 初始化 wandb（如果启用）
        if opt.training.use_wandb:
            setup_wandb(config)
        
        self.logger.info(f"初始化完成，使用设备: {self.device}")
        self.logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _build_model(self) -> nn.Module:
        """构建模型"""
        # 创建一个模拟的 opt 对象来匹配 FLOAT 的接口
        class OptMock:
            def __init__(self, opt):
                # 从 config 复制所有模型参数
                for key, value in opt.model.__dict__.items():
                    setattr(self, key, value)
                
                # 添加必要的设备信息
                self.rank = opt.rank
        
        opt = OptMock(self.opt)
        model = FLOAT(opt)
        model = model.to(self.device)
        
        # 如果使用分布式训练
        if self.opt.training.use_ddp and torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器"""
        opt = self.opt
        if opt.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt.learning_rate,
                betas=(opt.beta1, opt.beta2),
                eps=opt.eps,
                weight_decay=opt.weight_decay
            )
        elif opt.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt.learning_rate,
                betas=(opt.beta1, opt.beta2),
                eps=opt.eps,
                weight_decay=opt.weight_decay
            )
        elif opt.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt.learning_rate,
                momentum=0.9,
                weight_decay=opt.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {opt.optimizer}")
        
        return optimizer
    
    def _build_lr_scheduler(self):
        """构建学习率调度器"""
        return get_lr_scheduler(
            self.optimizer,
            scheduler_type=self.opt.lr_scheduler,
            num_epochs=self.opt.num_epochs,
            warmup_epochs=self.opt.lr_warmup_epochs,
            min_lr=self.opt.lr_min
        )
    
    def _build_dataloaders(self) -> tuple:
        """构建数据加载器"""
        # 训练数据加载器
        train_loader = create_dataloader(
            data_root=self.config.data.data_root,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            train=True,
            video_fps=self.config.data.video_fps,
            audio_sample_rate=self.config.data.audio_sample_rate,
            wav2vec_sec=self.config.data.wav2vec_sec,
            sequence_length=self.config.data.sequence_length,
            prev_frames=self.config.data.prev_frames,
            image_size=self.config.data.image_size,
            motion_dim=self.config.data.motion_dim,
            audio_dim=self.config.data.audio_dim,
            emotion_dim=self.config.data.emotion_dim
        )
        
        # 验证数据加载器
        val_loader = create_dataloader(
            data_root=self.config.data.data_root,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.data.num_workers,
            train=False,
            video_fps=self.config.data.video_fps,
            audio_sample_rate=self.config.data.audio_sample_rate,
            wav2vec_sec=self.config.data.wav2vec_sec,
            sequence_length=self.config.data.sequence_length,
            prev_frames=self.config.data.prev_frames,
            image_size=self.config.data.image_size,
            motion_dim=self.config.data.motion_dim,
            audio_dim=self.config.data.audio_dim,
            emotion_dim=self.config.data.emotion_dim
        )
        
        return train_loader, val_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            step_start_time = time.time()
            
            # 训练步骤
            metrics = self.train_step(batch)
            self.train_metrics.update(metrics)
            
            # 更新全局步数
            self.global_step += 1
            
            # 记录日志
            if self.global_step % self.config.training.log_interval == 0:
                step_time = time.time() - step_start_time
                self.logger.info(
                    f"Epoch [{self.current_epoch}/{self.config.training.num_epochs}] "
                    f"Step [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {metrics['loss']:.6f} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} "
                    f"Time: {step_time:.3f}s"
                )
                
                # 记录到 wandb
                if self.config.training.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/grad_norm': metrics.get('grad_norm', 0),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/step_time': step_time,
                        'global_step': self.global_step
                    })
            
            # 验证
            if self.global_step % self.config.training.val_check_interval == 0:
                val_metrics = self.validate()
                self.logger.info(f"验证损失: {val_metrics['val_loss']:.6f}")
                
                # 记录到 wandb
                if self.config.training.use_wandb:
                    import wandb
                    wandb.log({
                        'val/loss': val_metrics['val_loss'],
                        'global_step': self.global_step
                    })
                
                # 早停检查
                if self._check_early_stopping(val_metrics['val_loss']):
                    self.logger.info("早停触发，停止训练")
                    return self.train_metrics.avg
            
            # 保存检查点
            if self.global_step % self.config.training.save_checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pth")
        
        epoch_time = time.time() - epoch_start_time
        self.logger.info(f"Epoch {self.current_epoch} 完成，耗时: {epoch_time:.2f}s")
        
        return self.train_metrics.avg
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行一个训练步骤"""
        # 准备数据
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        
        # 前向传播
        if self.config.training.use_amp:
            with autocast(dtype=torch.float16 if self.config.training.amp_dtype == "float16" else torch.bfloat16):
                loss_dict = self._compute_loss(batch)
                loss = loss_dict['loss']
        else:
            loss_dict = self._compute_loss(batch)
            loss = loss_dict['loss']
        
        # 梯度累积
        loss = loss / self.config.training.gradient_accumulation_steps
        
        # 反向传播
        if self.config.training.use_amp:
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.training.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                else:
                    grad_norm = 0
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            
            if (self.global_step + 1) % self.config.training.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.config.training.gradient_clip_val > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip_val
                    )
                else:
                    grad_norm = 0
                
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return {
            'loss': loss.item() * self.config.training.gradient_accumulation_steps,
            'grad_norm': grad_norm if 'grad_norm' in locals() else 0
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        x1 = batch['motion_latent']  # 目标动作序列
        
        conditions = {
            'wa': batch['audio_features'],
            'wr': batch['reference_motion'],
            'we': batch['emotion_features']
        }
        
        # 准备前一帧条件
        prev_conditions = None
        if 'prev_motion' in batch:
            prev_conditions = {
                'prev_x': batch['prev_motion'],
                'prev_wa': batch['prev_audio']
            }
        
        # 使用 FMT 模型计算损失
        loss_dict = self.loss_fn(self.model.fmt, x1, conditions, prev_conditions)
        
        return loss_dict
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in self.val_loader:
            # 准备数据
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)
            
            # 前向传播
            if self.config.training.use_amp:
                with autocast(dtype=torch.float16 if self.config.training.amp_dtype == "float16" else torch.bfloat16):
                    loss_dict = self._compute_loss(batch)
            else:
                loss_dict = self._compute_loss(batch)
            
            metrics = {'val_loss': loss_dict['loss'].item()}
            self.val_metrics.update(metrics)
        
        return self.val_metrics.avg
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """检查早停条件"""
        if val_loss < self.best_val_loss - self.config.training.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.training.early_stopping_patience
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            config=self.config,
            filepath=checkpoint_path
        )
        self.logger.info(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        # 加载模型状态
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载学习率调度器状态
        if 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        # 加载训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"检查点已加载: {checkpoint_path}")
        self.logger.info(f"继续训练从 epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        
        # 如果有检查点，加载它
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # 保存配置
        config_path = os.path.join(self.config.output_dir, "config.json")
        self.config.save(config_path)
        
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # 训练一个 epoch
                train_metrics = self.train_epoch()
                
                # 更新学习率
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                
                # 验证
                val_metrics = self.validate()
                
                # 记录 epoch 级别的指标
                self.logger.info(
                    f"Epoch {epoch} 完成 - "
                    f"训练损失: {train_metrics['loss']:.6f}, "
                    f"验证损失: {val_metrics['val_loss']:.6f}"
                )
                
                # 保存最佳模型
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint("best_model.pth")
                
                # 保存定期检查点
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
                
                # 早停检查
                if self._check_early_stopping(val_metrics['val_loss']):
                    self.logger.info("早停触发，停止训练")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
            self.save_checkpoint("interrupted_checkpoint.pth")
        
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            self.save_checkpoint("error_checkpoint.pth")
            raise
        
        finally:
            self.logger.info("训练完成")
            if self.config.training.use_wandb:
                import wandb
                wandb.finish()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FLOAT 模型训练")
    parser.add_argument("--config", type=str, default="default", help="配置名称")
    parser.add_argument("--config-file", type=str, help="配置文件路径")
    parser.add_argument("--output-dir", type=str, help="输出目录")
    parser.add_argument("--resume", type=str, help="从检查点恢复训练")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    parser.add_argument("--learning-rate", type=float, help="学习率")
    parser.add_argument("--num-epochs", type=int, help="训练轮数")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config_file:
        config = Config.load(args.config_file)
    else:
        config = get_config(args.config)
    
    # 从命令行参数更新配置
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.device != "auto":
        config.device = args.device
    
    # 创建训练器并开始训练
    trainer = FLOATTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
