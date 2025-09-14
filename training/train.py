"""
Simplified training script for FLOAT model with Rectified Flow
仿照 meanflow.py 的简洁风格重写的 FLOAT 训练脚本
"""

import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from tqdm import tqdm
from accelerate import Accelerator
import wandb
from models.float.FMT import FlowMatchingTransformer

# 添加项目路径

# from training.dataset import FLOATDataset  # 原始版本，已注释
from training.dataset_test import create_dataloader_optimized  # 优化版本
from training.rectified_flow import RectifiedFlow
from training.utils import set_seed
from options.base_options import BaseOptions


def cycle(iterable):
    """无限循环迭代器"""
    while True:
        for i in iterable:
            yield i


def build_argparser_from_base() -> argparse.Namespace:
    """使用 BaseOptions 并补充训练相关参数"""
    base = BaseOptions()
    parser = base.initialize(argparse.ArgumentParser(description="FLOAT 模型训练"))
    
    # 添加 wandb 相关参数
    parser.add_argument('--wandb_project', type=str, default='float-fmt',
                       help='wandb 项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='wandb 运行名称，如果为 None 则自动生成')
    parser.add_argument('--disable_wandb', action='store_true',
                       help='禁用 wandb 日志记录')

    # 添加数据集优化相关参数
    parser.add_argument('--force_preprocess', action='store_true',
                       help='强制重新预处理数据集（忽略缓存）')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='缓存目录路径，如果为 None 则使用 data_root/cache')
    
    # 使用修改后的 parser 解析参数，而不是 base.parse()
    opt = parser.parse_args()
    if not hasattr(opt, 'rank'):
        opt.rank = 0
    return opt


def main(opt: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 训练参数
    n_steps = opt.steps
    batch_size = opt.batch_size
    learning_rate = opt.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 修复路径问题 - 确保指向正确的 checkpoints 目录
    # 当前工作目录是 /home/mli374/float，所以 checkpoints 应该在项目根目录下
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到项目根目录
    
    if not os.path.isabs(opt.wav2vec_model_path):
        # 如果路径是相对路径，从项目根目录开始解析
        opt.wav2vec_model_path = os.path.join(project_root, opt.wav2vec_model_path.lstrip('./'))
    if not os.path.isabs(opt.audio2emotion_path):
        opt.audio2emotion_path = os.path.join(project_root, opt.audio2emotion_path.lstrip('./'))
    
    # print(f"项目根目录: {project_root}")
    # print(f"Wav2Vec2 模型路径: {opt.wav2vec_model_path}")
    # print(f"情感模型路径: {opt.audio2emotion_path}")
    
    # 验证路径是否存在
    if not os.path.exists(opt.wav2vec_model_path):
        raise FileNotFoundError(f"Wav2Vec2 模型路径不存在: {opt.wav2vec_model_path}")
    if not os.path.exists(opt.audio2emotion_path):
        raise FileNotFoundError(f"情感模型路径不存在: {opt.audio2emotion_path}")
    
    # 创建输出目录
    os.makedirs('checkpoints/float_fmt', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # 初始化 Accelerator
    accelerator = Accelerator(mixed_precision=opt.mixed_precision)
    
    # 初始化 wandb（只在主进程中初始化）
    if accelerator.is_main_process and not getattr(opt, 'disable_wandb', False):
        # 生成运行名称
        if getattr(opt, 'wandb_name', None) is None:
            wandb_name = f"float_fmt_{opt.steps}steps_{int(time.time())}"
        else:
            wandb_name = opt.wandb_name
            
        wandb.init(
            project=getattr(opt, 'wandb_project', 'float-fmt'),
            name=wandb_name,
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "steps": n_steps,
                "mixed_precision": opt.mixed_precision,
                "seed": getattr(opt, 'seed', 42),
                "data_root": opt.data_root,
                "input_size": opt.input_size,
                "dim_w": opt.dim_w,
                "dim_m": opt.dim_m,
                "dim_e": opt.dim_e,
                "fps": opt.fps,
                "wav2vec_sec": opt.wav2vec_sec,
                "num_prev_frames": opt.num_prev_frames,
                "wandb_project": getattr(opt, 'wandb_project', 'float-fmt'),
                "wandb_name": wandb_name,
                # 优化版本相关配置
                "dataset_version": "optimized",
                "force_preprocess": getattr(opt, 'force_preprocess', False),
                "cache_dir": getattr(opt, 'cache_dir', None),
            }
        )
    
    # 设置随机种子
    set_seed(opt.seed if hasattr(opt, 'seed') else 42)
    
    # 创建优化的数据集和数据加载器
    print("使用优化版本的数据集 (dataset_test.py)...")

    # 使用优化的数据加载器创建函数
    train_dataloader = create_dataloader_optimized(
        data_root=opt.data_root,
        batch_size=batch_size,
        num_workers=getattr(opt, 'num_workers', 0),
        train=True,
        opt=opt,
        cache_dir=getattr(opt, 'cache_dir', None),  # 可通过命令行参数指定缓存目录
        force_preprocess=getattr(opt, 'force_preprocess', False)  # 可通过命令行参数控制
    )

    # 获取数据集实例进行健壮性检查
    dataset = train_dataloader.dataset

    # 简单的健壮性检查，避免空数据集导致无限等待
    if len(dataset) == 0:
        raise ValueError(f"数据集为空: 请检查路径 {opt.data_root} 下是否存在有效数据")

    print(f"优化数据集加载完成，共 {len(dataset)} 个样本")
    
    # 创建模型
    model = FlowMatchingTransformer(opt).to(accelerator.device)
    
    # 注意：优化版本的数据集已经在预处理阶段完成了所有模型推理
    # 不需要在训练时加载和管理数据集中的模型组件
    print("优化版本数据集无需加载模型组件，所有计算已在预处理阶段完成")
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    # 创建损失函数
    # 为 RectifiedFlow 传入 opt（包含丢弃概率等，可通过命令行覆盖）
    rectified_flow = RectifiedFlow(opt)
    # 准备训练（优化版本不需要准备数据集，因为没有模型组件需要分布式处理）
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    # 在 prepare 之后包一层无限迭代器，保证 sampler 等加速器包装生效
    train_iter = cycle(train_dataloader)
    # 训练状态
    global_step = 0
    losses = 0.0
    
    log_step = 500
    sample_step = 2000
    save_step = 10000
    
    # print("开始训练 FLOAT 模型...")
    # print(f"设备: {accelerator.device}")
    # print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        
        for step in pbar:
            # 获取数据
            # print("获取数据")
            try:
                batch = next(train_iter)
            except StopIteration:
                # 理论上不会发生（因为有 cycle），但为了调试更安全
                # print("DataLoader 迭代结束，重新创建迭代器")
                train_iter = cycle(train_dataloader)
                batch = next(train_iter)
            # print("数据获取完成")
            # 准备数据
            # 兼容不同的数据字段命名
            def _get_first_available(d, keys):
                for k in keys:
                    if k in d:
                        return d[k]
                raise KeyError(f"找不到可用的键: {keys}")

            x1 = _get_first_available(batch, ['motion_latent', 'motion_latent_cur']).to(accelerator.device)
            # 处理条件张量的形状
            wa = _get_first_available(batch, ['audio_features', 'audio_latent_cur']).to(accelerator.device)
            wr = batch['reference_motion'].to(accelerator.device)
            we = batch['emotion_features'].to(accelerator.device)
            
            print("wa.shape: ", wa.shape)
            print("wr.shape: ", wr.shape)
            print("we.shape: ", we.shape)
            
            # 为 wr 和 we 添加序列维度，使其与 wa 兼容
            wr = wr.unsqueeze(1)  # (batch_size, motion_dim) -> (batch_size, 1, motion_dim)
            we = we.unsqueeze(1)  # (batch_size, emotion_dim) -> (batch_size, 1, emotion_dim)
            
            print("After unsqueeze - wr.shape: ", wr.shape)
            print("After unsqueeze - we.shape: ", we.shape)
            print("wa.shape[1] (seq_len): ", wa.shape[1])
            
            conditions = {
                'wa': wa,
                'wr': wr,
                'we': we,
            }

            
            
            # 前一帧条件（如果有）
            prev_conditions = None
            if 'prev_motion' in batch or 'motion_latent_prev' in batch:
                prev_conditions = {
                    'prev_x': _get_first_available(batch, ['prev_motion', 'motion_latent_prev']).to(accelerator.device),
                    'prev_wa': _get_first_available(batch, ['prev_audio', 'audio_latent_prev']).to(accelerator.device),
                }
            
            # print(x1, conditions, prev_conditions)
            
            # 计算损失
            loss_dict = rectified_flow.loss(model, x1, conditions, prev_conditions)
            loss = loss_dict['loss']
            
            # 反向传播
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            # 更新状态
            global_step += 1
            losses += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{losses / global_step:.6f}'
            })
            
            # 记录 wandb 日志
            if accelerator.is_main_process and not getattr(opt, 'disable_wandb', False):
                wandb.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": losses / global_step,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/global_step": global_step,
                }, step=global_step)
            
            # 记录日志
            if accelerator.is_main_process and global_step % opt.log_step == 0:
                current_time = time.asctime(time.localtime(time.time()))
                lr = optimizer.param_groups[0]['lr']
                
                log_message = (
                    f'{current_time}\n'
                    f'Global Step: {global_step}\n'
                    f'Loss: {losses / opt.log_step:.6f}\n'
                    f'Learning Rate: {lr:.6f}\n'
                    f'{"="*50}\n'
                )
                
                with open('training_log.txt', mode='a') as f:
                    f.write(log_message)
                
                losses = 0.0
            
            # 生成样本
            if global_step % opt.sample_step == 0 and accelerator.is_main_process:
                model.eval()
                with torch.no_grad():
                    try:
                        # 使用当前批次的条件进行采样
                        sample_conditions = {
                            'wa': conditions['wa'][:1],  # 取第一个样本
                            'wr': conditions['wr'][:1],
                            'we': conditions['we'][:1]
                        }
                        
                        # 生成样本（这里使用简化的采样方法）
                        # 实际应用中需要根据 FLOAT 模型的具体采样方法来实现
                        sample_path = f"samples/step_{global_step}.pt"
                        torch.save({
                            'conditions': sample_conditions,
                            'step': global_step
                        }, sample_path)
                        # print(f"样本已保存: {sample_path}")
                        
                    except Exception as e:
                        # print(f"采样失败: {e}")
                        pass
                
                model.train()
                accelerator.wait_for_everyone()
            
            # 保存检查点
            if global_step % opt.save_step == 0 and accelerator.is_main_process:
                model_module = model.module if hasattr(model, 'module') else model
                ckpt_path = f"checkpoints/step_{global_step}.pt"
                torch.save({
                    'model_state_dict': model_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'loss': loss.item()
                }, ckpt_path)
                
                # 记录检查点保存到 wandb
                if not getattr(opt, 'disable_wandb', False):
                    wandb.log({
                        "checkpoint/saved": True,
                        "checkpoint/step": global_step,
                        "checkpoint/loss": loss.item(),
                    }, step=global_step)
                # print(f"检查点已保存: {ckpt_path}")
    
    # 保存最终模型
    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, 'module') else model
        final_ckpt_path = f"checkpoints/final_step_{global_step}.pt"
        torch.save({
            'model_state_dict': model_module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'loss': loss.item()
        }, final_ckpt_path)
        
        # 记录最终模型保存到 wandb
        if not getattr(opt, 'disable_wandb', False):
            wandb.log({
                "final_model/saved": True,
                "final_model/step": global_step,
                "final_model/loss": loss.item(),
            }, step=global_step)
            
            # 完成 wandb 运行
            wandb.finish()
        # print(f"最终模型已保存: {final_ckpt_path}")


if __name__ == '__main__':
    # 从 BaseOptions 构建并解析参数
    args = build_argparser_from_base()
    main(args)
