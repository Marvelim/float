"""
简化的 FLOAT 模型训练脚本 - 不使用 accelerate
基于 train.py 改写，移除了 accelerate 依赖，使用原生 PyTorch 训练
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.float.FMT import FlowMatchingTransformer

# 添加项目路径
from models.float.FLOAT import FLOAT
from training.dataset import FLOATDataset
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
    opt = base.parse()
    if not hasattr(opt, 'rank'):
        opt.rank = 0
    return opt


def main(opt: argparse.Namespace):
    # 训练参数
    n_steps = opt.steps
    batch_size = opt.batch_size
    learning_rate = opt.lr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    
    # 修复路径问题 - 确保指向正确的 checkpoints 目录
    # 当前工作目录是 /home/mli374/float，所以 checkpoints 应该在项目根目录下
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到项目根目录
    
    if not os.path.isabs(opt.wav2vec_model_path):
        # 如果路径是相对路径，从项目根目录开始解析
        opt.wav2vec_model_path = os.path.join(project_root, opt.wav2vec_model_path.lstrip('./'))
    if not os.path.isabs(opt.audio2emotion_path):
        opt.audio2emotion_path = os.path.join(project_root, opt.audio2emotion_path.lstrip('./'))
    
    print(f"项目根目录: {project_root}")
    print(f"Wav2Vec2 模型路径: {opt.wav2vec_model_path}")
    print(f"情感模型路径: {opt.audio2emotion_path}")
    
    # 验证路径是否存在
    if not os.path.exists(opt.wav2vec_model_path):
        raise FileNotFoundError(f"Wav2Vec2 模型路径不存在: {opt.wav2vec_model_path}")
    if not os.path.exists(opt.audio2emotion_path):
        raise FileNotFoundError(f"情感模型路径不存在: {opt.audio2emotion_path}")
    
    # 创建输出目录
    os.makedirs('checkpoints/float_fmt', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # 设置随机种子
    set_seed(opt.seed if hasattr(opt, 'seed') else 42)
    
    # 创建数据集（直接使用 BaseOptions 解析得到的 opt）
    dataset = FLOATDataset(
        data_root=opt.data_root,
        train=True,
        opt=opt,
    )
    
    # 简单的健壮性检查，避免空数据集导致无限等待
    if len(dataset) == 0:
        raise ValueError(f"数据集为空: 请检查路径 {opt.data_root} 下是否存在有效数据")

    # 使用可配置的 num_workers，但为了避免多进程死锁，建议设置为 0
    num_workers = getattr(opt, 'num_workers', 0)
    if num_workers > 0:
        print(f"警告: 使用 num_workers={num_workers} 可能导致多进程死锁，建议设置为 0")
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,  # 设置为 False，因为数据已经在 CUDA 设备上
        persistent_workers=(num_workers > 0),
    )
    
    # 创建模型并移动到设备
    model = FlowMatchingTransformer(opt).to(device)
    
    # 将数据集中的模型组件移动到 CUDA 设备
    print("将数据集模型组件移动到 CUDA 设备...")
    dataset.motion_autoencoder = dataset.motion_autoencoder.to(device)
    dataset.audio_encoder = dataset.audio_encoder.to(device)
    
    # 确保 audio_encoder 中的所有子模块都被移动到 CUDA
    if hasattr(dataset.audio_encoder, 'wav2vec2'):
        dataset.audio_encoder.wav2vec2 = dataset.audio_encoder.wav2vec2.to(device)
    if hasattr(dataset.audio_encoder, 'audio_projection'):
        dataset.audio_encoder.audio_projection = dataset.audio_encoder.audio_projection.to(device)
    
    print("数据集模型组件已移动到 CUDA 设备")
    
    # 加载预训练权重到数据集中的模型组件
    checkpoint_path = "/home/mli374/float/checkpoints/float.pth"
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # 加载预训练权重到数据集中的模型组件
    with torch.no_grad():
        # 加载 motion_autoencoder 权重
        for name, param in dataset.motion_autoencoder.named_parameters():
            full_name = f"motion_autoencoder.{name}"
            if full_name in state_dict:
                param.copy_(state_dict[full_name])
                print(f"Loaded {name} from motion_autoencoder weights")
            else:
                print(f"! Warning: {name} not found in motion_autoencoder weights")
    
    print("预训练权重已加载到数据集模型组件")
    
    # 只训练 FlowMatchingTransformer，冻结其他组件
    for param in dataset.motion_autoencoder.parameters():
        param.requires_grad = False
    
    print("数据集模型组件已冻结，只训练 FlowMatchingTransformer")
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    
    # 创建损失函数
    # 为 RectifiedFlow 传入 opt（包含丢弃概率等，可通过命令行覆盖）
    rectified_flow = RectifiedFlow(opt)
    
    # 创建无限迭代器
    train_iter = cycle(train_dataloader)
    
    # 训练状态
    global_step = 0
    losses = 0.0
    
    log_step = getattr(opt, 'log_step', 500)
    sample_step = getattr(opt, 'sample_step', 2000)
    save_step = getattr(opt, 'save_step', 10000)
    
    print("开始训练 FLOAT 模型...")
    print(f"设备: {device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    with tqdm(range(n_steps), dynamic_ncols=True) as pbar:
        pbar.set_description("Training")
        model.train()
        
        for step in pbar:
            # 获取数据
            print("获取数据")
            try:
                batch = next(train_iter)
            except StopIteration:
                # 理论上不会发生（因为有 cycle），但为了调试更安全
                print("DataLoader 迭代结束，重新创建迭代器")
                train_iter = cycle(train_dataloader)
                batch = next(train_iter)
            print("数据获取完成")
            
            # 准备数据
            # 兼容不同的数据字段命名
            def _get_first_available(d, keys):
                for k in keys:
                    if k in d:
                        return d[k]
                raise KeyError(f"找不到可用的键: {keys}")

            x1 = _get_first_available(batch, ['motion_latent', 'motion_latent_cur']).to(device)
            conditions = {
                'wa': _get_first_available(batch, ['audio_features', 'audio_latent_cur']).to(device),
                'wr': batch['reference_motion'].to(device),
                'we': batch['emotion_features'].to(device),
            }
            
            # 前一帧条件（如果有）
            prev_conditions = None
            if 'prev_motion' in batch or 'motion_latent_prev' in batch:
                prev_conditions = {
                    'prev_x': _get_first_available(batch, ['prev_motion', 'motion_latent_prev']).to(device),
                    'prev_wa': _get_first_available(batch, ['prev_audio', 'audio_latent_prev']).to(device),
                }
            
            print(x1, conditions, prev_conditions)
            
            # 计算损失
            loss_dict = rectified_flow.loss(model.fmt, x1, conditions, prev_conditions)
            loss = loss_dict['loss']
            
            # 反向传播 - 使用原生 PyTorch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新状态
            global_step += 1
            losses += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{losses / global_step:.6f}'
            })
            
            # 记录日志
            if global_step % log_step == 0:
                current_time = time.asctime(time.localtime(time.time()))
                lr = optimizer.param_groups[0]['lr']
                
                log_message = (
                    f'{current_time}\n'
                    f'Global Step: {global_step}\n'
                    f'Loss: {losses / log_step:.6f}\n'
                    f'Learning Rate: {lr:.6f}\n'
                    f'{"="*50}\n'
                )
                
                with open('training_log.txt', mode='a') as f:
                    f.write(log_message)
                
                losses = 0.0
            
            # 生成样本
            if global_step % sample_step == 0:
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
                        print(f"样本已保存: {sample_path}")
                        
                    except Exception as e:
                        print(f"采样失败: {e}")
                
                model.train()
            
            # 保存检查点
            if global_step % save_step == 0:
                ckpt_path = f"checkpoints/step_{global_step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'loss': loss.item()
                }, ckpt_path)
                print(f"检查点已保存: {ckpt_path}")
    
    # 保存最终模型
    final_ckpt_path = f"checkpoints/final_step_{global_step}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'loss': loss.item()
    }, final_ckpt_path)
    print(f"最终模型已保存: {final_ckpt_path}")


if __name__ == '__main__':
    # 从 BaseOptions 构建并解析参数
    args = build_argparser_from_base()
    main(args)
