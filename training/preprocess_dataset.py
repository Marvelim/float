#!/usr/bin/env python3
"""
FLOAT 数据集预处理脚本
将数据集预处理从训练脚本中分离出来，方便独立运行
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from training.dataset_test import FLOATDatasetOptimized, clear_cache, check_cache_status
from options.base_options import BaseOptions


def build_argparser() -> argparse.Namespace:
    """构建参数解析器"""
    base = BaseOptions()
    parser = base.initialize(argparse.ArgumentParser(description="FLOAT 数据集预处理"))
    
    # 添加预处理相关参数
    parser.add_argument('--force_preprocess', action='store_true',
                       help='强制重新预处理数据集（忽略缓存）')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='缓存目录路径，如果为 None 则使用 data_root/cache')
    parser.add_argument('--clear_cache_first', action='store_true',
                       help='预处理前先清除现有缓存')
    parser.add_argument('--train_only', action='store_true',
                       help='只预处理训练集')
    parser.add_argument('--test_only', action='store_true',
                       help='只预处理测试集')
    parser.add_argument('--check_status', action='store_true',
                       help='只检查缓存状态，不进行预处理')
    
    opt = parser.parse_args()
    if not hasattr(opt, 'rank'):
        opt.rank = 0
    return opt


def clear_all_caches(opt):
    """清除所有相关的缓存"""
    print("=" * 60)
    print("清除缓存")
    print("=" * 60)
    
    # 清除数据集缓存
    print("清除数据集缓存...")
    clear_cache(opt.data_root, opt.cache_dir)
    
    # 清除 Python 缓存
    print("清除 Python 缓存...")
    import subprocess
    try:
        subprocess.run(['find', '.', '-name', '__pycache__', '-type', 'd', '-exec', 'rm', '-rf', '{}', '+'], 
                      cwd='/home/mli374/float', check=False)
        subprocess.run(['find', '.', '-name', '*.pyc', '-delete'], 
                      cwd='/home/mli374/float', check=False)
        print("Python 缓存清除完成")
    except Exception as e:
        print(f"清除 Python 缓存时出错: {e}")
    
    # 清除 CUDA 缓存
    if torch.cuda.is_available():
        print("清除 CUDA 缓存...")
        torch.cuda.empty_cache()
        print("CUDA 缓存清除完成")
    
    print("所有缓存清除完成")


def preprocess_dataset(opt):
    """预处理数据集"""
    print("=" * 60)
    print("开始数据集预处理")
    print("=" * 60)
    
    # 验证路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isabs(opt.wav2vec_model_path):
        opt.wav2vec_model_path = os.path.join(project_root, opt.wav2vec_model_path.lstrip('./'))
    if not os.path.isabs(opt.audio2emotion_path):
        opt.audio2emotion_path = os.path.join(project_root, opt.audio2emotion_path.lstrip('./'))
    
    print(f"项目根目录: {project_root}")
    print(f"数据根目录: {opt.data_root}")
    print(f"Wav2Vec2 模型路径: {opt.wav2vec_model_path}")
    print(f"情感模型路径: {opt.audio2emotion_path}")
    
    # 验证路径是否存在
    if not os.path.exists(opt.wav2vec_model_path):
        raise FileNotFoundError(f"Wav2Vec2 模型路径不存在: {opt.wav2vec_model_path}")
    if not os.path.exists(opt.audio2emotion_path):
        raise FileNotFoundError(f"情感模型路径不存在: {opt.audio2emotion_path}")
    if not os.path.exists(opt.data_root):
        raise FileNotFoundError(f"数据根目录不存在: {opt.data_root}")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    start_time = time.time()
    
    # 预处理训练集
    if not opt.test_only:
        print("\n" + "=" * 40)
        print("预处理训练集")
        print("=" * 40)
        
        train_start = time.time()
        train_dataset = FLOATDatasetOptimized(
            data_root=opt.data_root,
            train=True,
            opt=opt,
            cache_dir=opt.cache_dir,
            force_preprocess=opt.force_preprocess
        )
        train_end = time.time()
        
        print(f"训练集预处理完成:")
        print(f"  - 样本数量: {len(train_dataset)}")
        print(f"  - 耗时: {train_end - train_start:.2f} 秒")
    
    # 预处理测试集
    if not opt.train_only:
        print("\n" + "=" * 40)
        print("预处理测试集")
        print("=" * 40)
        
        test_start = time.time()
        test_dataset = FLOATDatasetOptimized(
            data_root=opt.data_root,
            train=False,
            opt=opt,
            cache_dir=opt.cache_dir,
            force_preprocess=opt.force_preprocess
        )
        test_end = time.time()
        
        print(f"测试集预处理完成:")
        print(f"  - 样本数量: {len(test_dataset)}")
        print(f"  - 耗时: {test_end - test_start:.2f} 秒")
    
    total_time = time.time() - start_time
    print(f"\n总预处理时间: {total_time:.2f} 秒")
    
    # 显示最终缓存状态
    print("\n" + "=" * 40)
    print("预处理完成后的缓存状态")
    print("=" * 40)
    check_cache_status(opt.data_root, opt.cache_dir)


def main():
    """主函数"""
    opt = build_argparser()
    
    print("FLOAT 数据集预处理工具")
    print("=" * 60)
    print(f"数据根目录: {opt.data_root}")
    print(f"缓存目录: {opt.cache_dir or f'{opt.data_root}/cache'}")
    print(f"强制重新预处理: {opt.force_preprocess}")
    print(f"预处理前清除缓存: {opt.clear_cache_first}")
    print("=" * 60)
    
    try:
        # 检查缓存状态
        if opt.check_status:
            print("检查缓存状态...")
            check_cache_status(opt.data_root, opt.cache_dir)
            return
        
        # 清除缓存（如果需要）
        if opt.clear_cache_first:
            clear_all_caches(opt)
            print()
        
        # 预处理数据集
        preprocess_dataset(opt)
        
        print("\n" + "=" * 60)
        print("预处理完成！现在可以运行训练脚本了。")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n用户中断预处理")
    except Exception as e:
        print(f"\n预处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
