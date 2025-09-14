#!/usr/bin/env python3
"""
快速数据集测试脚本
简化版本，用于快速验证数据集加载是否正常工作
"""

import os
import sys
import torch
import traceback
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.dataset import FLOATDataset
from options.base_options import BaseOptions


def quick_test():
    """快速测试数据集加载"""
    print("🚀 快速数据集测试开始...")
    
    try:
        # 解析配置
        opt = BaseOptions().parse()
        print(f"📁 数据根目录: {opt.data_root}")
        
        # 检查数据目录是否存在
        if not os.path.exists(opt.data_root):
            print(f"❌ 数据根目录不存在: {opt.data_root}")
            print("💡 请确保数据目录存在，或使用 --data_root 参数指定正确的路径")
            return False
        
        # 创建数据集
        print("🔄 创建数据集...")
        dataset = FLOATDataset(
            data_root=opt.data_root,
            train=True,
            opt=opt
        )
        
        print(f"✅ 数据集创建成功！")
        print(f"📊 数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("⚠️  警告: 数据集为空")
            print("💡 请检查数据目录结构和文件是否正确")
            return False
        
        # 测试加载第一个数据项
        print("🔄 测试加载第一个数据项...")
        data_item = dataset[0]
        
        print("✅ 数据项加载成功！")
        print("📋 数据项包含的键:")
        for key, value in data_item.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {tuple(value.shape)} ({value.dtype})")
            else:
                print(f"   {key}: {type(value)} = {value}")
        
        # 简单的数据验证
        required_keys = ['video_cur', 'audio_latent_cur', 'emotion_features']
        missing_keys = [key for key in required_keys if key not in data_item]
        
        if missing_keys:
            print(f"⚠️  缺少必要的键: {missing_keys}")
        else:
            print("✅ 包含所有必要的数据键")
        
        # 检查数据类型
        tensor_keys = ['video_cur', 'motion_latent_cur', 'audio_latent_cur']
        for key in tensor_keys:
            if key in data_item and not isinstance(data_item[key], torch.Tensor):
                print(f"⚠️  {key} 不是 torch.Tensor 类型")
        
        print("\n🎉 快速测试完成！数据集基本功能正常。")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print("🔍 错误详情:")
        print(traceback.format_exc())
        return False


def test_with_custom_config():
    """使用自定义配置测试"""
    print("🔧 使用自定义配置测试...")
    
    # 创建一个最小配置
    class MinimalConfig:
        def __init__(self):
            # 必要的配置项
            self.input_size = 512
            self.dim_w = 512
            self.dim_m = 20
            self.dim_a = 512
            self.dim_e = 7
            self.wav2vec_sec = 0.64
            self.fps = 25
            self.num_prev_frames = 4
            self.sampling_rate = 16000
            self.wav2vec_model_path = "./checkpoints/wav2vec2-base-960h"
            self.audio2emotion_path = "./checkpoints/wav2vec-english-speech-emotion-recognition"
            self.only_last_features = True
            
            # 数据路径
            self.data_root = "/home/mli374/float/datasets"  # 默认路径
    
    try:
        opt = MinimalConfig()
        
        # 检查必要的模型文件
        model_files = [opt.wav2vec_model_path, opt.audio2emotion_path]
        missing_models = [f for f in model_files if not os.path.exists(f)]
        
        if missing_models:
            print("⚠️  缺少模型文件:")
            for f in missing_models:
                print(f"   {f}")
            print("💡 请确保模型文件存在，或下载相应的预训练模型")
        
        # 尝试创建数据集
        dataset = FLOATDataset(
            data_root=opt.data_root,
            train=True,
            opt=opt
        )
        
        print(f"✅ 使用自定义配置创建数据集成功！大小: {len(dataset)}")
        return True
        
    except Exception as e:
        print(f"❌ 自定义配置测试失败: {str(e)}")
        return False


def check_data_structure():
    """检查数据目录结构"""
    print("🔍 检查数据目录结构...")
    
    # 常见的数据目录结构
    possible_paths = [
        "/home/mli374/float/datasets",
        "./datasets",
        "../datasets",
        "./datasets/ravdess_processed",
        "./datasets/ravdess_raw"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到目录: {path}")
            
            # 列出子目录
            try:
                subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
                if subdirs:
                    print(f"   子目录: {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
                else:
                    print("   (无子目录)")
            except PermissionError:
                print("   (无法访问)")
        else:
            print(f"❌ 目录不存在: {path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='快速数据集测试')
    parser.add_argument('--data_root', type=str,
                       help='数据根目录')
    parser.add_argument('--check_structure', action='store_true',
                       help='检查数据目录结构')
    parser.add_argument('--custom_config', action='store_true',
                       help='使用自定义最小配置测试')

    args = parser.parse_args()

    success = True

    # 检查数据结构
    if args.check_structure:
        check_data_structure()
        return

    # 使用自定义配置测试
    if args.custom_config:
        success &= test_with_custom_config()

    # 如果指定了data_root，临时修改环境
    if args.data_root:
        # 修改命令行参数，让BaseOptions能够解析到
        sys.argv.extend(['--data_root', args.data_root])

    # 运行快速测试
    success &= quick_test()

    if success:
        print("\n🎉 所有测试通过！")
        print("💡 如需更详细的测试，请运行: python test_dataset.py")
        print("💡 所有配置都通过 BaseOptions 管理，无需 JSON 配置文件")
    else:
        print("\n💥 测试失败，请检查配置和数据文件")
        print("💡 常见问题:")
        print("   1. 数据目录不存在或为空")
        print("   2. 缺少预训练模型文件")
        print("   3. 数据格式不正确")
        print("   4. 依赖库未安装")
        print("💡 所有配置参数都在 options/base_options.py 中定义")


if __name__ == "__main__":
    main()
