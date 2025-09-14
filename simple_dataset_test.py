#!/usr/bin/env python3
"""
简单数据集测试脚本
专门针对 BaseOptions 配置系统，不依赖 JSON 配置文件
"""

import os
import sys
import torch
import traceback
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.dataset import FLOATDataset
from options.base_options import BaseOptions


def test_base_options():
    """测试 BaseOptions 配置解析"""
    print("🔍 测试 BaseOptions 配置解析...")
    
    try:
        opt = BaseOptions().parse()
        print("✅ BaseOptions 解析成功")
        
        # 显示关键配置
        key_configs = [
            'data_root', 'input_size', 'dim_w', 'dim_m', 'dim_a', 'dim_e',
            'fps', 'sampling_rate', 'wav2vec_sec', 'num_prev_frames',
            'wav2vec_model_path', 'audio2emotion_path'
        ]
        
        print("📋 关键配置参数:")
        for key in key_configs:
            if hasattr(opt, key):
                value = getattr(opt, key)
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: ❌ 未定义")
        
        return opt
        
    except Exception as e:
        print(f"❌ BaseOptions 解析失败: {e}")
        print(traceback.format_exc())
        return None


def check_paths(opt):
    """检查路径配置"""
    print("\n🔍 检查路径配置...")
    
    issues = []
    
    # 检查数据根目录
    if hasattr(opt, 'data_root'):
        if os.path.exists(opt.data_root):
            print(f"✅ 数据根目录存在: {opt.data_root}")
        else:
            issues.append(f"数据根目录不存在: {opt.data_root}")
    else:
        issues.append("配置中缺少 data_root")
    
    # 检查模型路径
    model_paths = [
        ('wav2vec_model_path', 'Wav2Vec2模型'),
        ('audio2emotion_path', '音频情感模型')
    ]
    
    for attr, desc in model_paths:
        if hasattr(opt, attr):
            path = getattr(opt, attr)
            if os.path.exists(path):
                print(f"✅ {desc}路径存在: {path}")
            else:
                issues.append(f"{desc}路径不存在: {path}")
        else:
            issues.append(f"配置中缺少 {attr}")
    
    if issues:
        print("❌ 发现问题:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ 所有路径检查通过")
        return True


def test_dataset_creation(opt):
    """测试数据集创建"""
    print("\n🔍 测试数据集创建...")
    
    try:
        dataset = FLOATDataset(
            data_root=opt.data_root,
            train=True,
            opt=opt
        )
        
        print(f"✅ 数据集创建成功")
        print(f"📊 数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("⚠️  警告: 数据集为空")
            return False, None
        
        return True, dataset
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        print("🔍 错误详情:")
        print(traceback.format_exc())
        return False, None


def test_data_loading(dataset):
    """测试单个数据加载"""
    print("\n🔍 测试单个数据加载...")

    try:
        # 加载第一个数据项
        data_item = dataset[0]

        print("✅ 数据项加载成功")
        print("📋 数据项结构:")

        for key, value in data_item.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {tuple(value.shape)} ({value.dtype})")
            else:
                print(f"   {key}: {type(value)} = {value}")

        # 检查必要的键
        required_keys = ['video_cur', 'audio_latent_cur', 'emotion_features']
        missing_keys = [key for key in required_keys if key not in data_item]

        if missing_keys:
            print(f"⚠️  缺少必要的键: {missing_keys}")
            return False
        else:
            print("✅ 包含所有必要的数据键")

        # 检查音频潜在表示的问题
        audio_cur = data_item.get('audio_latent_cur')
        audio_prev = data_item.get('audio_latent_prev')

        if audio_cur is not None and audio_cur.shape[0] == 0:
            print("⚠️  警告: audio_latent_cur 第一维为0，这是问题所在！")
            print(f"   实际形状: {audio_cur.shape}")
            print(f"   期望形状: (帧数, 512)")

        if audio_prev is not None and audio_prev.shape[0] == 0:
            print("⚠️  警告: audio_latent_prev 第一维为0，这是问题所在！")
            print(f"   实际形状: {audio_prev.shape}")
            print(f"   期望形状: (帧数, 512)")

        return True

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print(traceback.format_exc())
        return False


def test_batch_loading(dataset):
    """测试批量数据加载"""
    print("\n🔍 测试批量数据加载...")

    try:
        from torch.utils.data import DataLoader

        # 创建数据加载器，使用小批量
        batch_size = 2
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 避免多进程问题
            pin_memory=False,
            drop_last=False
        )

        print(f"📦 创建数据加载器成功，批量大小: {batch_size}")

        # 获取第一个批次
        batch_iter = iter(dataloader)
        batch = next(batch_iter)

        print("✅ 批量数据加载成功")
        print("📋 批量数据结构:")

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {tuple(value.shape)} ({value.dtype})")
            elif isinstance(value, (list, tuple)):
                print(f"   {key}: {type(value)} (长度: {len(value)})")
                if len(value) > 0:
                    print(f"      示例: {value[0]}")
            else:
                print(f"   {key}: {type(value)} = {value}")

        # 详细分析音频潜在表示
        if 'audio_latent_cur' in batch:
            audio_cur_batch = batch['audio_latent_cur']
            print(f"\n🔍 音频潜在表示详细分析:")
            print(f"   audio_latent_cur 批量形状: {audio_cur_batch.shape}")

            if audio_cur_batch.shape[1] == 0:
                print("   ❌ 问题确认: 批量中所有样本的 audio_latent_cur 第二维都为0")
                print("   这说明问题出现在单个样本的处理阶段")
            else:
                print("   ✅ 批量处理正常")

        if 'audio_latent_prev' in batch:
            audio_prev_batch = batch['audio_latent_prev']
            print(f"   audio_latent_prev 批量形状: {audio_prev_batch.shape}")

            if audio_prev_batch.shape[1] == 0:
                print("   ❌ 问题确认: 批量中所有样本的 audio_latent_prev 第二维都为0")
            else:
                print("   ✅ 批量处理正常")

        return True

    except Exception as e:
        print(f"❌ 批量数据加载失败: {e}")
        print("🔍 错误详情:")
        print(traceback.format_exc())
        return False


def main():
    print("🚀 简单数据集测试 (基于 BaseOptions)")
    print("=" * 60)
    
    # 步骤 1: 测试配置解析
    opt = test_base_options()
    if opt is None:
        print("\n💥 配置解析失败，无法继续测试")
        return
    
    # 步骤 2: 检查路径
    paths_ok = check_paths(opt)
    if not paths_ok:
        print("\n💥 路径检查失败")
        print("💡 解决方案:")
        print("   1. 使用 --data_root 参数指定正确的数据目录")
        print("   2. 确保预训练模型文件存在于指定路径")
        print("   3. 检查 options/base_options.py 中的默认路径设置")
        return
    
    # 步骤 3: 测试数据集创建
    dataset_ok, dataset = test_dataset_creation(opt)
    if not dataset_ok:
        print("\n💥 数据集创建失败")
        print("💡 解决方案:")
        print("   1. 检查数据目录结构是否正确")
        print("   2. 确保数据文件存在且格式正确")
        print("   3. 检查依赖库是否正确安装")
        return
    
    # 步骤 4: 测试单个数据加载
    loading_ok = test_data_loading(dataset)
    if not loading_ok:
        print("\n💥 单个数据加载失败")
        return

    # 步骤 5: 测试批量数据加载
    batch_ok = test_batch_loading(dataset)
    if not batch_ok:
        print("\n💥 批量数据加载失败")
        return

    # 成功总结
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！")
    print("📊 测试结果:")
    print("   ✅ BaseOptions 配置解析正常")
    print("   ✅ 路径配置正确")
    print("   ✅ 数据集创建成功")
    print("   ✅ 单个数据加载正常")
    print("   ✅ 批量数据加载正常")
    
    print(f"\n📋 数据集信息:")
    print(f"   数据集大小: {len(dataset)}")
    print(f"   数据根目录: {opt.data_root}")
    print(f"   输入尺寸: {opt.input_size}")
    print(f"   批次大小: {getattr(opt, 'batch_size', '未设置')}")
    
    print(f"\n💡 下一步:")
    print("   可以开始训练: python training/train.py")
    print("   或运行完整测试: python test_dataset.py")


if __name__ == "__main__":
    # 支持基本的命令行参数
    if len(sys.argv) > 1:
        print("💡 提示: 此脚本使用 BaseOptions 解析所有参数")
        print("💡 可用参数请查看: options/base_options.py")
        print("💡 示例: python simple_dataset_test.py --data_root /path/to/data")
        print()
    
    main()
