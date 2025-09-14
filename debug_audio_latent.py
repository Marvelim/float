#!/usr/bin/env python3
"""
音频潜在表示调试脚本
专门调试 audio_latent 维度问题
"""

import os
import sys
import torch
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.dataset import FLOATDataset
from options.base_options import BaseOptions
from models.float.FLOAT import AudioEncoder


def debug_audio_processing():
    """调试音频处理流程"""
    print("🔍 调试音频处理流程...")
    
    try:
        # 获取配置
        opt = BaseOptions().parse()
        print(f"配置信息:")
        print(f"  wav2vec_sec: {opt.wav2vec_sec}")
        print(f"  fps: {opt.fps}")
        print(f"  sampling_rate: {opt.sampling_rate}")
        print(f"  num_prev_frames: {opt.num_prev_frames}")
        print(f"  dim_a: {opt.dim_a}")
        
        # 计算预期的序列长度
        expected_seq_len = int(opt.wav2vec_sec * opt.fps)
        print(f"  预期序列长度: {expected_seq_len}")
        
        # 创建数据集
        dataset = FLOATDataset(
            data_root=opt.data_root,
            train=True,
            opt=opt
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return
        
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 获取第一个数据项的路径信息
        data_item = dataset.data_list[0]
        print(f"\n📁 测试数据项:")
        print(f"  视频路径: {data_item['video_path']}")
        print(f"  音频路径: {data_item['audio_path']}")
        
        # 手动测试音频加载
        print(f"\n🎵 手动测试音频加载...")
        audio_tensor = dataset._load_audio(data_item['audio_path'])
        print(f"  原始音频张量形状: {audio_tensor.shape}")
        
        # 手动测试视频加载
        print(f"\n🎬 手动测试视频加载...")
        video_frames = dataset._load_video(data_item['video_path'])
        print(f"  视频帧形状: {video_frames.shape}")
        num_frames = video_frames.shape[0]
        print(f"  视频帧数: {num_frames}")
        
        # 手动测试音频编码器
        print(f"\n🔊 手动测试音频编码器...")
        audio_encoder = AudioEncoder(opt)
        
        print(f"  音频编码器配置:")
        print(f"    num_frames_for_clip: {audio_encoder.num_frames_for_clip}")
        print(f"    num_prev_frames: {audio_encoder.num_prev_frames}")
        print(f"    only_last_features: {audio_encoder.only_last_features}")
        
        # 测试音频编码器推理
        print(f"\n🧠 测试音频编码器推理...")
        print(f"  输入音频形状: {audio_tensor.shape}")
        print(f"  目标序列长度: {num_frames}")
        
        w_audio = audio_encoder.inference(audio_tensor, seq_len=num_frames)
        print(f"  输出音频特征形状: {w_audio.shape}")
        
        # 分析问题
        print(f"\n🔍 问题分析:")
        if w_audio.shape[0] == 0:
            print("  ❌ 第一维为0，说明没有生成任何特征")
            print("  可能原因:")
            print("    1. 音频文件为空或损坏")
            print("    2. wav2vec2 模型处理失败")
            print("    3. linear_interpolation 函数问题")
        
        expected_shape = (1, num_frames, opt.dim_a)
        print(f"  期望形状: {expected_shape}")
        print(f"  实际形状: {w_audio.shape}")
        
        # 测试序列切片
        print(f"\n✂️  测试序列切片...")
        start_idx = 0
        end_idx = min(num_frames, expected_seq_len)
        prev_frames = min(dataset.prev_frames, num_frames)
        
        print(f"  start_idx: {start_idx}")
        print(f"  end_idx: {end_idx}")
        print(f"  prev_frames: {prev_frames}")
        
        if w_audio.shape[1] > 0:  # 如果有数据
            w_audio_cur = w_audio[:, start_idx + prev_frames:end_idx]
            w_audio_prev = w_audio[:, start_idx:start_idx + prev_frames]
            
            print(f"  w_audio_cur 形状: {w_audio_cur.shape}")
            print(f"  w_audio_prev 形状: {w_audio_prev.shape}")
        else:
            print("  ❌ 无法进行切片，因为音频特征为空")
        
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        print("🔍 错误详情:")
        print(traceback.format_exc())
        return False


def debug_wav2vec_model():
    """调试 wav2vec 模型"""
    print("\n🔍 调试 wav2vec 模型...")
    
    try:
        opt = BaseOptions().parse()
        
        # 检查模型路径
        if not os.path.exists(opt.wav2vec_model_path):
            print(f"❌ wav2vec 模型路径不存在: {opt.wav2vec_model_path}")
            return False
        
        print(f"✅ wav2vec 模型路径存在: {opt.wav2vec_model_path}")
        
        # 尝试加载模型
        from transformers import Wav2Vec2FeatureExtractor
        from models.wav2vec2 import Wav2VecModel
        
        print("🔄 加载 wav2vec 预处理器...")
        preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            opt.wav2vec_model_path, 
            local_files_only=True
        )
        print("✅ wav2vec 预处理器加载成功")
        
        print("🔄 加载 wav2vec 模型...")
        model = Wav2VecModel.from_pretrained(
            opt.wav2vec_model_path, 
            local_files_only=True
        )
        print("✅ wav2vec 模型加载成功")
        
        # 测试模型配置
        print(f"📋 模型配置:")
        print(f"  hidden_size: {model.config.hidden_size}")
        print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
        
        return True
        
    except Exception as e:
        print(f"❌ wav2vec 模型调试失败: {e}")
        print(traceback.format_exc())
        return False


def test_linear_interpolation():
    """测试线性插值函数"""
    print("\n🔍 测试线性插值函数...")
    
    try:
        from models.wav2vec2 import linear_interpolation
        
        # 创建测试数据
        batch_size = 1
        feature_dim = 768
        original_seq_len = 100
        target_seq_len = 50
        
        # 创建随机特征
        features = torch.randn(batch_size, original_seq_len, feature_dim)
        print(f"输入特征形状: {features.shape}")
        
        # 应用线性插值
        interpolated = linear_interpolation(features, target_seq_len)
        print(f"插值后特征形状: {interpolated.shape}")
        
        if interpolated.shape[1] == target_seq_len:
            print("✅ 线性插值函数工作正常")
            return True
        else:
            print("❌ 线性插值函数输出形状不正确")
            return False
        
    except Exception as e:
        print(f"❌ 线性插值测试失败: {e}")
        print(traceback.format_exc())
        return False


def main():
    print("🚀 音频潜在表示调试")
    print("=" * 60)
    
    # 测试步骤
    tests = [
        ("wav2vec 模型", debug_wav2vec_model),
        ("线性插值函数", test_linear_interpolation),
        ("音频处理流程", debug_audio_processing),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        results[name] = test_func()
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 调试结果总结:")
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    # 问题诊断
    print(f"\n🔍 问题诊断:")
    if not results.get("wav2vec 模型", False):
        print("  1. wav2vec 模型加载失败 - 检查模型文件")
    elif not results.get("线性插值函数", False):
        print("  2. 线性插值函数有问题 - 检查实现")
    elif not results.get("音频处理流程", False):
        print("  3. 音频处理流程有问题 - 需要详细调试")
    else:
        print("  所有测试通过，问题可能在数据集的其他部分")
    
    print(f"\n💡 建议:")
    print("  1. 检查音频文件是否存在且不为空")
    print("  2. 检查 wav2vec 模型文件完整性")
    print("  3. 检查配置参数是否合理")
    print("  4. 检查数据集的序列切片逻辑")


if __name__ == "__main__":
    main()
