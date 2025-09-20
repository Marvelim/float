#!/usr/bin/env python3
"""
测试批处理 load 函数
"""

import os
import sys
import torch
import time

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from options.base_options import BaseOptions
from training.dataset import create_dataloader
from models.float.FLOAT import FLOAT
from training.train import load

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

def test_batch_load():
    """测试批处理 load 函数"""
    print("测试批处理 load 函数...")
    
    # 解析选项
    opt = BaseOptions().parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.rank = device
    
    # 创建模型
    print("创建模型...")
    model = FLOAT(opt).to(device)
    model.audio_encoder.requires_grad_(False)
    model.emotion_encoder.requires_grad_(False)
    model.motion_autoencoder.requires_grad_(False)
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = create_dataloader(
        data_root="../datasets/ravdess_processed",
        batch_size=2,  # 更小的批次测试
        num_workers=0,  # 避免多进程问题
        train=True,
        opt=opt
    )
    
    # 获取一个批次
    print("获取批次数据...")
    batch_data = next(iter(dataloader))
    print(f"原始批次大小: {len(batch_data)}")
    
    # 测试批处理 load 函数
    print("测试批处理 load 函数...")
    start_time = time.time()
    
    try:
        processed_data = load(batch_data, model, device, opt)
        end_time = time.time()
        
        print(f"✅ 批处理 load 函数执行成功！")
        print(f"⏱️  处理时间: {end_time - start_time:.2f} 秒")
        print(f"📊 处理后的数据键: {list(processed_data.keys())}")
        
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
        
        # 验证完整数据形状
        actual_batch_size = processed_data['full_videos'].shape[0]  # 使用实际批次大小
        target_frames = processed_data['target_frames']
        
        expected_shapes = {
            'full_videos': (actual_batch_size, target_frames, 3, opt.input_size, opt.input_size),
            'full_motion_latent': (actual_batch_size, target_frames, opt.dim_w),  # 使用 dim_w 而不是 dim_m
            'full_audio_latent': (actual_batch_size, target_frames, opt.dim_a),
            'emotion_features': (actual_batch_size, opt.dim_e),
        }
        
        print("\n🔍 验证完整数据形状:")
        all_correct = True
        for key, expected_shape in expected_shapes.items():
            if key in processed_data:
                actual_shape = processed_data[key].shape
                if actual_shape == expected_shape:
                    print(f"  ✅ {key}: {actual_shape}")
                else:
                    print(f"  ❌ {key}: 期望 {expected_shape}, 实际 {actual_shape}")
                    all_correct = False
            else:
                print(f"  ❌ {key}: 缺失")
                all_correct = False
        
        # 测试 get_batch_sample 函数
        print("\n🧪 测试 get_batch_sample 函数...")
        from training.train import get_batch_sample
        
        try:
            batch_sample = get_batch_sample(processed_data, opt)
            print("✅ get_batch_sample 执行成功！")
            
            # 验证切分后的数据形状
            sequence_length = int(opt.wav2vec_sec * opt.fps)
            prev_frames = int(opt.num_prev_frames)
            actual_batch_size = batch_sample['video_cur'].shape[0]  # 使用实际批次大小
            
            expected_sample_shapes = {
                'video_cur': (actual_batch_size, sequence_length, 3, opt.input_size, opt.input_size),
                'video_prev': (actual_batch_size, prev_frames, 3, opt.input_size, opt.input_size),
                'motion_latent_cur': (actual_batch_size, sequence_length, opt.dim_w),  # 使用 dim_w
                'motion_latent_prev': (actual_batch_size, prev_frames, opt.dim_w),    # 使用 dim_w
                'audio_latent_cur': (actual_batch_size, sequence_length, opt.dim_a),
                'audio_latent_prev': (actual_batch_size, prev_frames, opt.dim_a),
                'reference_motion': (actual_batch_size, opt.dim_w),  # 使用 dim_w
                'emotion_features': (actual_batch_size, opt.dim_e),
            }
            
            print("\n🔍 验证切分后数据形状:")
            sample_correct = True
            for key, expected_shape in expected_sample_shapes.items():
                if key in batch_sample:
                    actual_shape = batch_sample[key].shape
                    if actual_shape == expected_shape:
                        print(f"  ✅ {key}: {actual_shape}")
                    else:
                        print(f"  ❌ {key}: 期望 {expected_shape}, 实际 {actual_shape}")
                        sample_correct = False
                else:
                    print(f"  ❌ {key}: 缺失")
                    sample_correct = False
            
            if sample_correct:
                print("\n🎉 所有切分后数据形状都正确！")
                from models.float.FLOAT import FLOAT
                from models.float.FMT import FlowMatchingTransformer
                model = FLOAT(opt).to(device)
                model.audio_encoder.requires_grad_(False)
                model.emotion_encoder.requires_grad_(False)
                model.motion_autoencoder.requires_grad_(False)
                

            else:
                print("\n⚠️  部分切分后数据形状不正确")
                
        except Exception as e:
            print(f"❌ get_batch_sample 执行失败: {e}")
            import traceback
            traceback.print_exc()
        
        if all_correct:
            print("\n🎉 所有完整数据形状都正确！")
        else:
            print("\n⚠️  部分完整数据形状不正确")
            
    except Exception as e:
        print(f"❌ 批处理 load 函数执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_batch_load()
