#!/usr/bin/env python3
"""
数据集加载测试脚本
测试 FLOATDataset 的各项功能，包括数据加载、预处理和数据完整性检查
"""

import os
import sys
import torch
import traceback
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import time
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.dataset import FLOATDataset, create_dataloader
from options.base_options import BaseOptions


class DatasetTester:
    """数据集测试器"""
    
    def __init__(self, config_path=None):
        """
        初始化测试器
        
        Args:
            config_path: 配置文件路径，如果为None则使用命令行参数
        """
        if config_path and os.path.exists(config_path):
            self.opt = self.load_config_from_json(config_path)
        else:
            self.opt = BaseOptions().parse()
        
        self.test_results = {
            'dataset_creation': False,
            'data_loading': False,
            'batch_loading': False,
            'data_shapes': False,
            'data_types': False,
            'error_handling': False,
            'performance': {}
        }
    
    def load_config_from_json(self, config_path):
        """从JSON配置文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建一个简单的配置对象
        class Config:
            pass
        
        opt = Config()
        
        # 合并所有配置项
        for section in config.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    setattr(opt, key, value)
        
        return opt
    
    def test_dataset_creation(self):
        """测试数据集创建"""
        print("🔍 测试数据集创建...")
        try:
            # 检查必要的路径
            if not hasattr(self.opt, 'data_root'):
                print("❌ 配置中缺少 data_root")
                return False
            
            if not os.path.exists(self.opt.data_root):
                print(f"❌ 数据根目录不存在: {self.opt.data_root}")
                return False
            
            # 创建数据集
            dataset = FLOATDataset(
                data_root=self.opt.data_root,
                train=True,
                opt=self.opt
            )
            
            print(f"✅ 数据集创建成功")
            print(f"   数据集大小: {len(dataset)}")
            
            if len(dataset) == 0:
                print("⚠️  警告: 数据集为空")
                return False
            
            self.dataset = dataset
            self.test_results['dataset_creation'] = True
            return True
            
        except Exception as e:
            print(f"❌ 数据集创建失败: {str(e)}")
            print(f"   错误详情: {traceback.format_exc()}")
            return False
    
    def test_single_data_loading(self):
        """测试单个数据项加载"""
        print("\n🔍 测试单个数据项加载...")
        try:
            if not hasattr(self, 'dataset'):
                print("❌ 数据集未创建")
                return False
            
            # 测试加载第一个数据项
            start_time = time.time()
            data_item = self.dataset[0]
            load_time = time.time() - start_time
            
            print(f"✅ 单个数据项加载成功")
            print(f"   加载时间: {load_time:.2f}秒")
            
            # 检查数据项结构
            expected_keys = [
                'video_cur', 'video_prev', 'motion_latent_cur', 'motion_latent_prev',
                'audio_latent_cur', 'audio_latent_prev', 'reference_frame', 
                'reference_motion', 'emotion_features', 'actor_id'
            ]
            
            missing_keys = [key for key in expected_keys if key not in data_item]
            if missing_keys:
                print(f"⚠️  缺少键: {missing_keys}")
            else:
                print("✅ 数据项包含所有必要的键")
            
            self.sample_data = data_item
            self.test_results['data_loading'] = True
            self.test_results['performance']['single_load_time'] = load_time
            return True
            
        except Exception as e:
            print(f"❌ 单个数据项加载失败: {str(e)}")
            print(f"   错误详情: {traceback.format_exc()}")
            return False
    
    def test_data_shapes_and_types(self):
        """测试数据形状和类型"""
        print("\n🔍 测试数据形状和类型...")
        try:
            if not hasattr(self, 'sample_data'):
                print("❌ 没有样本数据")
                return False
            
            data = self.sample_data
            
            # 检查各项数据的形状和类型
            checks = [
                ('video_cur', torch.Tensor, 4),  # (T, C, H, W)
                ('video_prev', torch.Tensor, 4),  # (T, C, H, W)
                ('motion_latent_cur', torch.Tensor, 2),  # (T, motion_dim)
                ('motion_latent_prev', torch.Tensor, 2),  # (T, motion_dim)
                ('audio_latent_cur', torch.Tensor, 2),  # (T, audio_dim)
                ('audio_latent_prev', torch.Tensor, 2),  # (T, audio_dim)
                ('reference_frame', torch.Tensor, 3),  # (C, H, W)
                ('reference_motion', torch.Tensor, 1),  # (motion_dim,)
                ('emotion_features', torch.Tensor, 1),  # (emotion_dim,)
            ]
            
            all_passed = True
            for key, expected_type, expected_dims in checks:
                if key in data:
                    value = data[key]
                    if not isinstance(value, expected_type):
                        print(f"❌ {key}: 类型错误，期望 {expected_type}，实际 {type(value)}")
                        all_passed = False
                    elif value.dim() != expected_dims:
                        print(f"❌ {key}: 维度错误，期望 {expected_dims}D，实际 {value.dim()}D")
                        all_passed = False
                    else:
                        print(f"✅ {key}: 形状 {tuple(value.shape)}, 类型 {type(value)}")
                else:
                    print(f"⚠️  缺少键: {key}")
                    all_passed = False
            
            # 检查数值范围
            if 'video_cur' in data:
                video_min, video_max = data['video_cur'].min(), data['video_cur'].max()
                print(f"   视频像素值范围: [{video_min:.3f}, {video_max:.3f}]")
                if video_min < -2 or video_max > 2:
                    print("⚠️  视频像素值范围异常，期望在 [-1, 1] 附近")
            
            self.test_results['data_shapes'] = all_passed
            self.test_results['data_types'] = all_passed
            return all_passed

        except Exception as e:
            print(f"❌ 数据形状和类型检查失败: {str(e)}")
            return False

    def test_batch_loading(self):
        """测试批量数据加载"""
        print("\n🔍 测试批量数据加载...")
        try:
            if not hasattr(self, 'dataset'):
                print("❌ 数据集未创建")
                return False

            batch_size = min(2, len(self.dataset))  # 使用小批量进行测试

            dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 避免多进程问题
                pin_memory=False
            )

            start_time = time.time()
            batch = next(iter(dataloader))
            batch_load_time = time.time() - start_time

            print(f"✅ 批量数据加载成功")
            print(f"   批次大小: {batch_size}")
            print(f"   批量加载时间: {batch_load_time:.2f}秒")

            # 检查批量数据形状
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {tuple(value.shape)}")
                else:
                    print(f"   {key}: {type(value)} (长度: {len(value) if hasattr(value, '__len__') else 'N/A'})")

            self.test_results['batch_loading'] = True
            self.test_results['performance']['batch_load_time'] = batch_load_time
            return True

        except Exception as e:
            print(f"❌ 批量数据加载失败: {str(e)}")
            print(f"   错误详情: {traceback.format_exc()}")
            return False

    def test_error_handling(self):
        """测试错误处理"""
        print("\n🔍 测试错误处理...")
        try:
            # 测试无效索引
            try:
                invalid_data = self.dataset[len(self.dataset)]
                print("⚠️  无效索引未抛出异常")
                return False
            except IndexError:
                print("✅ 无效索引正确抛出 IndexError")

            # 测试负索引
            try:
                negative_data = self.dataset[-1]
                print("✅ 负索引处理正常")
            except Exception as e:
                print(f"⚠️  负索引处理异常: {e}")

            self.test_results['error_handling'] = True
            return True

        except Exception as e:
            print(f"❌ 错误处理测试失败: {str(e)}")
            return False

    def test_performance(self):
        """测试性能"""
        print("\n🔍 测试性能...")
        try:
            if not hasattr(self, 'dataset'):
                print("❌ 数据集未创建")
                return False

            # 测试多个数据项的加载时间
            num_samples = min(5, len(self.dataset))

            start_time = time.time()
            for i in range(num_samples):
                _ = self.dataset[i]
            total_time = time.time() - start_time

            avg_time = total_time / num_samples
            print(f"✅ 性能测试完成")
            print(f"   测试样本数: {num_samples}")
            print(f"   总时间: {total_time:.2f}秒")
            print(f"   平均每个样本: {avg_time:.2f}秒")

            self.test_results['performance']['avg_sample_time'] = avg_time
            self.test_results['performance']['total_test_time'] = total_time

            # 性能建议
            if avg_time > 5.0:
                print("⚠️  警告: 数据加载较慢，建议检查数据预处理或使用更快的存储")
            elif avg_time > 2.0:
                print("⚠️  注意: 数据加载时间较长，可能影响训练效率")
            else:
                print("✅ 数据加载性能良好")

            return True

        except Exception as e:
            print(f"❌ 性能测试失败: {str(e)}")
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始数据集测试...")
        print("=" * 60)

        tests = [
            ('数据集创建', self.test_dataset_creation),
            ('单个数据加载', self.test_single_data_loading),
            ('数据形状和类型', self.test_data_shapes_and_types),
            ('批量数据加载', self.test_batch_loading),
            ('错误处理', self.test_error_handling),
            ('性能测试', self.test_performance),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"❌ {test_name} 测试异常: {str(e)}")

        print("\n" + "=" * 60)
        print("📊 测试结果汇总:")
        print(f"   通过测试: {passed_tests}/{total_tests}")
        print(f"   成功率: {passed_tests/total_tests*100:.1f}%")

        # 详细结果
        print("\n📋 详细结果:")
        for key, value in self.test_results.items():
            if key != 'performance':
                status = "✅ 通过" if value else "❌ 失败"
                print(f"   {key}: {status}")

        # 性能信息
        if self.test_results['performance']:
            print("\n⏱️  性能信息:")
            for key, value in self.test_results['performance'].items():
                print(f"   {key}: {value:.2f}秒")

        return passed_tests == total_tests


def main():
    parser = argparse.ArgumentParser(description='FLOAT数据集测试脚本')
    parser.add_argument('--config', type=str, default='training/example_config.json',
                       help='配置文件路径')
    parser.add_argument('--data_root', type=str,
                       help='数据根目录 (覆盖配置文件中的设置)')

    args = parser.parse_args()

    # 创建测试器
    tester = DatasetTester(args.config)

    # 如果指定了data_root，覆盖配置
    if args.data_root:
        tester.opt.data_root = args.data_root

    # 运行测试
    success = tester.run_all_tests()

    if success:
        print("\n🎉 所有测试通过！数据集加载功能正常。")
        sys.exit(0)
    else:
        print("\n💥 部分测试失败，请检查数据集配置和数据文件。")
        sys.exit(1)


if __name__ == "__main__":
    main()
