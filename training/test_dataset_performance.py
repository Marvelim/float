"""
性能测试脚本：比较原始数据集和优化数据集的性能
"""

import time
import torch
from dataset import FLOATDataset, create_dataloader
from dataset_test import FLOATDatasetOptimized, create_dataloader_optimized, check_cache_status, clear_cache

def test_dataset_performance(data_root: str, opt: dict, num_samples: int = 10):
    """
    测试数据集性能
    
    Args:
        data_root: 数据根目录
        opt: 配置选项
        num_samples: 测试样本数量
    """
    print("=" * 60)
    print("数据集性能测试")
    print("=" * 60)
    
    # 测试原始数据集
    print("\n1. 测试原始数据集 (dataset.py)")
    print("-" * 40)
    
    try:
        original_dataset = FLOATDataset(data_root=data_root, train=True, opt=opt)
        print(f"原始数据集大小: {len(original_dataset)}")
        
        # 测试 __getitem__ 性能
        start_time = time.time()
        for i in range(min(num_samples, len(original_dataset))):
            data = original_dataset[i]
            if i == 0:
                print(f"数据形状示例:")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
        
        original_time = time.time() - start_time
        print(f"原始数据集 {num_samples} 次 __getitem__ 耗时: {original_time:.2f}秒")
        print(f"平均每次耗时: {original_time/num_samples:.3f}秒")
        
    except Exception as e:
        print(f"原始数据集测试失败: {e}")
        original_time = float('inf')
    
    # 测试优化数据集
    print(f"\n2. 测试优化数据集 (dataset_test.py)")
    print("-" * 40)
    
    try:
        # 检查缓存状态
        check_cache_status(data_root)
        
        optimized_dataset = FLOATDatasetOptimized(
            data_root=data_root, 
            train=True, 
            opt=opt,
            force_preprocess=False
        )
        print(f"优化数据集大小: {len(optimized_dataset)}")
        
        # 测试 __getitem__ 性能
        start_time = time.time()
        for i in range(min(num_samples, len(optimized_dataset))):
            data = optimized_dataset[i]
            if i == 0:
                print(f"数据形状示例:")
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
        
        optimized_time = time.time() - start_time
        print(f"优化数据集 {num_samples} 次 __getitem__ 耗时: {optimized_time:.2f}秒")
        print(f"平均每次耗时: {optimized_time/num_samples:.3f}秒")
        
        # 计算性能提升
        if original_time != float('inf'):
            speedup = original_time / optimized_time
            print(f"\n性能提升: {speedup:.1f}x 倍")
        
    except Exception as e:
        print(f"优化数据集测试失败: {e}")
    
    print("\n" + "=" * 60)


def test_dataloader_performance(data_root: str, opt: dict, batch_size: int = 4, num_batches: int = 5):
    """
    测试数据加载器性能
    
    Args:
        data_root: 数据根目录
        opt: 配置选项
        batch_size: 批次大小
        num_batches: 测试批次数量
    """
    print("=" * 60)
    print("数据加载器性能测试")
    print("=" * 60)
    
    # 测试原始数据加载器
    print(f"\n1. 测试原始数据加载器 (batch_size={batch_size})")
    print("-" * 40)
    
    try:
        original_dataloader = create_dataloader(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=0,  # 避免多进程问题
            train=True,
            opt=opt
        )
        
        start_time = time.time()
        for i, batch in enumerate(original_dataloader):
            if i >= num_batches:
                break
            if i == 0:
                print(f"批次数据形状示例:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
        
        original_loader_time = time.time() - start_time
        print(f"原始数据加载器 {num_batches} 个批次耗时: {original_loader_time:.2f}秒")
        print(f"平均每批次耗时: {original_loader_time/num_batches:.3f}秒")
        
    except Exception as e:
        print(f"原始数据加载器测试失败: {e}")
        original_loader_time = float('inf')
    
    # 测试优化数据加载器
    print(f"\n2. 测试优化数据加载器 (batch_size={batch_size})")
    print("-" * 40)
    
    try:
        optimized_dataloader = create_dataloader_optimized(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=0,  # 避免多进程问题
            train=True,
            opt=opt,
            force_preprocess=False
        )
        
        start_time = time.time()
        for i, batch in enumerate(optimized_dataloader):
            if i >= num_batches:
                break
            if i == 0:
                print(f"批次数据形状示例:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
        
        optimized_loader_time = time.time() - start_time
        print(f"优化数据加载器 {num_batches} 个批次耗时: {optimized_loader_time:.2f}秒")
        print(f"平均每批次耗时: {optimized_loader_time/num_batches:.3f}秒")
        
        # 计算性能提升
        if original_loader_time != float('inf'):
            speedup = original_loader_time / optimized_loader_time
            print(f"\n数据加载器性能提升: {speedup:.1f}x 倍")
        
    except Exception as e:
        print(f"优化数据加载器测试失败: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 示例配置（需要根据实际情况调整）
    class MockOpt:
        def __init__(self):
            self.input_size = 256
            self.dim_w = 512
            self.dim_m = 256
            self.dim_e = 7
            self.wav2vec_sec = 2.0
            self.fps = 25
            self.num_prev_frames = 5
            self.sampling_rate = 16000
            self.wav2vec_model_path = "path/to/wav2vec2"
    
    opt = MockOpt()
    data_root = "/path/to/your/data"  # 请修改为实际数据路径
    
    print("请确保已设置正确的数据路径和模型路径！")
    print(f"当前数据路径: {data_root}")
    
    # 运行性能测试
    # test_dataset_performance(data_root, opt, num_samples=5)
    # test_dataloader_performance(data_root, opt, batch_size=2, num_batches=3)
    
    # 工具函数示例
    print("\n工具函数示例:")
    print("检查缓存状态:")
    check_cache_status(data_root)
    
    print("\n如需清理缓存，取消注释下面的行:")
    print("# clear_cache(data_root)")
