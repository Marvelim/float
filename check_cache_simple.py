#!/usr/bin/env python3
"""
简单的缓存检查脚本
"""

import os
import sys
from pathlib import Path

def check_cache_in_directory(cache_dir):
    """检查指定目录中的缓存"""
    cache_path = Path(cache_dir)
    
    print(f"检查缓存目录: {cache_path}")
    print("=" * 60)
    
    if not cache_path.exists():
        print(f"❌ 缓存目录不存在: {cache_path}")
        return False
    
    print(f"✅ 缓存目录存在: {cache_path}")
    
    # 检查训练集和测试集缓存文件
    train_cache = cache_path / "preprocessed_train.pkl"
    test_cache = cache_path / "preprocessed_test.pkl"
    
    print(f"\n📋 缓存文件状态:")
    
    if train_cache.exists():
        size_mb = train_cache.stat().st_size / (1024 * 1024)
        print(f"✅ 训练集缓存: {train_cache} ({size_mb:.2f} MB)")
    else:
        print(f"❌ 训练集缓存不存在: {train_cache}")
    
    if test_cache.exists():
        size_mb = test_cache.stat().st_size / (1024 * 1024)
        print(f"✅ 测试集缓存: {test_cache} ({size_mb:.2f} MB)")
    else:
        print(f"❌ 测试集缓存不存在: {test_cache}")
    
    # 列出目录中的所有文件
    print(f"\n📁 目录内容:")
    try:
        for item in cache_path.iterdir():
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  📄 {item.name} ({size_mb:.2f} MB)")
            elif item.is_dir():
                print(f"  📁 {item.name}/")
    except Exception as e:
        print(f"❌ 无法列出目录内容: {e}")
    
    return train_cache.exists() or test_cache.exists()

def main():
    # 检查几个可能的缓存位置
    possible_locations = [
        Path.home() / "tmp",
        Path.home() / "tmp" / "cache",
        Path("/home/mli374/float/datasets/cache"),
        Path("/tmp/float_cache"),
    ]
    
    print("🔍 检查可能的缓存位置...")
    print("=" * 60)
    
    found_cache = False
    
    for location in possible_locations:
        print(f"\n检查: {location}")
        if location.exists():
            if check_cache_in_directory(location):
                found_cache = True
        else:
            print(f"❌ 目录不存在: {location}")
    
    print("\n" + "=" * 60)
    if found_cache:
        print("✅ 找到了预处理缓存文件！")
        print("💡 下次训练时会直接使用缓存，无需重新预处理")
    else:
        print("❌ 没有找到预处理缓存文件")
        print("💡 首次训练时会进行预处理并创建缓存")

if __name__ == "__main__":
    main()
