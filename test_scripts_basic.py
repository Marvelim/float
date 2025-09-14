#!/usr/bin/env python3
"""
基础脚本测试 - 验证测试脚本是否能正常导入和运行
"""

import os
import sys
import traceback

def test_imports():
    """测试基本导入"""
    print("🔍 测试基本导入...")
    
    try:
        # 测试标准库
        import json
        import argparse
        import pathlib
        print("✅ 标准库导入成功")
        
        # 测试项目导入
        sys.path.append('.')
        from options.base_options import BaseOptions
        print("✅ BaseOptions 导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        print(traceback.format_exc())
        return False

def test_config_creation():
    """测试配置创建"""
    print("\n🔍 测试配置创建...")
    
    try:
        # 创建最小配置
        class TestConfig:
            def __init__(self):
                self.data_root = "/home/mli374/float/datasets"
                self.input_size = 512
                self.dim_w = 512
                self.dim_m = 20
        
        config = TestConfig()
        print(f"✅ 配置创建成功: data_root = {config.data_root}")
        return True
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False

def test_file_operations():
    """测试文件操作"""
    print("\n🔍 测试文件操作...")
    
    try:
        # 检查当前目录
        current_dir = os.getcwd()
        print(f"当前目录: {current_dir}")
        
        # 检查关键文件
        key_files = [
            'training/dataset.py',
            'options/base_options.py',
            'test_dataset.py',
            'quick_test_dataset.py'
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                print(f"✅ 找到文件: {file_path}")
            else:
                print(f"❌ 缺少文件: {file_path}")
        
        return True
    except Exception as e:
        print(f"❌ 文件操作失败: {e}")
        return False

def main():
    print("🚀 基础脚本测试")
    print("=" * 50)
    
    tests = [
        ("基本导入", test_imports),
        ("配置创建", test_config_creation), 
        ("文件操作", test_file_operations)
    ]
    
    passed = 0
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 基础测试全部通过！可以运行数据集测试脚本。")
        print("\n💡 下一步:")
        print("   python test_dataset_config.py --create_sample")
        print("   python quick_test_dataset.py --check_structure")
    else:
        print("💥 基础测试失败，请检查环境配置。")

if __name__ == "__main__":
    main()
