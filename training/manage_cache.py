#!/usr/bin/env python3
"""
缓存管理工具
用于管理优化数据集的缓存文件
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.dataset_test import check_cache_status, clear_cache


def main():
    parser = argparse.ArgumentParser(description="FLOAT 优化数据集缓存管理工具")
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录路径')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='缓存目录路径，如果为 None 则使用 data_root/cache')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 检查缓存状态
    status_parser = subparsers.add_parser('status', help='检查缓存状态')
    
    # 清理缓存
    clear_parser = subparsers.add_parser('clear', help='清理缓存')
    clear_parser.add_argument('--confirm', action='store_true',
                             help='确认清理缓存（不加此参数会提示确认）')
    
    # 预处理数据
    preprocess_parser = subparsers.add_parser('preprocess', help='预处理数据')
    preprocess_parser.add_argument('--config', type=str, required=True,
                                  help='配置文件路径或配置字典')
    preprocess_parser.add_argument('--train', action='store_true',
                                  help='预处理训练集')
    preprocess_parser.add_argument('--test', action='store_true',
                                  help='预处理测试集')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # 检查数据根目录是否存在
    if not Path(args.data_root).exists():
        print(f"错误: 数据根目录不存在: {args.data_root}")
        return
    
    if args.command == 'status':
        print("=" * 60)
        print("缓存状态检查")
        print("=" * 60)
        check_cache_status(args.data_root, args.cache_dir)
        
    elif args.command == 'clear':
        print("=" * 60)
        print("清理缓存")
        print("=" * 60)
        
        # 显示当前缓存状态
        check_cache_status(args.data_root, args.cache_dir)
        
        if not args.confirm:
            response = input("\n确认清理缓存？这将删除所有预处理数据 (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("取消清理缓存")
                return
        
        clear_cache(args.data_root, args.cache_dir)
        print("缓存清理完成")
        
    elif args.command == 'preprocess':
        print("=" * 60)
        print("预处理数据")
        print("=" * 60)
        
        # 这里需要导入相关模块进行预处理
        try:
            from training.dataset_test import FLOATDatasetOptimized
            from options.base_options import BaseOptions
            
            # 解析配置
            if args.config.endswith('.py') or '/' in args.config:
                # 配置文件路径
                print(f"从配置文件加载: {args.config}")
                # 这里可以添加配置文件加载逻辑
                print("配置文件加载功能待实现")
                return
            else:
                # 使用默认配置
                print("使用默认配置")
                base_opt = BaseOptions()
                opt = base_opt.parse()
                opt.data_root = args.data_root
            
            # 预处理训练集
            if args.train or (not args.train and not args.test):
                print("\n预处理训练集...")
                train_dataset = FLOATDatasetOptimized(
                    data_root=args.data_root,
                    train=True,
                    opt=opt,
                    cache_dir=args.cache_dir,
                    force_preprocess=True
                )
                print(f"训练集预处理完成，共 {len(train_dataset)} 个样本")
            
            # 预处理测试集
            if args.test or (not args.train and not args.test):
                print("\n预处理测试集...")
                test_dataset = FLOATDatasetOptimized(
                    data_root=args.data_root,
                    train=False,
                    opt=opt,
                    cache_dir=args.cache_dir,
                    force_preprocess=True
                )
                print(f"测试集预处理完成，共 {len(test_dataset)} 个样本")
            
            print("\n预处理完成！")
            
        except Exception as e:
            print(f"预处理失败: {e}")
            print("请确保配置正确且有足够的GPU内存")


def print_usage_examples():
    """打印使用示例"""
    print("使用示例:")
    print()
    print("1. 检查缓存状态:")
    print("   python manage_cache.py --data_root /path/to/data status")
    print()
    print("2. 清理缓存:")
    print("   python manage_cache.py --data_root /path/to/data clear")
    print("   python manage_cache.py --data_root /path/to/data clear --confirm")
    print()
    print("3. 预处理数据:")
    print("   python manage_cache.py --data_root /path/to/data preprocess --config default")
    print("   python manage_cache.py --data_root /path/to/data preprocess --config default --train")
    print("   python manage_cache.py --data_root /path/to/data preprocess --config default --test")
    print()
    print("4. 指定缓存目录:")
    print("   python manage_cache.py --data_root /path/to/data --cache_dir /tmp/cache status")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("FLOAT 优化数据集缓存管理工具")
        print("=" * 40)
        print_usage_examples()
    else:
        main()
