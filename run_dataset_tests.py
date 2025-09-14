#!/usr/bin/env python3
"""
一键运行数据集测试脚本
按推荐顺序运行所有测试，并提供清晰的结果报告
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 运行命令时出错: {e}")
        return False


def check_scripts_exist():
    """检查测试脚本是否存在"""
    required_scripts = [
        'test_dataset_config.py',
        'quick_test_dataset.py', 
        'test_dataset.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        print("❌ 缺少测试脚本:")
        for script in missing_scripts:
            print(f"   {script}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='一键运行数据集测试')
    parser.add_argument('--data_root', type=str,
                       help='数据根目录')
    parser.add_argument('--config', type=str, default='training/example_config.json',
                       help='配置文件路径')
    parser.add_argument('--skip_full_test', action='store_true',
                       help='跳过完整测试，只运行配置检查和快速测试')
    parser.add_argument('--create_sample_config', action='store_true',
                       help='创建示例配置文件后退出')
    
    args = parser.parse_args()
    
    print("🚀 FLOAT 数据集测试套件")
    print("=" * 60)
    
    # 创建示例配置
    if args.create_sample_config:
        cmd = ['python', 'test_dataset_config.py', '--create_sample']
        run_command(cmd, "创建示例配置文件")
        return
    
    # 检查测试脚本是否存在
    if not check_scripts_exist():
        print("\n💡 请确保所有测试脚本都在当前目录中")
        return
    
    # 构建通用参数
    common_args = []
    if args.data_root:
        common_args.extend(['--data_root', args.data_root])
    if args.config and os.path.exists(args.config):
        common_args.extend(['--config', args.config])
    
    # 测试结果跟踪
    test_results = {}
    
    # 步骤 1: 配置检查
    print("\n🎯 步骤 1/3: 配置检查")
    cmd = ['python', 'test_dataset_config.py'] + common_args
    test_results['config_check'] = run_command(cmd, "检查数据集配置")
    
    if not test_results['config_check']:
        print("\n💥 配置检查失败！")
        print("💡 请根据上述错误信息修复配置问题后重试")
        print("💡 可以使用 --create_sample_config 创建示例配置")
        return
    
    # 步骤 2: 快速测试
    print("\n🎯 步骤 2/3: 快速功能测试")
    cmd = ['python', 'quick_test_dataset.py'] + common_args
    test_results['quick_test'] = run_command(cmd, "快速数据集功能测试")
    
    if not test_results['quick_test']:
        print("\n💥 快速测试失败！")
        print("💡 请检查数据目录和文件是否正确")
        print("💡 可以运行: python quick_test_dataset.py --check_structure")
        
        # 询问是否继续完整测试
        if not args.skip_full_test:
            try:
                response = input("\n❓ 是否继续运行完整测试？(y/N): ").strip().lower()
                if response != 'y':
                    return
            except KeyboardInterrupt:
                print("\n👋 测试已取消")
                return
    
    # 步骤 3: 完整测试（可选）
    if not args.skip_full_test:
        print("\n🎯 步骤 3/3: 完整测试")
        cmd = ['python', 'test_dataset.py'] + common_args
        test_results['full_test'] = run_command(cmd, "完整数据集测试")
    else:
        print("\n⏭️  跳过完整测试")
        test_results['full_test'] = None
    
    # 生成测试报告
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    # 结果统计
    passed = sum(1 for result in test_results.values() if result is True)
    failed = sum(1 for result in test_results.values() if result is False)
    skipped = sum(1 for result in test_results.values() if result is None)
    
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"⏭️  跳过: {skipped}")
    
    # 详细结果
    test_names = {
        'config_check': '配置检查',
        'quick_test': '快速测试', 
        'full_test': '完整测试'
    }
    
    print(f"\n📋 详细结果:")
    for key, result in test_results.items():
        name = test_names.get(key, key)
        if result is True:
            print(f"   ✅ {name}: 通过")
        elif result is False:
            print(f"   ❌ {name}: 失败")
        else:
            print(f"   ⏭️  {name}: 跳过")
    
    # 最终建议
    print(f"\n💡 建议:")
    if all(r is not False for r in test_results.values()):
        print("   🎉 测试通过！数据集配置正确，可以开始训练")
        print("   📝 下一步: 运行训练脚本")
    else:
        print("   🔧 请根据上述错误信息修复问题")
        print("   📖 详细说明请查看: README_dataset_testing.md")
    
    # 常用命令提示
    print(f"\n🛠️  常用命令:")
    print("   检查数据结构: python quick_test_dataset.py --check_structure")
    print("   创建示例配置: python test_dataset_config.py --create_sample")
    print("   查看帮助文档: cat README_dataset_testing.md")


if __name__ == "__main__":
    main()
