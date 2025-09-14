#!/usr/bin/env python3
"""
数据集配置测试脚本
专门测试数据集配置是否正确，包括路径、模型文件、参数等
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from options.base_options import BaseOptions


class ConfigTester:
    """配置测试器"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []
    
    def check_paths(self, opt):
        """检查路径配置"""
        print("🔍 检查路径配置...")
        
        # 检查数据根目录
        if hasattr(opt, 'data_root'):
            if os.path.exists(opt.data_root):
                self.passed_checks.append(f"数据根目录存在: {opt.data_root}")
                
                # 检查数据子目录
                expected_subdirs = ['ravdess_processed', 'ravdess_raw']
                found_subdirs = []
                for subdir in expected_subdirs:
                    full_path = os.path.join(opt.data_root, subdir)
                    if os.path.exists(full_path):
                        found_subdirs.append(subdir)
                
                if found_subdirs:
                    self.passed_checks.append(f"找到数据子目录: {found_subdirs}")
                else:
                    self.warnings.append("未找到预期的数据子目录 (ravdess_processed, ravdess_raw)")
            else:
                self.issues.append(f"数据根目录不存在: {opt.data_root}")
        else:
            self.issues.append("配置中缺少 data_root")
        
        # 检查模型路径
        model_paths = [
            ('wav2vec_model_path', 'Wav2Vec2模型'),
            ('audio2emotion_path', '音频情感模型')
        ]
        
        for attr, desc in model_paths:
            if hasattr(opt, attr):
                path = getattr(opt, attr)
                if os.path.exists(path):
                    self.passed_checks.append(f"{desc}路径存在: {path}")
                else:
                    self.issues.append(f"{desc}路径不存在: {path}")
            else:
                self.issues.append(f"配置中缺少 {attr}")
    
    def check_parameters(self, opt):
        """检查参数配置"""
        print("🔍 检查参数配置...")
        
        # 必要参数检查
        required_params = [
            ('input_size', int, [256, 512, 1024]),
            ('dim_w', int, None),
            ('dim_m', int, None),
            ('dim_a', int, None),
            ('dim_e', int, [7]),  # 情感维度通常是7
            ('fps', (int, float), None),
            ('sampling_rate', int, [16000, 22050, 44100]),
            ('wav2vec_sec', (int, float), None),
            ('num_prev_frames', int, None)
        ]
        
        for param, expected_type, valid_values in required_params:
            if hasattr(opt, param):
                value = getattr(opt, param)
                
                # 检查类型
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        self.issues.append(f"{param} 类型错误: 期望 {expected_type}, 实际 {type(value)}")
                        continue
                else:
                    if not isinstance(value, expected_type):
                        self.issues.append(f"{param} 类型错误: 期望 {expected_type}, 实际 {type(value)}")
                        continue
                
                # 检查值范围
                if valid_values and value not in valid_values:
                    self.warnings.append(f"{param} 值可能不常见: {value}, 常见值: {valid_values}")
                
                self.passed_checks.append(f"{param}: {value}")
            else:
                self.issues.append(f"配置中缺少必要参数: {param}")
        
        # 参数合理性检查
        if hasattr(opt, 'input_size') and hasattr(opt, 'dim_w'):
            if opt.dim_w > opt.input_size * 2:
                self.warnings.append(f"dim_w ({opt.dim_w}) 相对于 input_size ({opt.input_size}) 可能过大")
        
        if hasattr(opt, 'wav2vec_sec') and hasattr(opt, 'fps'):
            expected_frames = int(opt.wav2vec_sec * opt.fps)
            self.passed_checks.append(f"预期序列长度: {expected_frames} 帧")
    
    def check_dependencies(self):
        """检查依赖库"""
        print("🔍 检查依赖库...")
        
        required_libs = [
            ('torch', 'PyTorch'),
            ('cv2', 'OpenCV'),
            ('librosa', 'Librosa'),
            ('transformers', 'Transformers'),
            ('numpy', 'NumPy')
        ]
        
        for lib, desc in required_libs:
            try:
                __import__(lib)
                self.passed_checks.append(f"{desc} 已安装")
            except ImportError:
                self.issues.append(f"缺少依赖库: {desc} ({lib})")
    
    def check_json_config(self, config_path):
        """检查JSON配置文件"""
        print(f"🔍 检查JSON配置文件: {config_path}")
        
        if not os.path.exists(config_path):
            self.issues.append(f"配置文件不存在: {config_path}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.passed_checks.append("JSON配置文件格式正确")
            
            # 检查配置结构
            expected_sections = ['model', 'training', 'data']
            for section in expected_sections:
                if section in config:
                    self.passed_checks.append(f"包含 {section} 配置节")
                else:
                    self.warnings.append(f"缺少 {section} 配置节")
            
            return config
            
        except json.JSONDecodeError as e:
            self.issues.append(f"JSON配置文件格式错误: {e}")
            return None
    
    def run_all_checks(self, config_path=None):
        """运行所有检查"""
        print("🚀 开始配置检查...")
        print("=" * 60)
        
        # 检查JSON配置（如果提供）
        json_config = None
        if config_path:
            json_config = self.check_json_config(config_path)
        
        # 检查依赖库
        self.check_dependencies()
        
        # 获取配置
        try:
            opt = BaseOptions().parse()
            self.passed_checks.append("成功解析命令行配置")
        except Exception as e:
            self.issues.append(f"解析配置失败: {e}")
            return False
        
        # 检查路径和参数
        self.check_paths(opt)
        self.check_parameters(opt)
        
        # 输出结果
        self.print_results()
        
        return len(self.issues) == 0
    
    def print_results(self):
        """打印检查结果"""
        print("\n" + "=" * 60)
        print("📊 配置检查结果:")
        
        if self.passed_checks:
            print(f"\n✅ 通过的检查 ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"   ✅ {check}")
        
        if self.warnings:
            print(f"\n⚠️  警告 ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        if self.issues:
            print(f"\n❌ 问题 ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   ❌ {issue}")
        
        # 总结
        total_checks = len(self.passed_checks) + len(self.warnings) + len(self.issues)
        success_rate = len(self.passed_checks) / total_checks * 100 if total_checks > 0 else 0
        
        print(f"\n📈 总结:")
        print(f"   通过: {len(self.passed_checks)}")
        print(f"   警告: {len(self.warnings)}")
        print(f"   错误: {len(self.issues)}")
        print(f"   成功率: {success_rate:.1f}%")


def create_sample_config():
    """创建示例配置文件"""
    sample_config = {
        "model": {
            "input_size": 512,
            "dim_w": 512,
            "dim_m": 20,
            "dim_a": 512,
            "dim_e": 7,
            "fps": 25,
            "sampling_rate": 16000,
            "wav2vec_sec": 0.64,
            "num_prev_frames": 4,
            "wav2vec_model_path": "./checkpoints/wav2vec2-base-960h",
            "audio2emotion_path": "./checkpoints/wav2vec-english-speech-emotion-recognition",
            "only_last_features": True
        },
        "data": {
            "data_root": "/home/mli374/float/datasets",
            "num_workers": 4
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 1e-4
        }
    }
    
    config_path = "sample_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 创建示例配置文件: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='数据集配置测试')
    parser.add_argument('--config', type=str, default='training/example_config.json',
                       help='配置文件路径')
    parser.add_argument('--create_sample', action='store_true',
                       help='创建示例配置文件')
    parser.add_argument('--data_root', type=str,
                       help='数据根目录 (覆盖配置文件设置)')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config()
        return
    
    # 如果指定了data_root，添加到命令行参数
    if args.data_root:
        sys.argv.extend(['--data_root', args.data_root])
    
    # 运行配置检查
    tester = ConfigTester()
    success = tester.run_all_checks(args.config)
    
    if success:
        print("\n🎉 配置检查通过！可以尝试运行数据集测试。")
        print("💡 下一步: python quick_test_dataset.py")
    else:
        print("\n💥 配置存在问题，请根据上述提示进行修复。")
        print("💡 可以使用 --create_sample 创建示例配置文件")


if __name__ == "__main__":
    main()
