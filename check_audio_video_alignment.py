#!/usr/bin/env python3
"""
检查数据集中音视频对齐情况的脚本
Check audio-video alignment in the dataset
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_video_info(video_path: str) -> Dict:
    """获取视频信息 - 使用 ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        # 查找视频流
        video_stream = None
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            return None
        
        # 获取时长
        duration = float(data['format']['duration'])
        fps = eval(video_stream['r_frame_rate'])  # 例如 "25/1" -> 25.0
        frame_count = int(duration * fps)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
    except Exception as e:
        print(f"Error getting video info for {video_path}: {e}")
        return None

def get_audio_info(audio_path: str) -> Dict:
    """获取音频信息 - 使用 ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        
        # 查找音频流
        audio_stream = None
        for stream in data['streams']:
            if stream['codec_type'] == 'audio':
                audio_stream = stream
                break
        
        if not audio_stream:
            return None
        
        # 获取时长
        duration = float(data['format']['duration'])
        sample_rate = int(audio_stream['sample_rate'])
        channels = int(audio_stream['channels'])
        samples = int(duration * sample_rate)
        
        return {
            'sample_rate': sample_rate,
            'duration': duration,
            'samples': samples,
            'channels': channels
        }
    except Exception as e:
        print(f"Error getting audio info for {audio_path}: {e}")
        return None

def check_alignment(video_path: str, audio_path: str, tolerance: float = 0.1) -> Dict:
    """检查音视频对齐情况"""
    video_info = get_video_info(video_path)
    audio_info = get_audio_info(audio_path)
    
    if video_info is None or audio_info is None:
        return {
            'aligned': False,
            'error': 'Failed to load video or audio',
            'video_info': video_info,
            'audio_info': audio_info
        }
    
    # 计算时长差异
    duration_diff = abs(video_info['duration'] - audio_info['duration'])
    is_aligned = duration_diff <= tolerance
    
    return {
        'aligned': is_aligned,
        'duration_diff': duration_diff,
        'video_duration': video_info['duration'],
        'audio_duration': audio_info['duration'],
        'video_info': video_info,
        'audio_info': audio_info,
        'tolerance': tolerance
    }

def analyze_dataset(data_root: str, max_samples: int = 100) -> Dict:
    """分析数据集中的音视频对齐情况"""
    data_root = Path(data_root)
    results = []
    
    # 统计信息
    stats = {
        'total_samples': 0,
        'aligned_samples': 0,
        'misaligned_samples': 0,
        'failed_samples': 0,
        'duration_diffs': [],
        'video_durations': [],
        'audio_durations': [],
        'fps_values': [],
        'sample_rates': []
    }
    
    print(f"分析数据集: {data_root}")
    print(f"最大样本数: {max_samples}")
    print("=" * 50)
    
    # 遍历训练和测试数据
    for split in ['train', 'test']:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"跳过不存在的目录: {split_dir}")
            continue
            
        print(f"\n处理 {split} 数据...")
        
        # 遍历所有演员文件夹
        for actor_dir in split_dir.glob("Actor_*"):
            if not actor_dir.is_dir():
                continue
                
            print(f"  处理 {actor_dir.name}...")
            
            # 遍历视频文件
            for video_file in actor_dir.glob("*_processed.mp4"):
                if stats['total_samples'] >= max_samples:
                    break
                    
                # 查找对应的音频文件
                audio_file = video_file.with_suffix('.wav')
                if not audio_file.exists():
                    # 尝试其他音频格式
                    for ext in ['.wav', '.mp3', '.flac']:
                        audio_file = video_file.with_suffix(ext)
                        if audio_file.exists():
                            break
                
                if not audio_file.exists():
                    print(f"    警告: 找不到音频文件 {video_file.name}")
                    stats['failed_samples'] += 1
                    continue
                
                # 检查对齐情况
                alignment_result = check_alignment(str(video_file), str(audio_file))
                results.append({
                    'video_path': str(video_file),
                    'audio_path': str(audio_file),
                    'actor': actor_dir.name,
                    'split': split,
                    **alignment_result
                })
                
                # 更新统计信息
                stats['total_samples'] += 1
                
                if 'error' in alignment_result:
                    stats['failed_samples'] += 1
                elif alignment_result['aligned']:
                    stats['aligned_samples'] += 1
                else:
                    stats['misaligned_samples'] += 1
                
                if 'duration_diff' in alignment_result:
                    stats['duration_diffs'].append(alignment_result['duration_diff'])
                    stats['video_durations'].append(alignment_result['video_duration'])
                    stats['audio_durations'].append(alignment_result['audio_duration'])
                    stats['fps_values'].append(alignment_result['video_info']['fps'])
                    stats['sample_rates'].append(alignment_result['audio_info']['sample_rate'])
                
                # 打印进度
                if stats['total_samples'] % 10 == 0:
                    print(f"    已处理 {stats['total_samples']} 个样本...")
            
            if stats['total_samples'] >= max_samples:
                break
        
        if stats['total_samples'] >= max_samples:
            break
    
    return {
        'results': results,
        'stats': stats
    }

def print_summary(analysis_result: Dict):
    """打印分析摘要"""
    stats = analysis_result['stats']
    
    print("\n" + "=" * 60)
    print("音视频对齐分析摘要")
    print("=" * 60)
    
    print(f"总样本数: {stats['total_samples']}")
    print(f"对齐样本数: {stats['aligned_samples']} ({stats['aligned_samples']/stats['total_samples']*100:.1f}%)")
    print(f"未对齐样本数: {stats['misaligned_samples']} ({stats['misaligned_samples']/stats['total_samples']*100:.1f}%)")
    print(f"加载失败样本数: {stats['failed_samples']} ({stats['failed_samples']/stats['total_samples']*100:.1f}%)")
    
    if stats['duration_diffs']:
        print(f"\n时长差异统计:")
        print(f"  平均差异: {np.mean(stats['duration_diffs']):.3f} 秒")
        print(f"  最大差异: {np.max(stats['duration_diffs']):.3f} 秒")
        print(f"  最小差异: {np.min(stats['duration_diffs']):.3f} 秒")
        print(f"  标准差: {np.std(stats['duration_diffs']):.3f} 秒")
    
    if stats['video_durations']:
        print(f"\n视频时长统计:")
        print(f"  平均时长: {np.mean(stats['video_durations']):.3f} 秒")
        print(f"  最大时长: {np.max(stats['video_durations']):.3f} 秒")
        print(f"  最小时长: {np.min(stats['video_durations']):.3f} 秒")
    
    if stats['audio_durations']:
        print(f"\n音频时长统计:")
        print(f"  平均时长: {np.mean(stats['audio_durations']):.3f} 秒")
        print(f"  最大时长: {np.max(stats['audio_durations']):.3f} 秒")
        print(f"  最小时长: {np.min(stats['audio_durations']):.3f} 秒")
    
    if stats['fps_values']:
        print(f"\n视频帧率统计:")
        unique_fps = list(set(stats['fps_values']))
        print(f"  唯一帧率值: {unique_fps}")
        print(f"  平均帧率: {np.mean(stats['fps_values']):.2f} fps")
    
    if stats['sample_rates']:
        print(f"\n音频采样率统计:")
        unique_sr = list(set(stats['sample_rates']))
        print(f"  唯一采样率值: {unique_sr}")
        print(f"  平均采样率: {np.mean(stats['sample_rates']):.1f} Hz")

def print_misaligned_samples(analysis_result: Dict, max_print: int = 10):
    """打印未对齐的样本详情"""
    results = analysis_result['results']
    misaligned = [r for r in results if not r.get('aligned', True) and 'error' not in r]
    
    if not misaligned:
        print("\n所有样本都对齐良好！")
        return
    
    print(f"\n未对齐样本详情 (显示前 {min(max_print, len(misaligned))} 个):")
    print("-" * 80)
    
    for i, sample in enumerate(misaligned[:max_print]):
        print(f"{i+1}. {Path(sample['video_path']).name}")
        print(f"   视频时长: {sample['video_duration']:.3f} 秒")
        print(f"   音频时长: {sample['audio_duration']:.3f} 秒")
        print(f"   时长差异: {sample['duration_diff']:.3f} 秒")
        print(f"   演员: {sample['actor']}")
        print()

def create_visualization(analysis_result: Dict, output_dir: str = "alignment_analysis"):
    """创建可视化图表"""
    stats = analysis_result['stats']
    
    if not stats['duration_diffs']:
        print("没有足够的数据进行可视化")
        return
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 时长差异分布
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(stats['duration_diffs'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('时长差异 (秒)')
    plt.ylabel('样本数量')
    plt.title('音视频时长差异分布')
    plt.axvline(x=0.1, color='red', linestyle='--', label='容忍度 (0.1秒)')
    plt.legend()
    
    # 2. 视频时长分布
    plt.subplot(2, 2, 2)
    plt.hist(stats['video_durations'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('视频时长 (秒)')
    plt.ylabel('样本数量')
    plt.title('视频时长分布')
    
    # 3. 音频时长分布
    plt.subplot(2, 2, 3)
    plt.hist(stats['audio_durations'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('音频时长 (秒)')
    plt.ylabel('样本数量')
    plt.title('音频时长分布')
    
    # 4. 对齐情况饼图
    plt.subplot(2, 2, 4)
    labels = ['对齐', '未对齐', '加载失败']
    sizes = [stats['aligned_samples'], stats['misaligned_samples'], stats['failed_samples']]
    colors = ['lightgreen', 'lightcoral', 'lightgray']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('音视频对齐情况')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'alignment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {output_dir / 'alignment_analysis.png'}")

def main():
    parser = argparse.ArgumentParser(description='检查数据集中音视频对齐情况')
    parser.add_argument('--data-root', type=str, default='../datasets/ravdess_processed', 
                       help='数据根目录')
    parser.add_argument('--max-samples', type=int, default=100, 
                       help='最大检查样本数')
    parser.add_argument('--tolerance', type=float, default=0.1, 
                       help='时长差异容忍度 (秒)')
    parser.add_argument('--output-dir', type=str, default='alignment_analysis', 
                       help='输出目录')
    parser.add_argument('--visualize', action='store_true', 
                       help='生成可视化图表')
    
    args = parser.parse_args()
    
    # 分析数据集
    analysis_result = analyze_dataset(args.data_root, args.max_samples)
    
    # 打印摘要
    print_summary(analysis_result)
    
    # 打印未对齐样本
    print_misaligned_samples(analysis_result)
    
    # 生成可视化
    if args.visualize:
        create_visualization(analysis_result, args.output_dir)
    
    print(f"\n分析完成！检查了 {analysis_result['stats']['total_samples']} 个样本。")

if __name__ == "__main__":
    main()
