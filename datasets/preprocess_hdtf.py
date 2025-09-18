#!/usr/bin/env python3
"""
HDTF 数据集预处理脚本
支持处理HDTF数据集中的视频文件，从视频中分离音频并进行标准预处理
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import hashlib
import argparse
import cv2
import librosa
import soundfile as sf
import numpy as np
import face_alignment
import tempfile
import subprocess
import shutil

# 添加父目录到路径以导入项目模块
sys.path.append('..')
try:
    from options.base_options import BaseOptions
    from generate import DataProcessor
except ImportError:
    print("Warning: Could not import project modules. Some features may be unavailable.")

class HDTFPreprocessor:
    """HDTF数据预处理器"""
    
    def __init__(self, raw_dir="./hdtf_raw", processed_dir="./hdtf_preprocessed", 
                 target_fps=25, target_sr=16000, target_size=512):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_fps = target_fps
        self.target_sr = target_sr
        self.target_size = target_size
        # 初始化人脸检测器
        try:
            self.fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, 
                flip_input=False,
                    device='cuda' if os.system('nvidia-smi') == 0 else 'cpu'
                )
        except Exception as e:
            print(f"Warning: Could not initialize face alignment: {e}")
            self.fa = None
    
    def get_crop_params_like_generate(self, img):
        """按照generate.py中process_img的方式计算裁剪参数"""
        if self.fa is None:
            # 如果没有人脸检测器，返回中心区域裁剪参数
            h, w = img.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            return {
                'type': 'center',
                'y': y, 'x': x, 'size': size,
                'bs': 0, 'my': 0, 'mx': 0
            }
        
        try:
            # 缩放图像以提高检测速度（与generate.py完全一致）
            mult = 360. / img.shape[0]
            resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult, 
                                   interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
            
            # 检测人脸
            bboxes = self.fa.face_detector.detect_from_image(resized_img)
            # 过滤置信度大于0.95的检测结果（与generate.py一致）
            bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) 
                     for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
            
            if len(bboxes) == 0:
                # 没有检测到高置信度人脸，返回中心区域裁剪参数
                h, w = img.shape[:2]
                size = min(h, w)
                y = (h - size) // 2
                x = (w - size) // 2
                return {
                    'type': 'center',
                    'y': y, 'x': x, 'size': size,
                    'bs': 0, 'my': 0, 'mx': 0
                }
            
            # 使用第一个检测到的人脸（与generate.py一致）
            bboxes = bboxes[0]
            
            # 计算人脸中心和半尺寸（与generate.py完全一致）
            bsy = int((bboxes[3] - bboxes[1]) / 2)
            bsx = int((bboxes[2] - bboxes[0]) / 2)
            my = int((bboxes[1] + bboxes[3]) / 2)
            mx = int((bboxes[0] + bboxes[2]) / 2)
            
            # 计算扩展尺寸（与generate.py完全一致）
            bs = int(max(bsy, bsx) * 1.6)
            
            return {
                'type': 'face',
                'bs': bs, 'my': my, 'mx': mx,
                'mult': mult
            }
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            # 返回中心区域裁剪参数
            h, w = img.shape[:2]
            size = min(h, w)
            y = (h - size) // 2
            x = (w - size) // 2
            return {
                'type': 'center',
                'y': y, 'x': x, 'size': size,
                'bs': 0, 'my': 0, 'mx': 0
            }
    
    def apply_crop_params(self, img, crop_params):
        """根据裁剪参数处理图像"""
        if crop_params['type'] == 'center':
            # 中心裁剪
            y, x, size = crop_params['y'], crop_params['x'], crop_params['size']
            crop_img = img[y:y+size, x:x+size]
            return cv2.resize(crop_img, (self.target_size, self.target_size))
        
        elif crop_params['type'] == 'face':
            # 人脸裁剪（与generate.py完全一致）
            bs, my, mx = crop_params['bs'], crop_params['my'], crop_params['mx']
            mult = crop_params['mult']
            
            # 添加边框（与generate.py完全一致）
            img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
            my, mx = my + bs, mx + bs  # 更新中心坐标
            
            # 裁剪正方形区域（与generate.py完全一致）
            crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
            
            # 调整到目标尺寸（与generate.py完全一致）
            crop_img = cv2.resize(crop_img, dsize=(self.target_size, self.target_size), 
                                interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
            
            return crop_img
    
    def extract_audio_from_video(self, video_path, audio_path):
        """从视频中提取音频"""
        try:
            # 使用ffmpeg从视频中提取音频
            extract_cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vn',  # 不处理视频流
                '-acodec', 'pcm_s16le',  # 使用PCM编码
                '-ar', str(self.target_sr),  # 设置采样率
                '-ac', '1',  # 单声道
                str(audio_path), '-y'
            ]
            
            with open(os.devnull, 'wb') as devnull:
                result = subprocess.run(extract_cmd, stdout=devnull, stderr=devnull)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"音频提取失败: {e}")
            return False

    def process_separate_audio(self, input_path, output_path):
        """处理单独的音频文件"""
        try:
            # 加载音频
            audio_data, sr = librosa.load(str(input_path), sr=self.target_sr)
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存处理后的音频
            sf.write(str(output_path), audio_data, self.target_sr)
                
            return True
            
        except Exception as e:
            print(f"❌ 音频处理失败: {e}")
            return False

    def validate_video_file(self, video_path):
        """验证视频文件是否完整可用"""
        try:
            # 使用ffmpeg检查视频文件
            check_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', str(video_path)
            ]
            
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"ffprobe检查失败: {result.stderr}"
            
            # 使用OpenCV进行二次验证
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                cap.release()
                return False, "OpenCV无法打开视频文件"
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if frame_count == 0 or fps == 0:
                cap.release()
                return False, f"视频元数据无效: 帧数={frame_count}, FPS={fps}"
            
            # 尝试读取第一帧
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False, "无法读取视频帧"
            
            return True, "视频文件验证通过"
            
        except Exception as e:
            return False, f"视频验证异常: {str(e)}"

    def process_hdtf_video(self, video_path, output_path):
        """处理HDTF视频文件，从视频中分离音频"""
        try:
            # 首先验证视频文件
            is_valid, validation_msg = self.validate_video_file(video_path)
            if not is_valid:
                print(f"❌ 视频文件损坏或无效: {video_path}")
                print(f"   原因: {validation_msg}")
                return False
            
            # 读取视频
            cap = cv2.VideoCapture(str(video_path))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                print(f"❌ 无法读取视频: {video_path}")
                cap.release()
                return False
            
            # 读取第一帧用于人脸检测
            ret, first_frame = cap.read()
            if not ret:
                print(f"❌ 无法读取视频帧: {video_path}")
                cap.release()
                return False
            
            # 计算裁剪参数（只在第一帧计算一次）
            crop_params = self.get_crop_params_like_generate(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建临时视频文件（无音频）
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = temp_video.name
            
            # 设置视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, self.target_fps, 
                                (self.target_size, self.target_size))
            
            # 计算帧采样策略以保持时长一致
            frame_ratio = original_fps / self.target_fps
            target_frame_count = int(frame_count / frame_ratio)
            
            # 处理每一帧（使用第一帧计算的裁剪参数）
            processed_frames = 0
            frame_indices = [int(i * frame_ratio) for i in range(target_frame_count)]
            
            for target_frame_idx in frame_indices:
                # 跳转到目标帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 应用裁剪参数（所有帧使用相同的裁剪参数）
                processed_frame = self.apply_crop_params(frame, crop_params)
                
                # 写入帧
                out.write(processed_frame)
                processed_frames += 1
            
            cap.release()
            out.release()
            
            # 从视频中提取音频
            temp_audio_path = output_path.with_suffix('.wav')
            audio_extracted = self.extract_audio_from_video(video_path, temp_audio_path)
            
            if processed_frames > 0 and audio_extracted:
                # 合并视频和音频
                merge_success = self.merge_video_audio(temp_video_path, 
                                                     temp_audio_path,
                                                     output_path)
                
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                
                if merge_success:
                    print(f"✅ 视频处理完成: {processed_frames} 帧, {original_fps:.1f}fps → {self.target_fps}fps")
                    return True
                else:
                    print(f"❌ 视频音频合并失败")
                    return False
            else:
                print(f"❌ 视频或音频处理失败")
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                return False
                
        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
            return False

    def process_video_with_separate_audio(self, video_path, audio_path, output_path):
        """使用单独的高质量音频处理视频"""
        try:
            # 读取视频
            cap = cv2.VideoCapture(str(video_path))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                print(f"❌ 无法读取视频: {video_path}")
                return False
            
            # 读取第一帧用于人脸检测
            ret, first_frame = cap.read()
            if not ret:
                print(f"❌ 无法读取视频帧: {video_path}")
                cap.release()
                return False
            
            # 计算裁剪参数（只在第一帧计算一次）
            crop_params = self.get_crop_params_like_generate(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
            
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建临时视频文件（无音频）
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video_path = temp_video.name
            
            # 设置视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, self.target_fps, 
                                (self.target_size, self.target_size))
            
            # 计算帧采样策略以保持时长一致
            frame_ratio = original_fps / self.target_fps
            target_frame_count = int(frame_count / frame_ratio)
            
            # 处理每一帧（使用第一帧计算的裁剪参数）
            processed_frames = 0
            frame_indices = [int(i * frame_ratio) for i in range(target_frame_count)]
            
            for target_frame_idx in frame_indices:
                # 跳转到目标帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 应用裁剪参数（所有帧使用相同的裁剪参数）
                processed_frame = self.apply_crop_params(frame, crop_params)
                
                # 写入帧
                out.write(processed_frame)
                processed_frames += 1
            
            cap.release()
            out.release()
            
            # 处理音频
            audio_success = self.process_separate_audio(audio_path, output_path.with_suffix('.wav'))
            
            if processed_frames > 0 and audio_success:
                # 合并视频和音频
                merge_success = self.merge_video_audio(temp_video_path, 
                                                     output_path.with_suffix('.wav'),
                                                     output_path)
                
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                
                if merge_success:
                    print(f"✅ 视频处理完成: {processed_frames} 帧, {original_fps:.1f}fps → {self.target_fps}fps")
                    return True
                else:
                    print(f"❌ 视频音频合并失败")
                    return False
            else:
                print(f"❌ 视频或音频处理失败")
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                return False
                
        except Exception as e:
            print(f"❌ 视频处理失败: {e}")
            return False

    def merge_video_audio(self, video_path, audio_path, output_path):
        """合并视频和音频"""
        try:
            merge_cmd = [
                'ffmpeg', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac',
                str(output_path), '-y'
            ]
            
            with open(os.devnull, 'wb') as devnull:
                result = subprocess.run(merge_cmd, stdout=devnull, stderr=devnull)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"合并失败: {e}")
            return False
            
    def preprocess_dataset(self, file_patterns=None):
        """预处理整个HDTF数据集"""
        if file_patterns is None:
            file_patterns = ['RD_*', 'WDA_*', 'WRA_*']  # 默认处理所有类型的文件
        
        print(f"开始预处理 HDTF 数据集")
        print(f"文件模式: {file_patterns}")
        print(f"目标FPS: {self.target_fps}")
        print(f"目标采样率: {self.target_sr} Hz")
        print(f"目标分辨率: {self.target_size}x{self.target_size}")
        print("=" * 60)
        
        total_processed = 0
        total_failed = 0
        total_corrupted = 0
        corrupted_files = []
        
        # 获取所有匹配的视频文件
        video_files = []
        for pattern in file_patterns:
            video_files.extend(list(self.raw_dir.glob(f"{pattern}.mp4")))
        
        if not video_files:
            print(f"❌ 在 {self.raw_dir} 中未找到匹配的视频文件")
            return 0, 0
        
        print(f"找到 {len(video_files)} 个视频文件")
        
        for video_file in tqdm(video_files, desc="处理HDTF视频"):
            # 设置输出路径
            output_dir = self.processed_dir / "test"
            output_video_path = output_dir / f"{video_file.stem}_processed.mp4"
            
            # 检查是否已处理
            if output_video_path.exists():
                continue
            
            # 首先验证视频文件
            is_valid, validation_msg = self.validate_video_file(video_file)
            if not is_valid:
                total_corrupted += 1
                corrupted_files.append((video_file.name, validation_msg))
                print(f"⚠️  跳过损坏的视频文件: {video_file.name}")
                continue
            
            # 使用新的HDTF处理方法
            success = self.process_hdtf_video(video_file, output_video_path)
            
            if success:
                total_processed += 1
            else:
                total_failed += 1
        
        print(f"\n预处理完成!")
        print(f"成功处理: {total_processed} 个文件")
        print(f"处理失败: {total_failed} 个文件")
        print(f"损坏文件: {total_corrupted} 个文件")
        
        if corrupted_files:
            print(f"\n损坏的视频文件列表:")
            for filename, reason in corrupted_files:
                print(f"  - {filename}: {reason}")
        
        return total_processed, total_failed


def main():
    parser = argparse.ArgumentParser(description='HDTF 数据集预处理')
    parser.add_argument('--preprocess', action='store_true', help='预处理数据集')
    parser.add_argument('--patterns', type=str, default='RD_*,WDA_*,WRA_*', 
                       help='要处理的文件模式 (逗号分隔，例如: RD_*,WDA_* 或 WRA_*)')
    parser.add_argument('--raw_dir', type=str, default='./hdtf_raw', help='原始数据目录')
    parser.add_argument('--processed_dir', type=str, default='./hdtf_preprocessed', help='处理后数据目录')
    parser.add_argument('--target_fps', type=int, default=25, help='目标帧率')
    parser.add_argument('--target_sr', type=int, default=16000, help='目标采样率')
    parser.add_argument('--target_size', type=int, default=512, help='目标分辨率')
    
    args = parser.parse_args()
    
    # 解析文件模式
    if args.patterns.lower() == 'all':
        file_patterns = ['RD_*', 'WDA_*', 'WRA_*']
    else:
        file_patterns = [pattern.strip() for pattern in args.patterns.split(',')]
    
    # 预处理数据
    if args.preprocess:
        preprocessor = HDTFPreprocessor(
            raw_dir=args.raw_dir, 
            processed_dir=args.processed_dir,
            target_fps=args.target_fps,
            target_sr=args.target_sr,
            target_size=args.target_size
        )
        success_count, failed_count = preprocessor.preprocess_dataset(file_patterns)
        
        if failed_count == 0:
            print("✅ 所有文件预处理完成")
        else:
            print(f"⚠️  {failed_count} 个文件预处理失败")
    else:
        print("请使用 --preprocess 参数来预处理数据集")
        print("示例: python download_hdtf.py --preprocess --patterns RD_*,WDA_*")


if __name__ == "__main__":
    main()