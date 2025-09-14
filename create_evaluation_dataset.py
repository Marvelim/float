#!/usr/bin/env python3
"""
RAVDESS评估数据集创建脚本
提取预处理视频的第一帧和对应音频，生成fake视频用于评估
"""

import os
import sys
import cv2
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append('.')
from generate import InferenceAgent, InferenceOptions

class EvaluationDatasetCreator:
    """评估数据集创建器"""
    
    def __init__(self, processed_dir="./datasets/ravdess_processed", 
                 evaluation_dir="./evaluation", ckpt_path="./checkpoints/float.pth"):
        self.processed_dir = Path(processed_dir)
        self.evaluation_dir = Path(evaluation_dir)
        self.ckpt_path = ckpt_path
        
        # 创建评估目录结构
        self.real_dir = self.evaluation_dir / "real"
        self.fake_dir = self.evaluation_dir / "fake"
        
        # 初始化推理代理
        self.init_inference_agent()
    
    def init_inference_agent(self):
        """初始化FLOAT推理代理"""
        try:
            # 创建推理选项
            import sys
            original_argv = sys.argv.copy()
            sys.argv = ['create_evaluation_dataset.py', 
                       '--ckpt_path', str(self.ckpt_path),
                       '--nfe', '10', 
                       '--seed', '25']
            
            opt = InferenceOptions().parse()
            opt.rank, opt.ngpus = 0, 1
            
            # 恢复原始argv
            sys.argv = original_argv
            
            # 初始化推理代理
            self.agent = InferenceAgent(opt)
            print("✅ FLOAT推理代理初始化成功")
            
        except Exception as e:
            print(f"❌ FLOAT推理代理初始化失败: {e}")
            self.agent = None
    
    def setup_evaluation_dirs(self, actor_ids):
        """设置评估目录结构"""
        print("📁 设置评估目录结构...")
        
        for actor_id in actor_ids:
            actor_real_dir = self.real_dir / f"Actor_{actor_id:02d}"
            actor_fake_dir = self.fake_dir / f"Actor_{actor_id:02d}"
            
            actor_real_dir.mkdir(parents=True, exist_ok=True)
            actor_fake_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ 评估目录结构已创建: {self.evaluation_dir}")
    
    def extract_first_frame(self, video_path, output_path):
        """从视频中提取第一帧"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 保存为图片
                cv2.imwrite(str(output_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                return True
            else:
                print(f"❌ 无法读取视频第一帧: {video_path}")
                return False
                
        except Exception as e:
            print(f"❌ 提取第一帧失败: {e}")
            return False
    
    def copy_real_video(self, source_video, target_dir, filename):
        """复制real视频到评估目录"""
        try:
            target_path = target_dir / filename
            shutil.copy2(source_video, target_path)
            return True
        except Exception as e:
            print(f"❌ 复制real视频失败: {e}")
            return False
    
    def generate_fake_video(self, ref_image_path, audio_path, output_video_path):
        """使用FLOAT生成fake视频"""
        if self.agent is None:
            print("❌ 推理代理未初始化")
            return False

        try:
            # 恢复原始参数来重现问题
            result_path = self.agent.run_inference(
                res_video_path=str(output_video_path),
                ref_path=str(ref_image_path),
                audio_path=str(audio_path),
                a_cfg_scale=2.0,  # 恢复原始CFG设置
                r_cfg_scale=1.0,
                e_cfg_scale=1.0,
                emo='S2E',        # 恢复原始情感标签
                nfe=10,
                no_crop=False,
                seed=25,
                verbose=False
            )

            return os.path.exists(result_path)

        except Exception as e:
            print(f"❌ 生成fake视频失败: {e}")
            return False
    
    def process_actor(self, actor_id):
        """处理单个Actor的数据"""
        actor_processed_dir = self.processed_dir / "train" / f"Actor_{actor_id:02d}"
        actor_real_dir = self.real_dir / f"Actor_{actor_id:02d}"
        actor_fake_dir = self.fake_dir / f"Actor_{actor_id:02d}"
        
        if not actor_processed_dir.exists():
            print(f"⚠️  Actor_{actor_id:02d} 预处理目录不存在，跳过")
            return 0, 0
        
        # 获取所有处理过的视频文件
        video_files = list(actor_processed_dir.glob("*_processed.mp4"))
        
        if not video_files:
            print(f"⚠️  Actor_{actor_id:02d} 没有处理过的视频文件")
            return 0, 0
        
        print(f"\n🎬 处理 Actor_{actor_id:02d} ({len(video_files)}个视频)...")
        
        success_count = 0
        failed_count = 0
        
        for video_file in tqdm(video_files, desc=f"Actor_{actor_id:02d}"):
            try:
                # 对应的音频文件
                audio_file = video_file.with_suffix('.wav')
                
                if not audio_file.exists():
                    print(f"⚠️  音频文件不存在: {audio_file.name}")
                    failed_count += 1
                    continue
                
                # 生成文件名（去掉_processed后缀）
                base_name = video_file.stem.replace('_processed', '')
                
                # Real视频路径
                real_video_path = actor_real_dir / f"{base_name}_real.mp4"
                
                # Fake视频路径  
                fake_video_path = actor_fake_dir / f"{base_name}_fake.mp4"
                
                # 检查是否已经处理过
                if real_video_path.exists() and fake_video_path.exists():
                    success_count += 1
                    continue
                
                # 1. 复制real视频
                if not real_video_path.exists():
                    if not self.copy_real_video(video_file, actor_real_dir, real_video_path.name):
                        failed_count += 1
                        continue
                
                # 2. 提取第一帧作为参考图像
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_ref:
                    temp_ref_path = temp_ref.name
                
                try:
                    if not self.extract_first_frame(video_file, temp_ref_path):
                        failed_count += 1
                        continue
                    
                    # 3. 生成fake视频
                    if not fake_video_path.exists():
                        if self.generate_fake_video(temp_ref_path, audio_file, fake_video_path):
                            success_count += 1
                        else:
                            failed_count += 1
                    else:
                        success_count += 1
                        
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_ref_path):
                        os.unlink(temp_ref_path)
                        
            except Exception as e:
                print(f"❌ 处理文件失败 {video_file.name}: {e}")
                failed_count += 1
        
        return success_count, failed_count
    
    def create_evaluation_dataset(self, actor_ids=None):
        """创建评估数据集"""
        if actor_ids is None:
            actor_ids = [23, 24]  # 默认处理Actor 23和24

        print("🎯 开始创建RAVDESS评估数据集")
        print(f"Actor IDs: {actor_ids}")
        print(f"预处理目录: {self.processed_dir}")
        print(f"评估目录: {self.evaluation_dir}")
        print(f"模型检查点: {self.ckpt_path}")
        print("=" * 60)

        # 设置目录结构
        self.setup_evaluation_dirs(actor_ids)

        total_success = 0
        total_failed = 0

        for actor_id in actor_ids:
            success, failed = self.process_actor(actor_id)
            total_success += success
            total_failed += failed

            # 显示当前进度
            print(f"✅ Actor_{actor_id:02d} 完成: 成功 {success}, 失败 {failed}")

        print(f"\n🎉 评估数据集创建完成!")
        print(f"总计成功处理: {total_success} 个视频对")
        print(f"总计处理失败: {total_failed} 个视频对")
        print(f"成功率: {total_success/(total_success+total_failed)*100:.1f}%" if (total_success+total_failed) > 0 else "N/A")
        print(f"📁 Real视频目录: {self.real_dir}")
        print(f"📁 Fake视频目录: {self.fake_dir}")

        return total_success, total_failed


def main():
    parser = argparse.ArgumentParser(description='创建RAVDESS评估数据集')
    parser.add_argument('--actors', type=str, default='23,24',
                       help='要处理的Actor ID (逗号分隔)')
    parser.add_argument('--processed_dir', type=str, default='./datasets/ravdess_processed',
                       help='预处理数据目录')
    parser.add_argument('--evaluation_dir', type=str, default='./evaluation',
                       help='评估数据输出目录')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/float.pth',
                       help='FLOAT模型检查点路径')
    
    args = parser.parse_args()
    
    # 解析actor IDs
    if args.actors.lower() == 'all':
        actor_ids = list(range(1, 25))
    else:
        actor_ids = [int(x.strip()) for x in args.actors.split(',')]
    
    # 创建评估数据集创建器
    creator = EvaluationDatasetCreator(
        processed_dir=args.processed_dir,
        evaluation_dir=args.evaluation_dir,
        ckpt_path=args.ckpt_path
    )
    
    # 创建评估数据集
    success_count, failed_count = creator.create_evaluation_dataset(actor_ids)

    if failed_count == 0:
        print("✅ 所有视频处理成功")
    else:
        print(f"⚠️  {failed_count} 个视频处理失败，{success_count} 个视频处理成功")


if __name__ == "__main__":
    main()
