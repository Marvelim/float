#!/usr/bin/env python3
"""
综合视频质量评估脚本
支持自动计算 FID 和 FVD 指标，并生成汇总表格

使用方法:
python compute_metrics.py --data_dir evaluation/sampled_data --output results.json

数据目录结构:
data_dir/
├── fid/
│   ├── real/        # 真实图片帧
│   └── fake/        # 生成图片帧
└── fvd/
    ├── real/        # 真实视频clips
    └── fake/        # 生成视频clips
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 添加依赖库路径
sys.path.append('./utils/dependencies/common_metrics_on_video_quality')

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("警告: insightface 库不可用，CSIM 指标将被跳过")

class MetricsCalculator:
    """视频质量指标计算器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化 InsightFace 模型
        self.face_app = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(512, 512))
                print("✅ InsightFace 模型初始化成功")
            except Exception as e:
                print(f"❌ InsightFace 模型初始化失败: {e}")
                self.face_app = None
        
    def load_images_from_dir(self, image_dir):
        """从目录加载图片"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
        
        # 获取所有图片文件
        image_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        
        if not image_files:
            raise ValueError(f"目录中没有找到图片文件: {image_dir}")
        
        images = []
        print(f"加载 {len(image_files)} 张图片...")
        
        for img_path in tqdm(image_files, desc="加载图片"):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 归一化到 [0, 1]
                img = img.astype(np.float32) / 255.0
                images.append(img)
        
        return np.array(images)
    
    def load_videos_from_dir(self, video_dir):
        """从目录加载视频clips"""
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
        
        # 获取所有视频文件（递归搜索子目录）
        video_files = sorted(list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.avi")))
        
        if not video_files:
            raise ValueError(f"目录中没有找到视频文件: {video_dir}")
        
        videos = []
        print(f"加载 {len(video_files)} 个视频clips...")
        
        for video_path in tqdm(video_files, desc="加载视频"):
            frames = self._load_video_frames(video_path)
            if len(frames) > 0:
                videos.append(frames)
        
        if not videos:
            raise ValueError(f"没有成功加载任何视频: {video_dir}")
        
        # 转换为 tensor: [batch_size, timestamps, channel, h, w]
        # 确保所有视频都是 tensor 格式
        video_tensors = []
        for video in videos:
            if isinstance(video, np.ndarray):
                video_tensors.append(torch.from_numpy(video))
            else:
                video_tensors.append(video)
        
        videos_tensor = torch.stack(video_tensors)
        return videos_tensor
    
    def _load_video_frames(self, video_path):
        """加载单个视频的所有帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 归一化到 [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        if frames:
            # 转换为 tensor: [timestamps, h, w, channel] -> [timestamps, channel, h, w]
            frames_array = np.array(frames)
            frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
            return frames_tensor  # 返回 tensor
        else:
            return torch.empty(0)
    
    def calculate_fid_pytorch(self, real_dir, fake_dir):
        """使用 pytorch-fid 直接计算目录中的 FID"""
        try:
            from pytorch_fid import fid_score
            
            # 直接使用目录计算 FID
            fid_value = fid_score.calculate_fid_given_paths(
                [str(real_dir), str(fake_dir)],
                batch_size=50,
                device=self.device,
                dims=2048
            )
            
            return fid_value
                
        except ImportError:
            print("警告: pytorch-fid 库不可用")
            return None
    
    def _calculate_fid_manual(self, real_images, fake_images):
        """手动实现 FID 计算"""
        try:
            from torchvision.models import inception_v3
            import torch.nn.functional as F
            
            # 加载 Inception 模型
            model = inception_v3(pretrained=True, transform_input=False)
            model.fc = torch.nn.Identity()  # 移除最后的分类层
            model = model.to(self.device)
            model.eval()
            
            def get_features(images):
                """提取特征"""
                features = []
                batch_size = 32
                
                for i in range(0, len(images), batch_size):
                    batch = images[i:i+batch_size]
                    # 调整图片大小到 299x299 (Inception 输入要求)
                    batch_resized = F.interpolate(
                        torch.from_numpy(batch).permute(0, 3, 1, 2).float(),
                        size=(299, 299),
                        mode='bilinear',
                        align_corners=False
                    ).to(self.device)
                    
                    with torch.no_grad():
                        feat = model(batch_resized)
                        features.append(feat.cpu().numpy())
                
                return np.concatenate(features, axis=0)
            
            # 提取特征
            real_features = get_features(real_images)
            fake_features = get_features(fake_images)
            
            # 计算 FID
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            
            # 计算 Frechet 距离
            diff = mu_real - mu_fake
            covmean = self._sqrtm(sigma_real.dot(sigma_fake))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
            return float(fid)
            
        except Exception as e:
            print(f"FID 计算失败: {e}")
            return None
    
    def _sqrtm(self, matrix):
        """计算矩阵的平方根"""
        from scipy.linalg import sqrtm
        return sqrtm(matrix)
    
    def calculate_fvd(self, real_videos, fake_videos, method='styleganv'):
        """计算 FVD"""
        try:
            from calculate_fvd import calculate_fvd
            
            # 确保视频长度一致
            min_length = min(real_videos.shape[1], fake_videos.shape[1])
            real_videos = real_videos[:, :min_length]
            fake_videos = fake_videos[:, :min_length]
            
            # 确保有足够的帧数
            if min_length < 10:
                print(f"警告: 视频帧数不足 ({min_length} < 10)，无法计算 FVD")
                return None
            
            result = calculate_fvd(
                real_videos.float(), 
                fake_videos.float(), 
                self.device, 
                method=method, 
                only_final=True
            )
            
            return result["value"][0] if result["value"] else None
            
        except Exception as e:
            print(f"FVD 计算失败: {e}")
            return None
    
    def calculate_csim(self, fake_video_dir, real_video_dir):
        """计算 CSIM (Cosine Similarity) 指标 - 比较对应 fake 和 real 视频的随机帧"""
        if not INSIGHTFACE_AVAILABLE or self.face_app is None:
            print("InsightFace 不可用，跳过 CSIM 计算")
            return None
        
        try:
            fake_video_dir = Path(fake_video_dir)
            real_video_dir = Path(real_video_dir)
            
            if not fake_video_dir.exists():
                raise FileNotFoundError(f"Fake视频目录不存在: {fake_video_dir}")
            if not real_video_dir.exists():
                raise FileNotFoundError(f"Real视频目录不存在: {real_video_dir}")
            
            # 获取所有fake视频文件
            fake_video_files = sorted(list(fake_video_dir.rglob("*.mp4")) + list(fake_video_dir.rglob("*.avi")))
            
            if not fake_video_files:
                raise ValueError(f"Fake目录中没有找到视频文件: {fake_video_dir}")
            
            print(f"计算 {len(fake_video_files)} 个视频对的 CSIM...")
            
            video_csim_scores = []
            
            for i, fake_video_path in enumerate(fake_video_files):
                try:
                    print(f"\n[{i+1}/{len(fake_video_files)}] 处理: {fake_video_path.name}")
                    
                    # 直接找到对应的 real 视频
                    real_video_path = self._find_corresponding_real_video(fake_video_path, real_video_dir)
                    
                    if real_video_path is None:
                        print(f"  没有找到对应的 real 视频，跳过")
                        continue
                    
                    csim_score = self._calculate_video_pair_csim(fake_video_path, real_video_path)
                    if csim_score is not None:
                        video_csim_scores.append(csim_score)
                        current_avg = sum(video_csim_scores) / len(video_csim_scores)
                        print(f"  CSIM: {csim_score:.4f} | 当前平均值: {current_avg:.4f} | 已处理: {len(video_csim_scores)}/{i+1}")
                        print(f"  对比视频: {real_video_path.name}")
                    else:
                        print(f"  CSIM: 无法计算（未检测到人脸）")
                except Exception as e:
                    print(f"  处理失败: {e}")
                    continue
            
            if not video_csim_scores:
                print("没有成功处理任何视频")
                return None
            
            # 计算所有视频的平均 CSIM
            overall_csim = np.mean(video_csim_scores)
            
            print(f"成功处理 {len(video_csim_scores)}/{len(fake_video_files)} 个视频")
            print(f"CSIM 分数分布: 最小={np.min(video_csim_scores):.4f}, "
                  f"最大={np.max(video_csim_scores):.4f}, "
                  f"标准差={np.std(video_csim_scores):.4f}")
            
            return float(overall_csim)
            
        except Exception as e:
            print(f"CSIM 计算失败: {e}")
            return None
    
    def _find_corresponding_real_video(self, fake_video_path, real_video_dir):
        """找到对应的 real 视频"""
        # 将 fake 文件名转换为 real 文件名
        fake_name = fake_video_path.name
        real_name = fake_name.replace('_fake.mp4', '_real.mp4')
        
        # 搜索对应的 real 视频
        real_video_path = None
        for real_path in real_video_dir.rglob(real_name):
            real_video_path = real_path
            break
        
        return real_video_path
    
    def _calculate_video_pair_csim(self, fake_video_path, real_video_path):
        """计算一对视频的 CSIM 分数 - 各取随机一帧比较"""
        try:
            import random
            
            # 从fake视频随机选一帧
            fake_frame = self._get_random_frame(fake_video_path)
            if fake_frame is None:
                return None
            
            # 从real视频随机选一帧
            real_frame = self._get_random_frame(real_video_path)
            if real_frame is None:
                return None
            
            # 提取fake帧的人脸特征
            fake_faces = self.face_app.get(fake_frame)
            if len(fake_faces) == 0:
                return None
            
            # 提取real帧的人脸特征
            real_faces = self.face_app.get(real_frame)
            if len(real_faces) == 0:
                return None
            
            # 选择最大的人脸
            fake_face = max(fake_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            real_face = max(real_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # 计算余弦相似度
            similarity = np.dot(fake_face.normed_embedding, real_face.normed_embedding)
            
            return float(similarity)
            
        except Exception as e:
            return None
    
    def _get_random_frame(self, video_path):
        """从视频中随机获取一帧"""
        try:
            import random
            cap = cv2.VideoCapture(str(video_path))
            
            # 获取总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None
            
            # 随机选择一帧
            random_frame_idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None
                
        except Exception as e:
            return None
    
    
    def compute_metrics(self, data_dir, compute_fid=True, compute_fvd=True, compute_csim=True):
        """计算指定的指标"""
        data_dir = Path(data_dir)
        results = {
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(data_dir),
            "device": str(self.device),
            "metrics": {}
        }
        
        print(f"开始计算指标，数据目录: {data_dir}")
        enabled_metrics = []
        if compute_fid:
            enabled_metrics.append("FID")
        if compute_fvd:
            enabled_metrics.append("FVD")
        if compute_csim:
            enabled_metrics.append("CSIM")
        print(f"启用的指标: {', '.join(enabled_metrics)}")
        
        # 计算 FID
        if compute_fid:
            fid_dir = data_dir / "fid"
            if fid_dir.exists():
                print("\n=== 计算 FID ===")
                try:
                    real_dir = fid_dir / "real"
                    fake_dir = fid_dir / "fake"
                    
                    if not (real_dir.exists() and fake_dir.exists()):
                        raise FileNotFoundError("FID 真实或生成目录不存在")
                    
                    # 统计图片数量
                    real_count = len(list(real_dir.glob("*.png")) + list(real_dir.glob("*.jpg")))
                    fake_count = len(list(fake_dir.glob("*.png")) + list(fake_dir.glob("*.jpg")))
                    
                    print(f"真实图片数量: {real_count}")
                    print(f"生成图片数量: {fake_count}")
                    
                    fid_score = self.calculate_fid_pytorch(real_dir, fake_dir)
                    results["metrics"]["fid"] = fid_score
                    print(f"FID: {fid_score:.4f}")
                    
                except Exception as e:
                    print(f"FID 计算失败: {e}")
                    results["metrics"]["fid"] = None
            else:
                print("FID 数据目录不存在，跳过 FID 计算")
        
        # 计算 FVD
        if compute_fvd:
            fvd_dir = data_dir / "fvd"
            if fvd_dir.exists():
                print("\n=== 计算 FVD ===")
                try:
                    real_videos = self.load_videos_from_dir(fvd_dir / "real")
                    fake_videos = self.load_videos_from_dir(fvd_dir / "fake")
                    
                    print(f"真实视频数量: {len(real_videos)}")
                    print(f"生成视频数量: {len(fake_videos)}")
                    print(f"视频尺寸: {real_videos.shape}")
                    
                    fvd_score = self.calculate_fvd(real_videos, fake_videos)
                    results["metrics"]["fvd"] = fvd_score
                    print(f"FVD: {fvd_score:.4f}" if fvd_score else "FVD: 计算失败")
                    
                except Exception as e:
                    print(f"FVD 计算失败: {e}")
                    results["metrics"]["fvd"] = None
            else:
                print("FVD 数据目录不存在，跳过 FVD 计算")
        
        # 计算 CSIM (比较 evaluation/fake 和 evaluation/real)
        if compute_csim:
            fake_csim_dir = Path("/root/autodl-tmp/float/evaluation/fake")
            real_csim_dir = Path("/root/autodl-tmp/float/evaluation/real")
            
            if fake_csim_dir.exists() and real_csim_dir.exists():
                print("\n=== 计算 CSIM ===")
                try:
                    csim_score = self.calculate_csim(fake_csim_dir, real_csim_dir)
                    results["metrics"]["csim"] = csim_score
                    if csim_score is not None:
                        print(f"CSIM: {csim_score:.4f}")
                    else:
                        print("CSIM: 计算失败")
                    
                except Exception as e:
                    print(f"CSIM 计算失败: {e}")
                    results["metrics"]["csim"] = None
            else:
                print("CSIM 视频目录不存在，跳过 CSIM 计算")
                print(f"期望fake路径: {fake_csim_dir}")
                print(f"期望real路径: {real_csim_dir}")
        
        return results
    
    def create_summary_table(self, results_list):
        """创建汇总表格"""
        if not results_list:
            return None
        
        # 准备表格数据
        table_data = []
        for i, result in enumerate(results_list):
            row = {
                "序号": i + 1,
                "数据目录": result.get("data_dir", "N/A"),
                "计算时间": result.get("timestamp", "N/A")[:19],  # 只保留日期时间部分
                "FID": result.get("metrics", {}).get("fid", "N/A"),
                "FVD": result.get("metrics", {}).get("fvd", "N/A"),
                "CSIM": result.get("metrics", {}).get("csim", "N/A")
            }
            table_data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(table_data)
        
        # 计算统计信息
        fid_values = [r.get("metrics", {}).get("fid") for r in results_list 
                     if r.get("metrics", {}).get("fid") is not None]
        fvd_values = [r.get("metrics", {}).get("fvd") for r in results_list 
                     if r.get("metrics", {}).get("fvd") is not None]
        csim_values = [r.get("metrics", {}).get("csim") for r in results_list 
                      if r.get("metrics", {}).get("csim") is not None]
        
        stats = {}
        if fid_values:
            stats["FID"] = {
                "平均值": np.mean(fid_values),
                "标准差": np.std(fid_values),
                "最小值": np.min(fid_values),
                "最大值": np.max(fid_values)
            }
        
        if fvd_values:
            stats["FVD"] = {
                "平均值": np.mean(fvd_values),
                "标准差": np.std(fvd_values),
                "最小值": np.min(fvd_values),
                "最大值": np.max(fvd_values)
            }
        
        if csim_values:
            stats["CSIM"] = {
                "平均值": np.mean(csim_values),
                "标准差": np.std(csim_values),
                "最小值": np.min(csim_values),
                "最大值": np.max(csim_values)
            }
        
        return df, stats


def main():
    parser = argparse.ArgumentParser(description='计算视频质量评估指标')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径 (包含 fid/ 和 fvd/ 子目录)')
    parser.add_argument('--output', type=str, default='metrics_results.json',
                       help='结果输出文件')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--table', action='store_true',
                       help='生成汇总表格')
    
    # 指标选择参数
    parser.add_argument('--fid', action='store_true',
                       help='计算 FID 指标')
    parser.add_argument('--fvd', action='store_true',
                       help='计算 FVD 指标')
    parser.add_argument('--csim', action='store_true',
                       help='计算 CSIM 指标')
    
    args = parser.parse_args()
    
    # 如果没有指定任何指标，默认计算所有指标
    if not (args.fid or args.fvd or args.csim):
        compute_fid = compute_fvd = compute_csim = True
        print("未指定具体指标，将计算所有可用指标")
    else:
        compute_fid = args.fid
        compute_fvd = args.fvd
        compute_csim = args.csim
    
    # 创建计算器
    calculator = MetricsCalculator(device=args.device)
    
    # 计算指标
    results = calculator.compute_metrics(args.data_dir, compute_fid, compute_fvd, compute_csim)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {args.output}")
    
    # 打印结果摘要
    print("\n=== 结果摘要 ===")
    for metric, value in results["metrics"].items():
        if value is not None:
            print(f"{metric.upper()}: {value:.4f}")
        else:
            print(f"{metric.upper()}: 计算失败")
    
    # 生成表格
    if args.table:
        df, stats = calculator.create_summary_table([results])
        
        print("\n=== 指标表格 ===")
        print(df.to_string(index=False))
        
        if stats:
            print("\n=== 统计信息 ===")
            for metric, stat_dict in stats.items():
                print(f"\n{metric}:")
                for stat_name, value in stat_dict.items():
                    print(f"  {stat_name}: {value:.4f}")


if __name__ == "__main__":
    main()
