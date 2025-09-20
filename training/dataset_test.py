"""
Dataset loader for FLOAT training - Optimized Version
用于 FLOAT 训练的数据加载器 - 优化版本

This version preprocesses all model computations to avoid slow __getitem__ calls.
此版本预处理所有模型计算，避免缓慢的 __getitem__ 调用。

主要优化：
1. 将音频编码器推理移到预处理阶段
2. 将运动潜在表示提取移到预处理阶段
3. 将参考运动提取移到预处理阶段
4. 使用缓存系统避免重复预处理
5. __getitem__ 只进行简单的数据切片操作

使用方法：
```python
# 创建优化的数据加载器
dataloader = create_dataloader_optimized(
    data_root="/path/to/data",
    batch_size=8,
    train=True,
    opt=opt,  # 包含模型配置的选项
    force_preprocess=False  # 设为True强制重新预处理
)

# 检查缓存状态
check_cache_status("/path/to/data")

# 清理缓存（如果需要）
clear_cache("/path/to/data")
```

注意事项：
- 首次运行会进行预处理，可能需要较长时间
- 预处理结果会缓存到磁盘，后续运行会快速加载
- 如果数据或模型发生变化，需要设置 force_preprocess=True
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import cv2
from typing import Dict, List, Tuple, Any
import random
import pickle
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.float.generator import Generator
from models.float.FLOAT import AudioEncoder
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm
from options.base_options import BaseOptions

class FLOATDatasetOptimized(Dataset):
    """
    FLOAT 训练数据集 - 优化版本
    
    预处理所有模型计算，避免在 __getitem__ 中进行耗时操作
    """
    
    def __init__(self, 
                 data_root: str,
                 train: bool = True, 
                 opt: dict = None,
                 cache_dir: str = None,
                 force_preprocess: bool = False):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            train: 是否为训练模式
            opt: 选项
            cache_dir: 缓存目录，如果为None则使用data_root/cache
            force_preprocess: 是否强制重新预处理
        """
        self.data_root = Path(data_root)
        self.opt = opt
        self.train = train
        self.sequence_length = int(self.opt.wav2vec_sec * self.opt.fps)
        self.prev_frames = int(self.opt.num_prev_frames)
        
        # 设置缓存目录
        if cache_dir is None:
            self.cache_dir = self.data_root / "cache"
            print("LOAD cache")
        else:
            self.cache_dir = Path(cache_dir)
            print("FAILED LOAD cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # 情感标签映射
        self.emotion_labels = {
            "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
            "neutral": 4, "sad": 5, "surprise": 6
        }
        
        # 缓存文件路径
        split_name = "train" if train else "test"
        self.cache_file = self.cache_dir / f"preprocessed_{split_name}.pkl"
        
        print("cache_file: ", self.cache_file)
        # 加载或创建预处理数据
        if self.cache_file.exists() and not force_preprocess:
            print(f"Loading preprocessed data from {self.cache_file}")
            self.preprocessed_data = self._load_cache()
        else:
            # assert (False)
            print("Preprocessing data...")
            # 只清除当前数据集的缓存文件，而不是整个目录
            clear_cache(self.data_root, self.cache_dir, self.cache_file.name)
            self.preprocessed_data = self._preprocess_all_data()
            self._save_cache()
        
        print(f"Dataset initialized with {len(self.preprocessed_data)} samples")

    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载数据列表
        
        Returns:
            数据项列表
        """
        data_list = []
        
        # 假设数据结构类似于 RAVDESS 数据集
        if self.train:
            data_dir = self.data_root / "train"
        else:
            data_dir = self.data_root / "test"
        
        print(f"Looking for data in: {data_dir}")
        print(f"Data directory exists: {data_dir.exists()}")
        
        if not data_dir.exists():
            # 如果没有预处理数据，使用原始数据
            data_dir = self.data_root / "ravdess_raw"
            print(f"Fallback to raw data: {data_dir}")
        
        # 遍历所有演员文件夹
        for actor_dir in data_dir.glob("Actor_*"):
            if not actor_dir.is_dir():
                continue
                
            # 遍历每个演员的文件
            for video_file in actor_dir.glob("*_processed.mp4"):
                # 解析文件名获取情感信息
                parts = video_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_id = int(parts[2])
                    # 映射到我们的情感标签
                    emotion_map = {1: 4, 2: 4, 3: 3, 4: 5, 5: 0, 6: 2, 7: 1, 8: 6}
                    emotion = emotion_map.get(emotion_id, 4)
                else:
                    emotion = 4
                
                # 查找对应的音频文件
                # 将 _processed.mp4 替换为 _processed.wav
                audio_file = video_file.with_name(video_file.stem.replace('_processed', '_processed') + '.wav')
                if not audio_file.exists():
                    # 尝试其他可能的音频文件扩展名
                    for ext in ['.wav', '.mp3', '.flac']:
                        audio_file = video_file.with_name(video_file.stem.replace('_processed', '_processed') + ext)
                        if audio_file.exists():
                            break
                
                if audio_file.exists():
                    data_list.append({
                        'video_path': str(video_file),
                        'audio_path': str(audio_file),
                        'emotion': emotion,
                        'actor_id': actor_dir.name,
                    })
        
        return data_list

    def _preprocess_all_data(self) -> List[Dict[str, torch.Tensor]]:
        """
        预处理所有数据，包括模型推理

        Returns:
            预处理后的数据列表
        """
        # 设置设备
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for preprocessing: {device}")

        # 初始化模型（仅在预处理时使用）
        print("Initializing models for preprocessing...")
        motion_autoencoder = Generator(
            size=self.opt.input_size,
            style_dim=self.opt.dim_w,
            motion_dim=self.opt.dim_m
        ).to(device)  # 移动到CUDA

        # 修复路径问题
        wav2vec_model_path = self.opt.wav2vec_model_path
        if not os.path.exists(wav2vec_model_path):
            # 尝试相对路径
            wav2vec_model_path = f"../{self.opt.wav2vec_model_path}"
        if not os.path.exists(wav2vec_model_path):
            # 尝试绝对路径
            wav2vec_model_path = f"/home/mli374/float/{self.opt.wav2vec_model_path}"
        
        print(f"Using wav2vec model path: {wav2vec_model_path}")
        
        # 临时修改opt中的路径
        original_path = self.opt.wav2vec_model_path
        self.opt.wav2vec_model_path = wav2vec_model_path
        
        audio_encoder = AudioEncoder(self.opt).to(device)  # 移动到CUDA
        
        wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_model_path,
            local_files_only=True
        )
        
        # 恢复原始路径
        self.opt.wav2vec_model_path = original_path

        # 加载预训练权重到 motion_autoencoder
        print("Loading motion_autoencoder weights from float.pth...")
        checkpoint_path = "/home/mli374/float/checkpoints/float.pth"
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)  # 直接加载到CUDA

        with torch.no_grad():
            # 加载 motion_autoencoder 权重
            loaded_count = 0
            for name, param in motion_autoencoder.named_parameters():
                full_name = f"motion_autoencoder.{name}"
                if full_name in state_dict:
                    param.copy_(state_dict[full_name])
                    loaded_count += 1
                else:
                    print(f"! Warning: {name} not found in motion_autoencoder weights")

        print(f"Successfully loaded {loaded_count} motion_autoencoder parameters")
        del state_dict  # 释放内存

        # 将模型设置为评估模式
        motion_autoencoder.eval()
        audio_encoder.eval()
        
        # 获取原始数据列表
        raw_data_list = self._load_data_list()
        preprocessed_data = []

        print(f"Found {len(raw_data_list)} raw data samples")
        if len(raw_data_list) > 0:
            print(f"First sample: {raw_data_list[0]}")
        print(f"Preprocessing {len(raw_data_list)} samples...")

        # 启用CUDA优化
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # 优化卷积性能
            print("CUDA optimizations enabled")

        for data_item in tqdm(raw_data_list, desc="Preprocessing"):
            try:
                # 加载视频和音频
                video_frames = self._load_video(data_item['video_path'])
                audio = self._load_audio(data_item['audio_path'], wav2vec_preprocessor)
                
                num_frames = video_frames.shape[0]
                
                # 确保有足够的帧数
                if num_frames < self.sequence_length + self.prev_frames:
                    print(f"Skipping {data_item['video_path']}: insufficient frames ({num_frames})")
                    continue
                
                # 音频编码（预处理）- 使用CUDA加速
                audio = audio.to(device)
                with torch.no_grad():
                    w_audio = audio_encoder.inference(audio, seq_len=num_frames).squeeze(0)

                # 运动潜在表示提取（预处理）- 使用CUDA加速
                video_frames = video_frames.to(device)
                with torch.no_grad():
                    motion_latents = self._extract_motion_latent_batch(motion_autoencoder, video_frames)

                # 参考运动提取 - 使用CUDA加速
                reference_frame = video_frames[0:1]
                with torch.no_grad():
                    reference_motion = self._extract_motion_latent_batch(motion_autoencoder, reference_frame).squeeze(0)

                # 情感特征 - 移动到CUDA
                emotion_features = F.one_hot(torch.tensor(data_item['emotion']), num_classes=self.opt.dim_e).to(device)
                
                # 存储预处理结果（移回CPU以节省GPU内存）
                preprocessed_item = {
                    'video_frames': video_frames.cpu(),  # (T, C, H, W)
                    'motion_latents': motion_latents.cpu(),  # (T, motion_dim)
                    'audio_latents': w_audio.cpu(),  # (T, audio_dim)
                    'reference_frame': video_frames[0].cpu(),  # (C, H, W)
                    'reference_motion': reference_motion.cpu(),  # (motion_dim,)
                    'emotion_features': emotion_features.cpu(),  # 移回CPU
                    'actor_id': data_item['actor_id'],
                    'num_frames': num_frames,
                }
                
                preprocessed_data.append(preprocessed_item)

                # 定期清理CUDA缓存以防止内存溢出
                if device.type == 'cuda' and len(preprocessed_data) % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {data_item['video_path']}: {e}")
                # 清理CUDA缓存以防止内存泄漏
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue

        # 最终清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"Successfully preprocessed {len(preprocessed_data)} samples")
        return preprocessed_data

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        加载视频帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频帧张量 (T, C, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换颜色空间并调整大小
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.opt.input_size, self.opt.input_size))
            frame = frame.astype(np.float32) / 127.5 - 1.0
            frames.append(frame)
        
        cap.release()
        
        assert len(frames) > 0, f"No frames found in {video_path}"
        # 转换为张量 (T, H, W, C) -> (T, C, H, W)
        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        return video_tensor

    def _load_audio(self, audio_path: str, wav2vec_preprocessor) -> torch.Tensor:
        """
        加载音频
        
        Args:
            audio_path: 音频文件路径
            wav2vec_preprocessor: Wav2Vec预处理器
            
        Returns:
            音频张量 (1, T)
        """
        speech_array, sampling_rate = librosa.load(audio_path, sr=self.opt.sampling_rate)
        audio_tensor = wav2vec_preprocessor(
            speech_array, 
            sampling_rate=sampling_rate, 
            return_tensors='pt'
        ).input_values[0]
        
        # 确保形状是 (1, sequence_length)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor

    @torch.no_grad()
    def _extract_motion_latent_batch(self, motion_autoencoder, video_frames: torch.Tensor) -> torch.Tensor:
        """
        批量提取动作潜在表示
        
        Args:
            motion_autoencoder: 运动自编码器
            video_frames: 视频帧 (T, C, H, W)
            
        Returns:
            动作潜在表示 (T, motion_dim)
        """
        d_r, r_d_lambda, _ = motion_autoencoder.enc(video_frames, input_target=None)
        r_d_lambda = motion_autoencoder.enc.fc(d_r)
        r_d = motion_autoencoder.dec.direction(r_d_lambda)
        return r_d

    def _get_sequence_indices(self, total_frames: int) -> Tuple[int, int]:
        """
        获取序列索引
        
        Args:
            total_frames: 总帧数
            
        Returns:
            起始和结束索引
        """
        assert total_frames >= self.sequence_length + self.prev_frames
        
        if self.train:
            # 训练时随机选择起始位置
            max_start = total_frames - self.sequence_length - self.prev_frames
            start_idx = random.randint(0, max_start)
        else:
            # 验证时从中间开始
            start_idx = (total_frames - self.sequence_length - self.prev_frames) // 2
        
        end_idx = start_idx + self.sequence_length + self.prev_frames
        return start_idx, end_idx

    def _save_cache(self):
        """保存预处理缓存"""
        print(f"Saving preprocessed data to {self.cache_file}")
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.preprocessed_data, f)

    def _load_cache(self):
        """加载预处理缓存"""
        with open(self.cache_file, 'rb') as f:
            return pickle.load(f)

    def __len__(self) -> int:
        return len(self.preprocessed_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项 - 优化版本，无模型计算
        
        Args:
            idx: 数据索引
            
        Returns:
            数据字典
        """
        data_item = self.preprocessed_data[idx]
        
        # 获取序列索引
        start_idx, end_idx = self._get_sequence_indices(data_item['num_frames'])
        
        # 从预处理数据中提取序列
        video_cur = data_item['video_frames'][start_idx + self.prev_frames:end_idx]
        video_prev = data_item['video_frames'][start_idx:start_idx + self.prev_frames]
        motion_latent_cur = data_item['motion_latents'][start_idx + self.prev_frames:end_idx]
        motion_latent_prev = data_item['motion_latents'][start_idx:start_idx + self.prev_frames]
        audio_latent_cur = data_item['audio_latents'][start_idx + self.prev_frames:end_idx]
        audio_latent_prev = data_item['audio_latents'][start_idx:start_idx + self.prev_frames]
        
        return {
            'video_cur': video_cur,  # (T, C, H, W)
            'video_prev': video_prev,
            'motion_latent_cur': motion_latent_cur,  # (T, motion_dim)
            'motion_latent_prev': motion_latent_prev,  # (T, motion_dim)
            'audio_latent_cur': audio_latent_cur,  # (T, audio_dim)
            'audio_latent_prev': audio_latent_prev,  # (T, audio_dim)
            'reference_frame': data_item['reference_frame'],  # (C, H, W)
            'reference_motion': data_item['reference_motion'],  # (motion_dim,)
            'emotion_features': data_item['emotion_features'],
            'actor_id': data_item['actor_id'],
        }


def create_dataloader_optimized(data_root: str,
                               batch_size: int = 8,
                               num_workers: int = 4,
                               train: bool = True,
                               cache_dir: str = None,
                               force_preprocess: bool = False,
                               **dataset_kwargs) -> DataLoader:
    """
    创建优化的数据加载器

    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        train: 是否为训练模式
        cache_dir: 缓存目录
        force_preprocess: 是否强制重新预处理
        **dataset_kwargs: 数据集额外参数

    Returns:
        数据加载器
    """
    dataset = FLOATDatasetOptimized(
        data_root=data_root,
        train=train,
        cache_dir=cache_dir,
        force_preprocess=force_preprocess,
        **dataset_kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train
    )

    return dataloader


# 工具函数：清理缓存
def clear_cache(data_root: str, cache_dir: str = None, specific_file: str = None):
    """
    清理预处理缓存

    Args:
        data_root: 数据根目录
        cache_dir: 缓存目录
        specific_file: 特定文件（如 'preprocessed_train.pkl'），如果为None则清除整个目录
    """
    if cache_dir is None:
        cache_dir = Path(data_root) / "cache"
    else:
        cache_dir = Path(cache_dir)

    if specific_file:
        # 只删除特定文件
        file_path = cache_dir / specific_file
        if file_path.exists():
            file_path.unlink()
            print(f"Cache file cleared: {file_path}")
        else:
            print(f"Cache file does not exist: {file_path}")
    else:
        # 删除整个目录（原有逻辑）
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cache cleared: {cache_dir}")
        else:
            print(f"Cache directory does not exist: {cache_dir}")


# 工具函数：检查缓存状态
def check_cache_status(data_root: str, cache_dir: str = None):
    """
    检查缓存状态

    Args:
        data_root: 数据根目录
        cache_dir: 缓存目录
    """
    if cache_dir is None:
        cache_dir = Path(data_root) / "cache"
    else:
        cache_dir = Path(cache_dir)

    train_cache = cache_dir / "preprocessed_train.pkl"
    test_cache = cache_dir / "preprocessed_test.pkl"

    print(f"Cache directory: {cache_dir}")
    print(f"Train cache exists: {train_cache.exists()}")
    print(f"Test cache exists: {test_cache.exists()}")

    if train_cache.exists():
        size_mb = train_cache.stat().st_size / (1024 * 1024)
        print(f"Train cache size: {size_mb:.2f} MB")

    if test_cache.exists():
        size_mb = test_cache.stat().st_size / (1024 * 1024)
        print(f"Test cache size: {size_mb:.2f} MB")

if __name__ == "__main__":
    # 创建并解析选项
    opt = BaseOptions().parse()
    
    dataset = FLOATDatasetOptimized(
        data_root="../datasets/ravdess_processed",
        train=True,
        opt=opt,
        cache_dir="../datasets/ravdess_processed/cache",
        force_preprocess=True
    )
    print(f"Dataset size: {len(dataset)}")

    dataloader = create_dataloader_optimized(
        data_root="../datasets/ravdess_processed",
        train=True,
        opt=opt,
        cache_dir="../datasets/ravdess_processed/cache",
        force_preprocess=True
    )
    batch_data = next(iter(dataloader))
    print(batch_data)