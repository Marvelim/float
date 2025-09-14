"""
Dataset loader for FLOAT training
用于 FLOAT 训练的数据加载器
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import cv2
from typing import Dict, List, Tuple, Optional, Any
import json
import random
from pathlib import Path
from models.float.generator import Generator
from models.float.FLOAT import AudioEncoder, Audio2Emotion
from transformers import Wav2Vec2FeatureExtractor

class FLOATDataset(Dataset):
    """
    FLOAT 训练数据集
    
    加载视频、音频和情感标签，用于 flow matching 训练
    """
    
    def __init__(self, 
                 data_root: str,
                 train: bool = True, 
                 opt: dict = None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            train: 是否为训练模式
            opt: 选项
        """
        self.data_root = Path(data_root)
        self.opt = opt
        self.train = train
        self.motion_autoencoder = Generator(size = self.opt.input_size, style_dim = self.opt.dim_w, motion_dim = self.opt.dim_m)
        self.audio_encoder = AudioEncoder(opt)
        self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
        self.sequence_length = int(self.opt.wav2vec_sec * self.opt.fps)
        self.prev_frames = int(self.opt.num_prev_frames)

        # 情感标签映射
        self.emotion_labels = {
            "angry": 0, "disgust": 1, "fear": 2, "happy": 3,
            "neutral": 4, "sad": 5, "surprise": 6
        }
        
        # 加载数据列表
        self.data_list = self._load_data_list()


        
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载数据列表
        
        Returns:
            数据项列表
        """
        data_list = []
        
        # 假设数据结构类似于 RAVDESS 数据集
        if self.train:
            data_dir = self.data_root / "ravdess_processed" / "train"
        else:
            data_dir = self.data_root / "ravdess_processed" / "test"
        
        if not data_dir.exists():
            # 如果没有预处理数据，使用原始数据
            data_dir = self.data_root / "ravdess_raw"
        
        # 遍历所有演员文件夹
        for actor_dir in data_dir.glob("Actor_*"):
            if not actor_dir.is_dir():
                continue
                
            # 遍历每个演员的文件
            for video_file in actor_dir.glob("*.mp4"):
                # 解析文件名获取情感信息
                # RAVDESS 格式: 03-01-06-01-02-01-12.mp4
                # 第3个数字是情感标签 (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
                parts = video_file.stem.split('-')
                if len(parts) >= 3:
                    emotion_id = int(parts[2])
                    # 映射到我们的情感标签
                    emotion_map = {1: 4, 2: 4, 3: 3, 4: 5, 5: 0, 6: 2, 7: 1, 8: 6}  # RAVDESS -> 我们的标签
                    emotion = emotion_map.get(emotion_id, 4)  # 默认为 neutral
                else:
                    emotion = 4  # 默认为 neutral
                
                # 查找对应的音频文件
                audio_file = video_file.with_suffix('.wav')
                if not audio_file.exists():
                    # 尝试其他音频格式
                    for ext in ['.wav', '.mp3', '.flac']:
                        audio_file = video_file.with_suffix(ext)
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
    
    def __len__(self) -> int:
        return len(self.data_list)
    
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
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """
        加载音频
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频张量 (1, T)
        """
        
        speech_array, sampling_rate = librosa.load(audio_path, sr = self.opt.sampling_rate)
        audio_tensor = self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]
        # 确保形状是 (1, sequence_length)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        return audio_tensor
    

    @torch.no_grad()
    def _extract_motion_latent(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        提取动作潜在表示（模拟）
        
        在实际应用中，这里应该使用预训练的运动编码器
        
        Args:
            video_frames: 视频帧 (T, C, H, W)
            
        Returns:
            动作潜在表示 (T, motion_dim)
        """
        # 确保输入张量在正确的设备上
        device = next(self.motion_autoencoder.parameters()).device
        video_frames = video_frames.to(device)
        
        d_r, r_d_lambda, d_r_feats = self.motion_autoencoder.enc(video_frames, input_target=None)
        r_d_lambda = self.motion_autoencoder.enc.fc(d_r)
        r_d = self.motion_autoencoder.dec.direction(r_d_lambda)
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # print("inside getitem")
        """
        获取数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            数据字典
        """
        data_item = self.data_list[idx]
        # print(f"Loading data item {idx}: {data_item['video_path']}")
        
        # 加载视频和音频
        # print("Loading video...")
        video_frames = self._load_video(data_item['video_path'])
        # print(f"Video loaded, shape: {video_frames.shape}")
        
        # print("Loading audio...")
        audio = self._load_audio(data_item['audio_path'])
        # print(f"Audio loaded, shape: {audio.shape}")

        # 注意：不在数据集中移动数据到CUDA设备，避免多进程CUDA错误
        # 数据移动将在训练循环中进行

        num_frames = video_frames.shape[0]
        # print(f"Starting audio encoder inference for {num_frames} frames...")
        
        # 注意：在多进程环境中，不能将数据移动到CUDA设备
        # 音频编码将在训练循环中进行，这里只返回原始音频数据
        # 确保音频张量在正确的设备上
        device = next(self.audio_encoder.parameters()).device
        audio = audio.to(device)
        w_audio = self.audio_encoder.inference(audio, seq_len=num_frames).squeeze(0)
        
        start_idx, end_idx = self._get_sequence_indices(num_frames)
        
        video_cur = video_frames[start_idx + self.prev_frames:end_idx]
        video_prev = video_frames[start_idx: start_idx + self.prev_frames]
        w_audio_cur = w_audio[start_idx + self.prev_frames:end_idx]
        w_audio_prev = w_audio[start_idx: start_idx + self.prev_frames]
        
        motion_latent_cur = self._extract_motion_latent(video_cur)
        motion_latent_prev = self._extract_motion_latent(video_prev)
        emotion = F.one_hot(torch.tensor(data_item['emotion']), num_classes = self.opt.dim_e)

        # 准备参考帧（第一帧）
        reference_frame = video_frames[0:1].squeeze(0)  # 保持维度 (1, C, H, W)
        # print("Extracting reference motion...")
        reference_motion = self._extract_motion_latent(reference_frame).squeeze(0)
        # print(f"Reference motion extracted, shape: {reference_motion.shape}")
        
        result = {
            'video_cur': video_cur,  # (T, C, H, W)
            'video_prev': video_prev,
            'motion_latent_cur': motion_latent_cur,  # (T, motion_dim)
            'motion_latent_prev': motion_latent_prev,  # (T, motion_dim)
            'audio_latent_cur': w_audio_cur,  # (T, audio_dim)
            'audio_latent_prev': w_audio_prev,  # (T, audio_dim)
            'reference_frame': reference_frame,  # (C, H, W)
            'reference_motion': reference_motion,  # (motion_dim,)
            'emotion_features': emotion, 
            'actor_id': data_item['actor_id'],
        }
        # print("inside getitem return")
        return result


def create_dataloader(data_root: str,
                     batch_size: int = 8,
                     num_workers: int = 4,
                     train: bool = True,
                     **dataset_kwargs) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        train: 是否为训练模式
        **dataset_kwargs: 数据集额外参数
        
    Returns:
        数据加载器
    """
    dataset = FLOATDataset(
        data_root=data_root,
        train=train,
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
    