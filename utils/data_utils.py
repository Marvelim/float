#!/usr/bin/env python3
"""
数据加载和处理工具函数
Data loading and processing utility functions for FLOAT training
"""

import os
import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from transformers import Wav2Vec2FeatureExtractor

from .memory_utils import get_video_cache, get_audio_cache, get_cache_size_limit


# 全局音频预处理器（避免重复创建）
_global_audio_preprocessor = None


def get_audio_preprocessor(opt):
    """获取全局音频预处理器"""
    global _global_audio_preprocessor
    if _global_audio_preprocessor is None:
        _global_audio_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(
            opt.wav2vec_model_path, local_files_only=True
        )
    return _global_audio_preprocessor


def load_video(video_path: str, opt) -> torch.Tensor:
    """加载视频帧（优化版本 + 缓存）"""
    _video_cache = get_video_cache()
    _cache_size_limit = get_cache_size_limit()
    
    # 检查缓存
    cache_key = f"{video_path}_{opt.input_size}"
    if cache_key in _video_cache:
        return _video_cache[cache_key]
    
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 预分配 numpy 数组（更高效）
    frames_array = np.zeros((total_frames, opt.input_size, opt.input_size, 3), dtype=np.float32)
    
    # 批量读取帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 优化图像处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (opt.input_size, opt.input_size), interpolation=cv2.INTER_LINEAR)
        frame = frame.astype(np.float32) / 127.5 - 1.0
        frames_array[frame_count] = frame
        frame_count += 1
    
    cap.release()
    
    assert frame_count > 0, f"No frames found in {video_path}"
    
    # 截取实际使用的帧数
    frames_array = frames_array[:frame_count]
    
    # 使用更高效的张量创建
    video_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)
    
    # 缓存结果（限制缓存大小）
    if len(_video_cache) < _cache_size_limit:
        _video_cache[cache_key] = video_tensor
    
    return video_tensor


def load_audio(audio_path: str, opt) -> torch.Tensor:
    """加载音频（优化版本 + 缓存）"""
    _audio_cache = get_audio_cache()
    _cache_size_limit = get_cache_size_limit()
    
    # 检查缓存
    cache_key = f"{audio_path}_{opt.sampling_rate}"
    if cache_key in _audio_cache:
        return _audio_cache[cache_key]
    
    # 使用全局预处理器
    wav2vec_preprocessor = get_audio_preprocessor(opt)
    
    # 使用更快的音频加载
    speech_array, sampling_rate = librosa.load(
        audio_path, 
        sr=opt.sampling_rate,
    )
    
    audio_tensor = wav2vec_preprocessor(speech_array, sampling_rate=sampling_rate, return_tensors='pt').input_values[0]
    
    # 确保形状是 (1, sequence_length)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # 缓存结果（限制缓存大小）
    if len(_audio_cache) < _cache_size_limit:
        _audio_cache[cache_key] = audio_tensor
    
    return audio_tensor


def extract_motion_latent(model, video_frames: torch.Tensor) -> torch.Tensor:
    """提取运动潜在表示（单个样本）"""
    with torch.no_grad():
        d_r, r_d_lambda, _ = model.motion_autoencoder.enc(video_frames, input_target=None)
        r_d_lambda = model.motion_autoencoder.enc.fc(d_r)
        r_d = model.motion_autoencoder.dec.direction(r_d_lambda)
        return r_d, _


def extract_motion_latent_batch(model, batch_videos: torch.Tensor) -> torch.Tensor:
    """批量提取运动潜在表示"""
    batch_size, seq_len, channels, height, width = batch_videos.shape
    
    with torch.no_grad():
        # 重塑为 (B*T, C, H, W) 进行批量处理
        videos_reshaped = batch_videos.view(-1, channels, height, width)
        
        # 批量编码
        d_r, r_d_lambda, d_r_feats = model.motion_autoencoder.enc(videos_reshaped, input_target=None)
        r_d_lambda = model.motion_autoencoder.enc.fc(d_r)
        r_d = model.motion_autoencoder.dec.direction(r_d_lambda)
        
        # 重塑回 (B, T, motion_dim)
        motion_latent = r_d.view(batch_size, seq_len, -1)
        
        return motion_latent


def get_sequence_indices(total_frames: int, opt) -> tuple:
    """获取序列索引"""
    sequence_length = int(opt.wav2vec_sec * opt.fps)
    prev_frames = int(opt.num_prev_frames)
    
    assert total_frames >= sequence_length + prev_frames
    
    # 训练时随机选择起始位置
    max_start = total_frames - sequence_length - prev_frames
    start_idx = torch.randint(0, max_start + 1, (1,)).item()
    
    end_idx = start_idx + sequence_length + prev_frames
    return start_idx, end_idx


def load(batch_data, model, device, opt):
    """
    批处理加载数据并提取视频和音频的潜在表示（内存优化版本）
    
    Args:
        batch_data: 批次数据，包含文件路径等信息
        model: FLOAT 模型
        device: 设备
        opt: 选项
        
    Returns:
        处理后的批次数据
    """
    # 处理 DataLoader 的默认 collate 结果
    if isinstance(batch_data, dict):
        batch_size = len(batch_data['video_path'])
        video_paths = batch_data['video_path']
        audio_paths = batch_data['audio_path']
        emotions = batch_data['emotion']
        actor_ids = batch_data['actor_id']
    else:
        batch_size = len(batch_data)
        video_paths = [item['video_path'] for item in batch_data]
        audio_paths = [item['audio_path'] for item in batch_data]
        emotions = [item['emotion'] for item in batch_data]
        actor_ids = [item['actor_id'] for item in batch_data]
    
    # 限制最大帧数以减少内存使用
    max_frames = int(opt.wav2vec_sec * opt.fps * 1.5)  # 减少到1.5倍
    
    # 1. 逐个处理样本（避免同时加载所有视频到内存）
    motion_latent_batch = []
    w_audio_batch = []
    random_frames_batch = []
    
    for i in range(batch_size):
        # 加载单个视频（限制帧数）
        video_frames = load_video(video_paths[i], opt)
        if video_frames.shape[0] > max_frames:
            video_frames = video_frames[:max_frames]
        
        # 加载音频
        audio = load_audio(audio_paths[i], opt)
        
        # 立即处理并释放原始视频数据
        video_tensor = video_frames.to(device)
        
        with torch.no_grad():
            # 提取运动特征
            motion_latent, _ = extract_motion_latent(model, video_tensor)
            motion_latent_batch.append(motion_latent)
            
            # 保存第一帧作为参考
            random_frame = video_tensor[0:1]
            random_frames_batch.append(random_frame)
            
            # 处理音频
            audio_tensor = audio.to(device)
            w_audio = model.audio_encoder.inference(audio_tensor, seq_len=video_frames.shape[0]).squeeze(0)
            w_audio_batch.append(w_audio)
        
        # 显式释放内存
        del video_frames, video_tensor, audio, audio_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. 统一长度并堆叠
    target_frames = min(motion_latent.shape[0] for motion_latent in motion_latent_batch)
    
    # 截断到统一长度
    for i in range(batch_size):
        motion_latent_batch[i] = motion_latent_batch[i][:target_frames]
        w_audio_batch[i] = w_audio_batch[i][:target_frames]
    
    # 堆叠成批次
    motion_latent_batch = torch.stack(motion_latent_batch)  # (B, T, motion_dim)
    w_audio_batch = torch.stack(w_audio_batch)  # (B, T, audio_dim)
    random_frames_batch = torch.stack(random_frames_batch)  # (B, C, H, W)
    
    # 3. 创建情感特征
    emotion_tensor = F.one_hot(torch.tensor(emotions), num_classes=opt.dim_e).unsqueeze(1).to(device)
    
    # 4. 准备批次数据
    batch_dict = {
        'full_random_frames': random_frames_batch,
        'full_motion_latent': motion_latent_batch,
        'full_audio_latent': w_audio_batch,
        'emotion_features': emotion_tensor,
        'actor_id': actor_ids,
        'target_frames': target_frames,
    }
    
    return batch_dict


def get_batch_sample(real_data, opt, model=None):
    """
    从完整数据中随机切分获取训练样本
    
    Args:
        real_data: 包含完整数据的字典
        opt: 选项
        
    Returns:
        切分后的训练样本
    """
    batch_size = real_data['full_motion_latent'].shape[0]
    target_frames = real_data['target_frames']
    
    # 计算序列参数
    sequence_length = int(opt.wav2vec_sec * opt.fps)
    prev_frames = int(opt.num_prev_frames)
    
    # 随机选择起始位置（对每个样本）
    max_start = target_frames - sequence_length - prev_frames
    start_indices = torch.randint(0, max_start + 1, (batch_size,)).tolist()
    
    # 批量切分
    #video_cur_batch = []
    #video_prev_batch = []
    motion_cur_batch = []
    motion_prev_batch = []
    audio_cur_batch = []
    audio_prev_batch = []
    reference_motion_batch = []
    # reference_feat_batch = []
    for i in range(batch_size):
        start_idx = start_indices[i]
        end_idx = start_idx + sequence_length + prev_frames
        
        # 视频切分
        # video_cur = real_data['full_videos'][i, start_idx + prev_frames:end_idx]
        # video_prev = real_data['full_videos'][i, start_idx:start_idx + prev_frames]
        # video_cur_batch.append(video_cur)
        # video_prev_batch.append(video_prev)
        
        # 运动特征切分
        motion_cur = real_data['full_motion_latent'][i, start_idx + prev_frames:end_idx]
        motion_prev = real_data['full_motion_latent'][i, start_idx:start_idx + prev_frames]
        motion_cur_batch.append(motion_cur)
        motion_prev_batch.append(motion_prev)
        
        # 音频特征切分
        audio_cur = real_data['full_audio_latent'][i, start_idx + prev_frames:end_idx]
        audio_prev = real_data['full_audio_latent'][i, start_idx:start_idx + prev_frames]
        audio_cur_batch.append(audio_cur)
        audio_prev_batch.append(audio_prev)
        
        # 参考运动（第一帧）- 对比两种方法
        first_frame = real_data['full_random_frames'][i]  # 取第0帧 [1, C, H, W]
        
        with torch.no_grad():
            reference_motion, _, feats = model.encode_image_into_latent(first_frame)
        
        reference_motion_batch.append(reference_motion)
        # reference_feat_batch.append(reference_feat)
    
    
    # 堆叠成最终批次并确保在GPU上
    batch_sample = {
        # 'video_cur': torch.stack(video_cur_batch).to(real_data['full_videos'].device),
        # 'video_prev': torch.stack(video_prev_batch).to(real_data['full_videos'].device),
        'motion_latent_cur': torch.stack(motion_cur_batch).to(real_data['full_motion_latent'].device),
        'motion_latent_prev': torch.stack(motion_prev_batch).to(real_data['full_motion_latent'].device),
        'audio_latent_cur': torch.stack(audio_cur_batch).to(real_data['full_audio_latent'].device),
        'audio_latent_prev': torch.stack(audio_prev_batch).to(real_data['full_audio_latent'].device),
        'reference_motion': torch.stack(reference_motion_batch).to(real_data['full_motion_latent'].device),
        # 'reference_feat': reference_feat_batch,
        'emotion_features': real_data['emotion_features'],  # 保持形状 [batch_size, 1, 7]
        'actor_id': real_data['actor_id'],
    }
    
    return batch_sample