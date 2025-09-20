#!/usr/bin/env python3
"""
内存管理工具函数
Memory management utility functions for FLOAT training
"""

import torch
import gc

# 简单的内存缓存
_video_cache = {}
_audio_cache = {}
_cache_size_limit = 100  # 限制缓存大小


def clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def clear_cache():
    """清理数据缓存"""
    global _video_cache, _audio_cache
    _video_cache.clear()
    _audio_cache.clear()
    clear_memory()


def get_cache_info():
    """获取缓存信息"""
    global _video_cache, _audio_cache
    return {
        'video_cache_size': len(_video_cache),
        'audio_cache_size': len(_audio_cache),
        'cache_size_limit': _cache_size_limit
    }


def get_video_cache():
    """获取视频缓存"""
    global _video_cache
    return _video_cache


def get_audio_cache():
    """获取音频缓存"""
    global _audio_cache
    return _audio_cache


def get_cache_size_limit():
    """获取缓存大小限制"""
    global _cache_size_limit
    return _cache_size_limit
