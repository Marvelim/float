#!/usr/bin/env python3
"""
图像处理工具函数
Image processing utility functions for FLOAT training and inference
"""

import cv2
import numpy as np
import face_alignment
import threading
import time
from typing import Dict, Optional, Tuple


class ImageProcessor:
    """图像处理器，提供统一的人脸检测和图像裁剪功能"""
    
    def __init__(self, input_size: int = 512, use_face_alignment: bool = True, 
                 fa_init_timeout: int = 10, device: str = 'auto'):
        """
        初始化图像处理器
        
        Args:
            input_size: 输出图像尺寸
            use_face_alignment: 是否使用人脸检测
            fa_init_timeout: 人脸检测初始化超时时间（秒）
            device: 设备类型 ('auto', 'cpu', 'cuda:0', etc.)
        """
        self.input_size = input_size
        self.use_face_alignment = use_face_alignment
        self.fa_init_timeout = fa_init_timeout
        
        # 初始化人脸检测器
        self.fa = None
        if use_face_alignment:
            self._init_face_alignment(device)
    
    def _init_face_alignment(self, device: str):
        """初始化人脸检测器"""
        try:
            _fa_holder = {}
            
            def _init_fa():
                try:
                    # 自动选择设备
                    if device == 'auto':
                        import torch
                        if torch.cuda.is_available() and torch.cuda.device_count() > 4:
                            fa_device = 'cuda:4'  # 使用GPU 4
                        else:
                            fa_device = 'cpu'
                    else:
                        fa_device = device
                    
                    _fa_holder['fa'] = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.TWO_D,
                        flip_input=False,
                        device=fa_device
                    )
                except Exception as ie:
                    _fa_holder['err'] = ie
            
            t = threading.Thread(target=_init_fa, daemon=True)
            t.start()
            t.join(timeout=max(1, int(self.fa_init_timeout)))
            
            if t.is_alive():
                print("Warning: face_alignment init timeout, fallback to center crop.")
            else:
                if 'err' in _fa_holder:
                    print(f"Warning: Could not initialize face alignment: {_fa_holder['err']}")
                else:
                    self.fa = _fa_holder.get('fa')
        except Exception as e:
            print(f"Warning: face_alignment setup failed: {e}")
            self.fa = None
    
    def get_crop_params(self, img: np.ndarray) -> Dict:
        """
        计算图像裁剪参数（仿照 generate.py 的 process_img 方法）
        
        Args:
            img: 输入图像 (H, W, C)
            
        Returns:
            裁剪参数字典，包含裁剪类型和参数
        """
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
            # 缩放图像以提高检测速度（与 generate.py 完全一致）
            mult = 360. / img.shape[0]
            resized_img = cv2.resize(img, dsize=(0, 0), fx=mult, fy=mult, 
                                   interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
            
            # 检测人脸
            bboxes = self.fa.face_detector.detect_from_image(resized_img)
            # 过滤置信度大于0.95的检测结果（与 generate.py 一致）
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
            
            # 使用第一个检测到的人脸（与 generate.py 一致）
            bboxes = bboxes[0]
            
            # 计算人脸中心和半尺寸（与 generate.py 完全一致）
            bsy = int((bboxes[3] - bboxes[1]) / 2)
            bsx = int((bboxes[2] - bboxes[0]) / 2)
            my = int((bboxes[1] + bboxes[3]) / 2)
            mx = int((bboxes[0] + bboxes[2]) / 2)
            
            # 计算扩展尺寸（与 generate.py 完全一致）
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
    
    def apply_crop_params(self, img: np.ndarray, crop_params: Dict) -> np.ndarray:
        """
        根据裁剪参数处理图像
        
        Args:
            img: 输入图像 (H, W, C)
            crop_params: 裁剪参数字典
            
        Returns:
            处理后的图像 (input_size, input_size, C)
        """
        if crop_params['type'] == 'center':
            # 中心裁剪
            y, x, size = crop_params['y'], crop_params['x'], crop_params['size']
            crop_img = img[y:y+size, x:x+size]
            return cv2.resize(crop_img, (self.input_size, self.input_size))
        
        elif crop_params['type'] == 'face':
            # 人脸裁剪（与 generate.py 完全一致）
            bs, my, mx = crop_params['bs'], crop_params['my'], crop_params['mx']
            mult = crop_params['mult']
            
            # 添加边框（与 generate.py 完全一致）
            img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
            my, mx = my + bs, mx + bs  # 更新中心坐标
            
            # 裁剪正方形区域（与 generate.py 完全一致）
            crop_img = img[my - bs:my + bs, mx - bs:mx + bs]
            
            # 调整到目标尺寸（与 generate.py 完全一致）
            crop_img = cv2.resize(crop_img, dsize=(self.input_size, self.input_size), 
                                interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
            
            return crop_img
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """
        处理图像（一步完成，仿照 generate.py 的 process_img 方法）
        
        Args:
            img: 输入图像 (H, W, C)
            
        Returns:
            处理后的图像 (input_size, input_size, C)
        """
        crop_params = self.get_crop_params(img)
        return self.apply_crop_params(img, crop_params)


# 全局图像处理器实例（用于向后兼容）
_global_image_processor = None


def get_image_processor(input_size: int = 512, use_face_alignment: bool = True, 
                       fa_init_timeout: int = 10, device: str = 'auto') -> ImageProcessor:
    """
    获取全局图像处理器实例
    
    Args:
        input_size: 输出图像尺寸
        use_face_alignment: 是否使用人脸检测
        fa_init_timeout: 人脸检测初始化超时时间（秒）
        device: 设备类型
        
    Returns:
        ImageProcessor 实例
    """
    global _global_image_processor
    if _global_image_processor is None:
        _global_image_processor = ImageProcessor(
            input_size=input_size,
            use_face_alignment=use_face_alignment,
            fa_init_timeout=fa_init_timeout,
            device=device
        )
    return _global_image_processor


def process_image_like_generate(img: np.ndarray, input_size: int = 512, 
                               use_face_alignment: bool = True) -> np.ndarray:
    """
    仿照 generate.py 的方式处理图像（向后兼容函数）
    
    Args:
        img: 输入图像 (H, W, C)
        input_size: 输出图像尺寸
        use_face_alignment: 是否使用人脸检测
        
    Returns:
        处理后的图像 (input_size, input_size, C)
    """
    processor = get_image_processor(input_size, use_face_alignment)
    return processor.process_image(img)


def get_crop_params_like_generate(img: np.ndarray, input_size: int = 512, 
                                 use_face_alignment: bool = True) -> Dict:
    """
    仿照 generate.py 的方式计算裁剪参数（向后兼容函数）
    
    Args:
        img: 输入图像 (H, W, C)
        input_size: 输出图像尺寸
        use_face_alignment: 是否使用人脸检测
        
    Returns:
        裁剪参数字典
    """
    processor = get_image_processor(input_size, use_face_alignment)
    return processor.get_crop_params(img)


def apply_crop_params_like_generate(img: np.ndarray, crop_params: Dict, 
                                   input_size: int = 512) -> np.ndarray:
    """
    根据裁剪参数处理图像（向后兼容函数）
    
    Args:
        img: 输入图像 (H, W, C)
        crop_params: 裁剪参数字典
        input_size: 输出图像尺寸
        
    Returns:
        处理后的图像 (input_size, input_size, C)
    """
    processor = get_image_processor(input_size)
    return processor.apply_crop_params(img, crop_params)
