#!/usr/bin/env python3
"""
RAVDESS 数据集下载和预处理脚本
支持下载音频和视频文件，并进行标准预处理
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
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

class RAVDESSDownloader:
    """RAVDESS数据集下载器"""
    
    def __init__(self, data_dir="./ravdess_raw", num_workers=1):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        # 并行分块下载的线程数（仅用于单文件分块下载）
        self.num_workers = max(1, int(num_workers))
        
        # RAVDESS官方下载链接
        self.download_urls = {
            # 音频文件
            "Audio_Speech_Actors_01-24.zip": {
                "url": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
                "md5": "bc696df654c87fed845eb13823edef8a",
                "size": "215.5 MB",
                "type": "audio"
            },
            "Audio_Song_Actors_01-24.zip": {
                "url": "https://zenodo.org/record/1188976/files/Audio_Song_Actors_01-24.zip",
                "md5": "5411230427d67a21e18aa4d466e6d1b9",
                "size": "225.5 MB",
                "type": "audio"
            },
            # 视频文件 (按Actor分组)
            **{f"Video_Speech_Actor_{i:02d}.zip": {
                "url": f"https://zenodo.org/record/1188976/files/Video_Speech_Actor_{i:02d}.zip",
                "md5": None,  # 需要时可以添加
                "type": "video_speech"
            } for i in range(1, 25)},
            **{f"Video_Song_Actor_{i:02d}.zip": {
                "url": f"https://zenodo.org/record/1188976/files/Video_Song_Actor_{i:02d}.zip", 
                "md5": None,
                "type": "video_song"
            } for i in range(1, 25)}
        }
    
    def calculate_md5(self, file_path):
        """计算文件MD5"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def download_file(self, url, local_path, expected_md5=None, show_progress=True, use_parallel=False, pbar_position=None):
        """下载文件并验证MD5。
        - 默认串行下载，显示进度条、超时与重试。
        - 当 use_parallel=True 且服务器支持 Range 时，启用分块并行下载以加速（仿照 HDTF 的并行思路）。
        """
        # 确保目录存在
        local_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                # 如果要求并行，尝试 Range 支持
                if use_parallel and self.num_workers > 1:
                    parallel_ok = self._download_file_parallel(url, tmp_path, show_progress, pbar_position=pbar_position)
                    if not parallel_ok:
                        # 回退到串行
                        self._download_file_serial(url, tmp_path, show_progress, pbar_position=pbar_position)
                else:
                    self._download_file_serial(url, tmp_path, show_progress, pbar_position=pbar_position)
                # 原子替换
                tmp_path.replace(local_path)

                # 验证MD5
                if expected_md5:
                    actual_md5 = self.calculate_md5(local_path)
                    if actual_md5 != expected_md5:
                        # 删除错误文件，重试
                        try:
                            local_path.unlink(missing_ok=True)
                        except TypeError:
                            if local_path.exists():
                                local_path.unlink()
                        raise IOError("MD5 mismatch")
                return True
            except Exception as e:
                if show_progress:
                    print(f"第 {attempt}/{attempts} 次下载失败: {e}")
                time.sleep(2 * attempt)
        # 清理临时文件
        try:
            tmp_path.unlink(missing_ok=True)
        except TypeError:
            if tmp_path.exists():
                tmp_path.unlink()
        return False

    def _download_file_serial(self, url, tmp_path: Path, show_progress: bool, pbar_position=None):
        with urllib.request.urlopen(url, timeout=120) as resp:
            total = resp.getheader('Content-Length')
            total = int(total) if total is not None else None
            desc = f"下载 {tmp_path.stem}"
            with open(tmp_path, 'wb') as out, \
                 tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                      disable=not show_progress, desc=desc, position=(pbar_position or 0), leave=True) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    if show_progress:
                        pbar.update(len(chunk))

    def _head_content_info(self, url):
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as resp:
                cl = resp.getheader('Content-Length')
                ar = resp.getheader('Accept-Ranges')
                return (int(cl) if cl is not None else None, (ar or '').lower())
        except Exception:
            return (None, '')

    def _download_file_parallel(self, url, tmp_path: Path, show_progress: bool, pbar_position=None) -> bool:
        total, accept_ranges = self._head_content_info(url)
        if total is None or 'bytes' not in accept_ranges:
            return False

        part_count = min(self.num_workers, 8)  # 保守上限
        part_size = total // part_count
        ranges = []
        start = 0
        for i in range(part_count):
            end = start + part_size - 1 if i < part_count - 1 else total - 1
            ranges.append((i, start, end))
            start = end + 1

        # 每个分片下载到独立文件后合并
        tmp_dir = tmp_path.parent
        pbar = tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024,
                    disable=not show_progress, desc=f"下载 {tmp_path.stem} (并行)", position=(pbar_position or 0), leave=True)
        lock = threading.Lock()
        errors = []

        def download_part(idx, s, e):
            headers = {"Range": f"bytes={s}-{e}"}
            req = urllib.request.Request(url, headers=headers)
            part_file = tmp_dir / f"{tmp_path.name}.part{idx}"
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, open(part_file, 'wb') as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        if show_progress:
                            with lock:
                                pbar.update(len(chunk))
            except Exception as e:
                errors.append((idx, str(e)))

        threads = []
        for i, s, e in ranges:
            t = threading.Thread(target=download_part, args=(i, s, e), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        pbar.close()

        if errors:
            # 清理分片
            for i, _, _ in ranges:
                part_file = tmp_dir / f"{tmp_path.name}.part{i}"
                if part_file.exists():
                    part_file.unlink()
            raise IOError(f"并行下载失败: {errors[:1][0][1]}")

        # 合并
        with open(tmp_path, 'wb') as out:
            for i, _, _ in ranges:
                part_file = tmp_dir / f"{tmp_path.name}.part{i}"
                with open(part_file, 'rb') as pf:
                    shutil.copyfileobj(pf, out)
                part_file.unlink()
        return True
    
    def extract_zip(self, zip_path, extract_to, show_progress=True):
        """解压ZIP文件（显示条目进度）"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.infolist()
                desc = f"解压 {zip_path.name}"
                with tqdm(total=len(members), unit='file', disable=not show_progress, desc=desc) as pbar:
                    for m in members:
                        zip_ref.extract(m, extract_to)
                        if show_progress:
                            pbar.update(1)
                return True
        except Exception as e:
            if show_progress:
                print(f"❌ 解压失败: {e}")
            return False
    
    def _download_and_extract_one(self, file_key, force_download=False, verbose=True):
        if file_key not in self.download_urls:
            if verbose:
                print(f"❌ 未知的文件: {file_key}")
            return False

        file_info = self.download_urls[file_key]
        zip_path = self.data_dir / file_key

        if verbose:
            print(f"\n{'='*60}")
            print(f"处理文件: {file_key}")
            print(f"{'='*60}")

        # 检查文件是否已存在
        if zip_path.exists() and not force_download:
            if file_info.get("md5"):
                if self.calculate_md5(zip_path) == file_info["md5"]:
                    if verbose:
                        print("✅ 文件已存在且完整，跳过下载")
                else:
                    if verbose:
                        print("❌ 文件存在但不完整，重新下载")
                    zip_path.unlink()
            else:
                if verbose:
                    print("✅ 文件已存在，跳过下载")

        # 下载文件
        if not zip_path.exists():
            success = self.download_file(
                file_info["url"],
                zip_path,
                file_info.get("md5"),
                show_progress=verbose and self.num_workers == 1,
            )
            if not success:
                if verbose:
                    print(f"❌ {file_key} 下载失败")
                return False

        # 解压文件
        extract_success = self.extract_zip(zip_path, self.data_dir, show_progress=verbose)
        if extract_success:
            if verbose:
                print(f"✅ {file_key} 处理完成")
            return True
        else:
            if verbose:
                print(f"❌ {file_key} 解压失败")
            return False

    def download_and_extract(self, file_keys, force_download=False):
        """下载并解压指定的文件。
        - 当 num_workers <= 1：串行下载 + 条目进度。
        - 当 num_workers > 1：多文件并行下载（外层进度条），随后串行解压（带条目进度）。
        """
        total_count = len(file_keys)

        # 串行路径
        if self.num_workers <= 1:
            success_count = 0
            for file_key in file_keys:
                if file_key not in self.download_urls:
                    print(f"❌ 未知的文件: {file_key}")
                    continue
                file_info = self.download_urls[file_key]
                zip_path = self.data_dir / file_key

                print(f"\n{'='*60}")
                print(f"处理文件: {file_key}")
                print(f"{'='*60}")

                # 已存在处理
                if zip_path.exists() and not force_download:
                    if file_info.get("md5"):
                        if self.calculate_md5(zip_path) == file_info["md5"]:
                            print("✅ 文件已存在且完整，跳过下载")
                        else:
                            print("❌ 文件存在但不完整，重新下载")
                            zip_path.unlink()
                    else:
                        # 无MD5的视频ZIP，做有效性校验
                        if not zipfile.is_zipfile(zip_path):
                            print("⚠️  现有文件不是有效的ZIP，重新下载")
                            zip_path.unlink()
                        else:
                            print("✅ 文件已存在，跳过下载")

                # 下载（串行，显示进度）
                if not zip_path.exists():
                    ok = self.download_file(file_info["url"], zip_path, file_info.get("md5"), show_progress=True, use_parallel=False, pbar_position=0)
                    if not ok:
                        print(f"❌ {file_key} 下载失败")
                        continue

                # 解压
                if self.extract_zip(zip_path, self.data_dir, show_progress=True):
                    success_count += 1
                    print(f"✅ {file_key} 处理完成")
                else:
                    print(f"❌ {file_key} 解压失败")

            print(f"\n总结: {success_count}/{total_count} 个文件处理成功")
            return success_count == total_count

        # 多文件并行下载
        print(f"并行下载启用：num_workers={self.num_workers}")
        dl_results = {}

        def download_worker(k, pos):
            if k not in self.download_urls:
                return (k, False)
            info = self.download_urls[k]
            zp = self.data_dir / k
            # 已存在处理
            if zp.exists() and not force_download:
                if info.get("md5"):
                    if self.calculate_md5(zp) == info["md5"]:
                        return (k, True)
                    else:
                        try:
                            zp.unlink()
                        except Exception:
                            pass
                else:
                    # 无MD5的视频ZIP，做有效性校验
                    if zipfile.is_zipfile(zp):
                        return (k, True)
                    else:
                        try:
                            zp.unlink()
                        except Exception:
                            pass
            ok = self.download_file(info["url"], zp, info.get("md5"), show_progress=True, use_parallel=False, pbar_position=pos)
            return (k, ok)

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {}
            for idx, k in enumerate(file_keys):
                pos = (idx % self.num_workers)
                futures[ex.submit(download_worker, k, pos)] = k
            with tqdm(total=len(futures), desc='下载文件', unit='file', position=self.num_workers, leave=True) as pbar:
                for fut in as_completed(futures):
                    k, ok = fut.result()
                    dl_results[k] = ok
                    pbar.update(1)

        # 串行解压
        success_count = 0
        for k in file_keys:
            if not dl_results.get(k, False):
                print(f"跳过解压（下载失败或文件缺失）: {k}")
                continue
            zp = self.data_dir / k
            print(f"开始解压: {k}")
            if self.extract_zip(zp, self.data_dir, show_progress=True):
                success_count += 1
                print(f"✅ 解压完成: {k}")
            else:
                print(f"❌ 解压失败: {k}")

        print(f"\n总结: {success_count}/{total_count} 个文件处理成功")
        return success_count == total_count


class RAVDESSPreprocessor:
    """RAVDESS数据预处理器"""
    
    def __init__(self, raw_dir="./ravdess_raw", processed_dir="./ravdess_processed", 
                 target_fps=25, target_sr=16000, target_size=512, use_face_alignment=True, fa_init_timeout=10):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_fps = target_fps
        self.target_sr = target_sr
        self.target_size = target_size
        
        # 初始化人脸检测器（可禁用，并带超时避免卡死）
        self.fa = None
        if use_face_alignment:
            try:
                import threading
                _fa_holder = {}
                def _init_fa():
                    try:
                        # 使用GPU 4-7中的第一个可用GPU
                        import torch
                        if torch.cuda.is_available() and torch.cuda.device_count() > 4:
                            device = 'cuda:4'  # 使用GPU 4
                        else:
                            device = 'cpu'
                        _fa_holder['fa'] = face_alignment.FaceAlignment(
                            face_alignment.LandmarksType.TWO_D,
                            flip_input=False,
                            device=device
                        )
                    except Exception as ie:
                        _fa_holder['err'] = ie
                t = threading.Thread(target=_init_fa, daemon=True)
                t.start()
                t.join(timeout=max(1, int(fa_init_timeout)))
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
            
    def find_matching_audio(self, video_file):
        """为视频文件找到匹配的高质量音频文件"""
        # 视频文件格式: 01-01-... (Video+Audio)
        # 音频文件格式: 03-01-... (Audio only)
        
        video_parts = video_file.stem.split('-')
        if len(video_parts) < 7:
            return None
        
        # 构造对应的音频文件名：将第一个字段从'01'改为'03'
        audio_filename = '03-' + '-'.join(video_parts[1:]) + '.wav'
        audio_path = video_file.parent / audio_filename
        
        if audio_path.exists():
            return audio_path
        else:
            return None

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
            
    def preprocess_dataset(self, actor_ids=None, modalities=None):
        """预处理整个数据集"""
        if actor_ids is None:
            actor_ids = list(range(1, 25))  # Actor 01-24
        
        if modalities is None:
            modalities = ['speech']  # 默认只处理speech
        
        print(f"开始预处理 RAVDESS 数据集")
        print(f"Actor IDs: {actor_ids}")
        print(f"模态: {modalities}")
        print(f"目标FPS: {self.target_fps}")
        print(f"目标采样率: {self.target_sr} Hz")
        print(f"目标分辨率: {self.target_size}x{self.target_size}")
        print("=" * 60)
        
        total_processed = 0
        total_failed = 0
        
        for actor_id in actor_ids:
            actor_dir = self.raw_dir / f"Actor_{actor_id:02d}"
            
            if not actor_dir.exists():
                print(f"⚠️  Actor_{actor_id:02d} 目录不存在，跳过")
                continue
            
            print(f"\n处理 Actor_{actor_id:02d}...")
            
            # 处理视频文件
            video_files = list(actor_dir.glob("*.mp4"))
            
            for video_file in tqdm(video_files, desc=f"Actor_{actor_id:02d}"):
                # 检查是否是目标模态
                filename = video_file.name
                is_speech = filename.startswith("01-01")  # Speech modality
                is_song = filename.startswith("01-02")    # Song modality
                
                should_process = False
                if 'speech' in modalities and is_speech:
                    should_process = True
                elif 'song' in modalities and is_song:
                    should_process = True
                
                if not should_process:
                    continue
                
                # 设置输出路径
                output_dir = self.processed_dir / f"Actor_{actor_id:02d}"
                output_video_path = output_dir / f"{video_file.stem}_processed.mp4"
                
                # 检查是否已处理
                if output_video_path.exists():
                    continue
                
                # 查找匹配的高质量音频文件
                audio_file = self.find_matching_audio(video_file)
                
                if audio_file:
                    # 使用高质量音频处理
                    success = self.process_video_with_separate_audio(video_file, audio_file, output_video_path)
                else:
                    # 跳过没有匹配音频文件的视频
                    print(f"⚠️  未找到匹配的音频文件，跳过: {video_file.name}")
                    continue
                
                if success:
                    total_processed += 1
                else:
                    total_failed += 1
        
        print(f"\n预处理完成!")
        print(f"成功处理: {total_processed} 个文件")
        print(f"处理失败: {total_failed} 个文件")
        
        return total_processed, total_failed


def main():
    parser = argparse.ArgumentParser(description='RAVDESS 数据集下载和预处理')
    parser.add_argument('--download', action='store_true', help='下载数据集')
    parser.add_argument('--preprocess', action='store_true', help='预处理数据集')
    parser.add_argument('--actors', type=str, default=
    '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24', 
                       help='要处理的Actor ID (逗号分隔，例如: 1,2,3 或 23,24)')
    parser.add_argument('--modalities', type=str, default='speech',
                       choices=['speech', 'song', 'both'], help='要处理的模态')
    parser.add_argument('--download_audio', action='store_true', 
                       help='下载音频文件 (Audio_Speech 和 Audio_Song)')
    parser.add_argument('--download_video', action='store_true',
                       help='下载视频文件')
    parser.add_argument('--force_download', action='store_true', help='强制重新下载')
    parser.add_argument('--raw_dir', type=str, default='./ravdess_raw', help='原始数据目录')
    parser.add_argument('--processed_dir', type=str, default='./ravdess_processed', help='处理后数据目录')
    parser.add_argument('--num_workers', type=int, default=1, help='下载并行线程数（建议 2-8）')
    parser.add_argument('--no_face', action='store_true', help='预处理时禁用人脸检测，使用中心裁剪（避免卡住）')
    parser.add_argument('--fa_init_timeout', type=int, default=10, help='人脸检测初始化超时（秒），超过则回退中心裁剪')
    
    args = parser.parse_args()
    
    # 解析actor IDs
    if args.actors.lower() == 'all':
        actor_ids = list(range(1, 25))
    else:
        actor_ids = [int(x.strip()) for x in args.actors.split(',')]
    
    # 解析modalities
    if args.modalities == 'both':
        modalities = ['speech', 'song']
    else:
        modalities = [args.modalities]
    
    # 初始化下载器
    downloader = RAVDESSDownloader(args.raw_dir, num_workers=args.num_workers)
    
    # 下载文件
    if args.download:
        files_to_download = []
        
        # 添加音频文件
        if args.download_audio:
            if 'speech' in modalities:
                files_to_download.append("Audio_Speech_Actors_01-24.zip")
            if 'song' in modalities:
                files_to_download.append("Audio_Song_Actors_01-24.zip")
        
        # 添加视频文件
        if args.download_video:
            for actor_id in actor_ids:
                if 'speech' in modalities:
                    files_to_download.append(f"Video_Speech_Actor_{actor_id:02d}.zip")
                if 'song' in modalities:
                    files_to_download.append(f"Video_Song_Actor_{actor_id:02d}.zip")
        
        if files_to_download:
            print(f"准备下载 {len(files_to_download)} 个文件:")
            for f in files_to_download:
                print(f"  - {f}")
            
            success = downloader.download_and_extract(files_to_download, args.force_download)
            
            if success:
                print("✅ 所有文件下载完成")
            else:
                print("❌ 部分文件下载失败")
        else:
            print("❌ 没有指定要下载的文件")
            print("使用 --download_audio 和/或 --download_video 指定要下载的内容")
    
    # 预处理数据
    if args.preprocess:
        preprocessor = RAVDESSPreprocessor(args.raw_dir, args.processed_dir,
                                           use_face_alignment=(not args.no_face),
                                           fa_init_timeout=args.fa_init_timeout)
        success_count, failed_count = preprocessor.preprocess_dataset(actor_ids, modalities)
        
        if failed_count == 0:
            print("✅ 所有文件预处理完成")
        else:
            print(f"⚠️  {failed_count} 个文件预处理失败")


if __name__ == "__main__":
    main()
