#!/usr/bin/env python3
"""
视频生成和保存工具函数
Video generation and saving utility functions for FLOAT training
"""

import os
import tempfile
import subprocess
import datetime
import torch
import torchvision


def save_generated_video(video_tensor, output_path, audio_path, fps):
    """保存生成的视频"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_filename = temp_video.name
        
        # 处理视频张量
        vid = video_tensor.permute(0, 2, 3, 1)
        vid = vid.detach().clamp(-1, 1).cpu()
        vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
        
        # 写入临时视频文件
        torchvision.io.write_video(temp_filename, vid, fps=fps)
        
        # 合并音频
        if audio_path and os.path.exists(audio_path):
            with open(os.devnull, 'wb') as f:
                command = f"ffmpeg -i {temp_filename} -i {audio_path} -c:v copy -c:a aac {output_path} -y"
                subprocess.call(command, shell=True, stdout=f, stderr=f)
            if os.path.exists(output_path):
                os.remove(temp_filename)
        else:
            os.rename(temp_filename, output_path)


def save_test_video(img_tensor, output_path, fps):
    """保存测试视频（重复图片帧）"""
    # 重复图片帧创建简单视频
    video_frames = img_tensor.unsqueeze(1).repeat(1, 30, 1, 1, 1)  # 30帧
    vid = video_frames.permute(0, 2, 3, 1)
    vid = vid.detach().clamp(-1, 1).cpu()
    vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
    
    torchvision.io.write_video(output_path, vid, fps=fps)


def save_data_out_as_video(data_out, opt, step=None, audio_path=None):
    """
    仿照 generate.py 的方式保存 data_out 为视频
    
    Args:
        data_out: 模型输出的视频张量
        opt: 选项
        step: 训练步数（可选）
        audio_path: 音频文件路径（可选，用于合并音频）
    """
    # 生成输出路径
    if step is not None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = f"../results/data_out_step_{step}_{timestamp}.mp4"
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = f"../results/data_out_{timestamp}.mp4"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 使用临时文件保存视频 - 完全仿照 generate.py 的方式
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_filename = temp_video.name
        
        # 处理视频张量 - 仿照 generate.py 的方式
        vid = data_out.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        vid = vid.detach().clamp(-1, 1).cpu()
        vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
        
        # 写入临时视频文件
        torchvision.io.write_video(temp_filename, vid, fps=opt.fps)
        
        # 如果有音频文件，合并音频（仿照 generate.py）
        if audio_path is not None and os.path.exists(audio_path):
            with open(os.devnull, 'wb') as f:
                command = f"ffmpeg -i {temp_filename} -i {audio_path} -c:v copy -c:a aac {output_path} -y"
                subprocess.call(command, shell=True, stdout=f, stderr=f)
            if os.path.exists(output_path):
                os.remove(temp_filename)
                print(f"data_out 视频（含音频）已保存: {output_path}")
            else:
                print(f"警告: 合并音频失败")
        else:
            # 没有音频文件，直接重命名
            if os.path.exists(temp_filename):
                os.rename(temp_filename, output_path)
                print(f"data_out 视频已保存: {output_path}")
            else:
                print(f"警告: 临时视频文件未生成")
    
    return output_path


def ensure(model, new_batch, opt, step=None):
    """
    确保模型输出并保存为视频
    
    Args:
        model: FLOAT 模型
        new_batch: 批次数据
        opt: 选项
        step: 训练步数（可选，用于命名）
    """
    try:
        # 确保张量维度正确
        s_r = new_batch['reference_motion'][0, 0:1]  # (1, motion_dim) - 保持批次维度
        s_r_feat = new_batch['reference_feat'][0]  # 保持原始格式
        r_s = model.encode_identity_into_motion(s_r)
        r_d = new_batch['motion_latent_cur'][0:1]  # (1, T, motion_dim)
        # r_d = r_s.repeat(1, r_d.shape[1], 1)
        # r_d = torch.zeros_like(r_d)
        
        
        # 调用模型解码
        data_out = model.decode_latent_into_image(s_r=s_r, s_r_feats=s_r_feat, r_d=r_d)
        
        # 保存为视频
        save_data_out_as_video(data_out['d_hat'], opt, step)
        # save_data_out_as_video(new_batch['video_cur'][0], opt, step)
        
        return data_out
        
    except Exception as e:
        print(f"ensure 函数出错: {e}")
        print(f"批次数据键: {list(new_batch.keys())}")
        for key, value in new_batch.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        raise e


def generate_sample(model, device, opt, step):
    """使用generate.py的方法从assets目录中的图片和音频生成样本视频"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    # 直接使用绝对路径，避免路径计算问题
    project_root = Path("/home/mli374/float")
    sys.path.append(str(project_root))
    
    try:
        # 导入generate.py中的类
        from generate import DataProcessor, InferenceAgent
        
        model.eval()
        
        # 设置assets路径
        assets_dir = project_root / "assets"
        ref_image_path = assets_dir / "sam_altman_512x512.jpg"
        audio_path = assets_dir / "audio.wav"
        
        print(f"项目根目录: {project_root}")
        print(f"Assets目录: {assets_dir}")
        print(f"图片路径: {ref_image_path}")
        print(f"音频路径: {audio_path}")
        print(f"图片存在: {ref_image_path.exists()}")
        print(f"音频存在: {audio_path.exists()}")
        
        if not ref_image_path.exists() or not audio_path.exists():
            print(f"警告: 找不到assets文件，跳过样本生成")
            model.train()
            return None
        
        # 创建数据处理器
        data_processor = DataProcessor(opt)
        
        # 预处理数据
        print(f"预处理图片和音频...")
        data = data_processor.preprocess(str(ref_image_path), str(audio_path), no_crop=False)
        print(f"预处理完成")
        
        # 将数据移动到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        # 使用模型进行推理
        print(f"开始推理生成...")
        with torch.no_grad():
            generated_video = model.inference(
                data=data,
                a_cfg_scale=2.0,
                r_cfg_scale=1.0,
                e_cfg_scale=1.0,
                emo='neutral',
                nfe=10,
                seed=25
            )['d_hat']
        
        # 保存生成的视频
        timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_path = f"../results/training_sample_step_{step}_{timestamp}.mp4"
        
        # 使用generate.py中的保存方法
        save_generated_video(generated_video, output_path, str(audio_path), opt.fps)
        print(f"样本视频已保存: {output_path}")
        
        model.train()
        return output_path
        
    except Exception as e:
        print(f"样本生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    model.train()
    return None