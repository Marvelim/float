# generate.py 重构总结

## 重构概述

本次重构将 `generate.py` 中的一些函数替换为 `utils/` 文件夹中已有的工具函数，提高代码复用性和一致性。

## 重构内容

### 1. 添加的导入语句

```python
# 导入工具函数
from utils.checkpoint_utils import load_weight
from utils.video_utils import save_generated_video
from utils.data_utils import get_audio_preprocessor, load_audio
```

### 2. 替换的函数

#### 2.1 音频预处理器初始化
**位置**: `DataProcessor.__init__()` 方法 (第33行)

**原始代码**:
```python
self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)
```

**替换后**:
```python
self.wav2vec_preprocessor = get_audio_preprocessor(opt)
```

**优势**: 
- 使用统一的音频预处理器获取方法
- 避免重复创建预处理器实例
- 保持与训练代码的一致性

#### 2.2 音频加载函数
**位置**: `DataProcessor.default_aud_loader()` 方法 (第68-74行)

**原始代码**:
```python
def default_aud_loader(self, path: str) -> torch.Tensor:
    speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
    return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]
```

**替换后**:
```python
def default_aud_loader(self, path: str) -> torch.Tensor:
    # 使用 utils 中的 load_audio 函数
    audio_tensor = load_audio(path, self.opt)
    # 确保形状是 (1, sequence_length)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    return audio_tensor
```

**优势**:
- 使用统一的音频加载逻辑
- 包含缓存机制，提高加载效率
- 保持与训练代码的一致性

#### 2.3 模型权重加载函数
**位置**: `InferenceAgent.load_weight()` 方法 (第110-112行)

**原始代码**:
```python
def load_weight(self, checkpoint_path: str, rank: int) -> None:
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    with torch.no_grad():
        for model_name, model_param in self.G.named_parameters():
            if model_name in state_dict:
                model_param.copy_(state_dict[model_name].to(rank))
            elif "wav2vec2" in model_name: pass
            else:
                print(f"! Warning; {model_name} not found in state_dict.")
    del state_dict
```

**替换后**:
```python
def load_weight(self, checkpoint_path: str, rank: int) -> None:
    # 使用 utils 中的 load_weight 函数
    load_weight(self.G, checkpoint_path, torch.device(rank))
```

**优势**:
- 使用统一的权重加载逻辑
- 减少代码重复
- 保持与训练代码的一致性

#### 2.4 视频保存函数
**位置**: `InferenceAgent.save_video()` 方法 (第114-117行)

**原始代码**:
```python
def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_filename = temp_video.name
        vid = vid_target_recon.permute(0, 2, 3, 1)
        vid = vid.detach().clamp(-1, 1).cpu()
        vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
        torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)			
        if audio_path is not None:
            with open(os.devnull, 'wb') as f:
                command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(temp_filename, audio_path, video_path)
                subprocess.call(command, shell=True, stdout=f, stderr=f)
            if os.path.exists(video_path):
                os.remove(temp_filename)
        else:
            os.rename(temp_filename, video_path)
        return video_path
```

**替换后**:
```python
def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
    # 使用 utils 中的 save_generated_video 函数
    save_generated_video(vid_target_recon, video_path, audio_path, self.opt.fps)
    return video_path
```

**优势**:
- 使用统一的视频保存逻辑
- 减少代码重复
- 保持与训练代码的一致性

## 重构效果

### 代码减少
- **原始 generate.py**: 221 行
- **重构后 generate.py**: 210 行
- **减少**: 11 行 (约 5%)

### 功能改进
1. **代码复用**: 使用统一的工具函数，避免重复实现
2. **一致性**: 推理和训练代码使用相同的底层函数
3. **维护性**: 修改工具函数时，推理和训练代码都会自动更新
4. **缓存优化**: 音频加载现在包含缓存机制

### 保持的功能
- 所有原有的推理功能都得到保留
- 命令行接口保持不变
- 输出格式和路径保持不变

## 使用方式

重构后的代码使用方式保持不变：

```bash
python generate.py --ref_path assets/sam_altman_512x512.jpg --aud_path assets/audio.wav --emo neutral
```

## 注意事项

1. 确保 `utils/` 文件夹在 Python 路径中
2. 所有工具函数都包含适当的导入语句
3. 重构后的代码通过了语法检查
4. 保持了原有的错误处理和日志输出

## 文件结构

```
float/
├── utils/
│   ├── checkpoint_utils.py (包含 load_weight)
│   ├── video_utils.py (包含 save_generated_video)
│   └── data_utils.py (包含 get_audio_preprocessor, load_audio)
├── generate.py (重构后)
└── GENERATE_REFACTORING_SUMMARY.md
```
