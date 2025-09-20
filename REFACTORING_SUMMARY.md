# FLOAT 训练代码重构总结

## 重构概述

本次重构将 `training/train.py` 中与训练无关的代码移动到了 `utils/` 文件夹下的专门工具文件中，使代码结构更加清晰和模块化。

## 重构内容

### 1. 创建的工具文件

#### `utils/wandb_utils.py`
- **功能**: Wandb 实验跟踪相关功能
- **包含函数**:
  - `init_wandb(opt)`: 初始化 wandb
  - `log_to_wandb(step, loss_dict, lr, elapsed_time, opt)`: 记录训练指标
  - `log_sample_to_wandb(step, video_path, opt)`: 记录样本视频
  - `finish_wandb()`: 完成 wandb 运行

#### `utils/memory_utils.py`
- **功能**: 内存管理和缓存相关功能
- **包含函数**:
  - `clear_memory()`: 清理GPU内存
  - `clear_cache()`: 清理数据缓存
  - `get_cache_info()`: 获取缓存信息
  - `get_video_cache()`: 获取视频缓存
  - `get_audio_cache()`: 获取音频缓存
  - `get_cache_size_limit()`: 获取缓存大小限制

#### `utils/device_utils.py`
- **功能**: 设备管理相关功能
- **包含函数**:
  - `setup_device()`: 设置训练设备
  - `get_device_info()`: 获取设备信息

#### `utils/checkpoint_utils.py`
- **功能**: 检查点保存和加载相关功能
- **包含函数**:
  - `save_checkpoint(model, optimizer, scheduler, step, loss, save_path)`: 保存检查点
  - `load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)`: 加载检查点
  - `load_weight(model, checkpoint_path, device)`: 加载模型权重

#### `utils/data_utils.py`
- **功能**: 数据处理和加载相关功能
- **包含函数**:
  - `get_audio_preprocessor(opt)`: 获取全局音频预处理器
  - `load_video(video_path, opt)`: 加载视频帧
  - `load_audio(audio_path, opt)`: 加载音频
  - `extract_motion_latent(model, video_frames)`: 提取运动潜在表示
  - `extract_motion_latent_batch(model, batch_videos)`: 批量提取运动潜在表示
  - `get_sequence_indices(total_frames, opt)`: 获取序列索引
  - `load(batch_data, model, device, opt)`: 批处理加载数据
  - `get_batch_sample(real_data, opt)`: 从完整数据中获取训练样本

#### `utils/video_utils.py`
- **功能**: 视频生成和保存相关功能
- **包含函数**:
  - `generate_sample(model, device, opt, step)`: 生成样本视频
  - `save_generated_video(video_tensor, output_path, audio_path, fps)`: 保存生成的视频
  - `save_test_video(img_tensor, output_path, fps)`: 保存测试视频
  - `ensure(model, new_batch, opt, step)`: 确保模型输出并保存为视频
  - `save_data_out_as_video(data_out, opt, step, audio_path)`: 保存 data_out 为视频

#### `utils/model_utils.py`
- **功能**: 模型创建和管理相关功能
- **包含函数**:
  - `create_model(opt, device)`: 创建和初始化模型
  - `create_optimizer(model, opt)`: 创建优化器
  - `create_scheduler(optimizer, opt)`: 创建学习率调度器
  - `get_model_info(model)`: 获取模型信息

### 2. 重构后的 `training/train.py`

重构后的训练脚本现在只包含核心训练逻辑：

- **保留的函数**:
  - `train_step()`: 执行一个训练步骤
  - `train()`: 主训练循环
  - `main()`: 主函数

- **导入的工具函数**:
  - 从各个工具文件导入所需的函数
  - 保持了原有的功能，但代码结构更加清晰

## 重构优势

1. **模块化**: 将不同功能的代码分离到专门的工具文件中
2. **可维护性**: 每个工具文件专注于特定功能，便于维护和修改
3. **可重用性**: 工具函数可以在其他脚本中重复使用
4. **代码清晰度**: 训练脚本现在只关注核心训练逻辑
5. **测试友好**: 每个工具模块可以独立测试

## 使用方式

重构后的代码使用方式保持不变：

```bash
python training/train.py --batch-size 2 --steps 1000 --use-wandb
```

所有原有的命令行参数和功能都得到保留。

## 注意事项

1. 确保 `utils/` 文件夹在 Python 路径中
2. 所有工具文件都包含适当的导入语句
3. 全局变量（如缓存）现在在相应的工具文件中管理
4. 重构后的代码通过了语法检查，确保没有语法错误

## 文件结构

```
float/
├── utils/
│   ├── __init__.py
│   ├── wandb_utils.py
│   ├── memory_utils.py
│   ├── device_utils.py
│   ├── checkpoint_utils.py
│   ├── data_utils.py
│   ├── video_utils.py
│   └── model_utils.py
├── training/
│   └── train.py (重构后)
└── REFACTORING_SUMMARY.md
```
