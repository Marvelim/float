# Wandb 日志记录使用说明

## 基本用法

### 1. 启用 wandb 日志记录（默认）
```bash
python train.py --num_workers 0
```

### 2. 自定义项目名称
```bash
python train.py --num_workers 0 --wandb_project "my-float-project"
```

### 3. 自定义运行名称
```bash
python train.py --num_workers 0 --wandb_name "experiment_v1"
```

### 4. 禁用 wandb 日志记录
```bash
python train.py --num_workers 0 --disable_wandb
```

## 记录的指标

训练过程中会自动记录以下指标到 wandb：

### 训练指标
- `train/loss`: 当前步骤的损失值
- `train/avg_loss`: 平均损失值
- `train/learning_rate`: 当前学习率
- `train/global_step`: 全局训练步数

### 检查点指标
- `checkpoint/saved`: 检查点是否已保存
- `checkpoint/step`: 保存检查点时的步数
- `checkpoint/loss`: 保存检查点时的损失值

### 最终模型指标
- `final_model/saved`: 最终模型是否已保存
- `final_model/step`: 最终模型的训练步数
- `final_model/loss`: 最终模型的损失值

## 配置参数

训练脚本会自动记录以下配置参数到 wandb：

- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `steps`: 总训练步数
- `mixed_precision`: 是否使用混合精度
- `seed`: 随机种子
- `data_root`: 数据根目录
- `input_size`: 输入图像大小
- `dim_w`: 风格维度
- `dim_m`: 运动维度
- `dim_e`: 情感维度
- `fps`: 视频帧率
- `wav2vec_sec`: wav2vec 音频长度（秒）
- `num_prev_frames`: 前一帧数量
- `wandb_project`: wandb 项目名称
- `wandb_name`: wandb 运行名称

## 注意事项

1. wandb 只在主进程中初始化，避免多进程冲突
2. 如果禁用 wandb，所有相关日志记录都会被跳过
3. 运行名称会自动包含时间戳，确保唯一性
4. 训练完成后会自动调用 `wandb.finish()` 完成运行
