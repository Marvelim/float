# FLOAT 训练脚本使用说明

## 概述

基于 `rectified_flow.py` 和 `dataset.py` 创建的 FLOAT 训练脚本，实现了流匹配训练的核心功能。

## 文件结构

```
training/
├── train.py              # 主训练脚本
├── test_train.py         # 训练代码测试脚本
├── dataset.py            # 数据集加载器
├── rectified_flow.py     # Rectified Flow 损失函数
└── README_TRAIN.md       # 本说明文档
```

## 快速开始

### 1. 测试训练代码

在开始训练之前，建议先运行测试脚本验证代码是否正常工作：

```bash
cd /home/mli374/float/training
conda activate FLOAT
python test_train.py
```

### 2. 基本训练

```bash
cd /home/mli374/float/training
conda activate FLOAT
python train.py
```

### 3. 自定义参数训练

```bash
python train.py \
    --data-root ../datasets/ravdess_processed \
    --batch-size 4 \
    --steps 10000 \
    --lr 1e-4 \
    --log-step 100 \
    --sample-step 1000 \
    --save-step 2000
```

## 参数说明

### 必需参数

- `--data-root`: 数据根目录路径（默认: `../datasets/ravdess_processed`）

### 训练参数

- `--batch-size`: 批次大小（默认: 8）
- `--steps`: 训练步数（默认: 200000）
- `--lr`: 学习率（默认: 1e-4）
- `--num-workers`: 数据加载器工作进程数（默认: 4）

### 日志和保存参数

- `--log-step`: 日志记录间隔步数（默认: 500）
- `--sample-step`: 样本生成间隔步数（默认: 2000）
- `--save-step`: 检查点保存间隔步数（默认: 10000）

### 恢复训练

- `--resume`: 从指定检查点恢复训练

```bash
python train.py --resume checkpoints/checkpoint_step_10000.pt
```

## 输出文件

### 检查点文件

训练过程中会在 `checkpoints/` 目录下保存以下文件：

- `checkpoint_step_*.pt`: 定期保存的检查点
- `final_step_*.pt`: 最终训练完成的模型

### 样本文件

在 `samples/` 目录下保存生成的样本：

- `sample_step_*.pt`: 训练过程中生成的样本

### 训练日志

控制台会输出详细的训练信息，包括：

- 当前步骤和总步数
- 损失值（总损失、前一帧损失、当前帧损失）
- 学习率
- 训练时间

## 训练流程

1. **初始化**: 创建模型、优化器、数据加载器等
2. **数据加载**: 从 RAVDESS 数据集加载视频和音频数据
3. **前向传播**: 通过 FMT 模型进行前向传播
4. **损失计算**: 使用 Rectified Flow 计算损失
5. **反向传播**: 更新模型参数
6. **日志记录**: 定期记录训练状态
7. **样本生成**: 定期生成验证样本
8. **检查点保存**: 定期保存模型检查点

## 损失函数

使用 Rectified Flow 损失函数，包含两个部分：

1. **前一帧损失**: 确保模型能正确预测前一帧的运动
2. **当前帧损失**: 确保模型能正确预测当前帧的运动

总损失 = 前一帧损失 + 当前帧损失

## 模型架构

训练脚本使用以下模型组件：

- **FLOAT 主模型**: 包含运动编码器、音频编码器、情感编码器和 FMT
- **FMT (Flow Matching Transformer)**: 核心的流匹配变换器
- **Rectified Flow**: 损失函数模块

## 注意事项

### 内存使用

- 根据 GPU 内存调整批次大小
- 如果内存不足，可以减少 `--batch-size`
- 可以使用 `--num-workers 0` 减少内存使用

### 训练时间

- 完整训练需要很长时间（数小时到数天）
- 建议先用小步数测试（如 `--steps 1000`）
- 使用检查点恢复功能避免训练中断

### 数据要求

- 确保数据路径正确
- 数据需要经过预处理（使用 `dataset.py` 中的预处理功能）
- 支持 RAVDESS 数据集格式

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减少批次大小
   python train.py --batch-size 2
   ```

2. **数据加载错误**
   ```bash
   # 检查数据路径
   ls ../datasets/ravdess_processed/
   ```

3. **模型加载失败**
   ```bash
   # 运行测试脚本检查
   python test_train.py
   ```

### 调试模式

如果遇到问题，可以：

1. 运行测试脚本：`python test_train.py`
2. 使用小批次和少步数进行测试
3. 检查控制台输出的错误信息
4. 验证数据路径和文件权限

## 扩展功能

可以根据需要添加以下功能：

- Wandb 日志记录
- 更详细的验证指标
- 早停机制
- 学习率调度器
- 数据增强
- 多 GPU 训练支持

## 性能优化

- 使用混合精度训练（FP16）
- 调整数据加载器工作进程数
- 使用 SSD 存储数据以提高 I/O 性能
- 定期清理不需要的检查点文件
