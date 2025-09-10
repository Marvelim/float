# FLOAT Rectified Flow 训练系统

本目录包含了基于 Rectified Flow 方法的 FLOAT 模型训练代码。

## 文件结构

```
training/
├── __init__.py              # 包初始化文件
├── rectified_flow.py        # Rectified Flow 核心实现
├── dataset.py               # 数据加载器
├── config.py                # 训练配置
├── train.py                 # 主训练脚本
├── utils.py                 # 训练工具函数
├── train_rectified_flow.sh  # 训练启动脚本
└── README.md                # 说明文档
```

## 核心组件

### 1. Rectified Flow (`rectified_flow.py`)

实现了 Rectified Flow 算法的核心组件：

- **RectifiedFlow**: 核心流匹配类，实现直线路径插值
- **RectifiedFlowLoss**: 损失函数模块
- **FlowMatchingTrainer**: 训练器类

#### 核心原理

Rectified Flow 使用直线路径连接噪声和数据：
- 插值路径: `x_t = (1-t) * x0 + t * x1`
- 速度场: `v_t = x1 - x0`
- 损失函数: `L = E[||v_θ(x_t, t) - v_t||^2]`

### 2. 数据加载器 (`dataset.py`)

- **FLOATDataset**: 主数据集类，支持视频、音频和情感数据
- **create_dataloader**: 数据加载器工厂函数
- **DataConfig**: 数据配置类

### 3. 配置系统 (`config.py`)

提供了灵活的配置管理：
- **ModelConfig**: 模型相关配置
- **TrainingConfig**: 训练相关配置  
- **DataConfig**: 数据相关配置
- **Config**: 完整配置类

预定义配置：
- `default`: 标准配置
- `small`: 小型配置（用于调试）
- `large`: 大型配置（用于完整训练）

### 4. 训练脚本 (`train.py`)

完整的训练流程实现：
- 支持混合精度训练
- 支持分布式训练
- 支持检查点恢复
- 支持早停机制
- 集成 Weights & Biases 日志记录

## 使用方法

### 1. 基础训练

```bash
# 使用默认配置训练
./training/train_rectified_flow.sh

# 指定配置
./training/train_rectified_flow.sh --config small --batch-size 4 --num-epochs 50
```

### 2. Python 直接调用

```bash
# 使用默认配置
python training/train.py

# 指定参数
python training/train.py --config large --batch-size 16 --learning-rate 5e-5
```

### 3. 从检查点恢复训练

```bash
python training/train.py --resume /path/to/checkpoint.pth
```

### 4. 自定义配置

```python
from training.config import Config

# 创建自定义配置
config = Config()
config.training.batch_size = 16
config.training.learning_rate = 1e-4
config.model.fmt_depth = 12

# 保存配置
config.save("my_config.json")

# 使用自定义配置训练
python training/train.py --config-file my_config.json
```

## 配置参数

### 模型配置 (ModelConfig)

```python
# 核心维度
input_size: int = 512        # 输入尺寸
dim_w: int = 512            # 动作潜在维度
dim_h: int = 1024           # 隐藏层维度
dim_a: int = 512            # 音频维度
dim_e: int = 7              # 情感维度

# Transformer 参数
fmt_depth: int = 12         # FMT 层数
num_heads: int = 16         # 注意力头数
mlp_ratio: float = 4.0      # MLP 比例
```

### 训练配置 (TrainingConfig)

```python
# 基础参数
batch_size: int = 8         # 批次大小
num_epochs: int = 100       # 训练轮数
learning_rate: float = 1e-4 # 学习率

# 优化器
optimizer: str = "adamw"    # 优化器类型
weight_decay: float = 1e-5  # 权重衰减

# 损失函数
loss_type: str = "mse"      # 损失类型
sigma_min: float = 0.0      # 最小噪声
sigma_max: float = 1.0      # 最大噪声
```

### 数据配置 (DataConfig)

```python
# 数据路径
data_root: str = "datasets" # 数据根目录

# 视频参数
video_fps: int = 25         # 视频帧率
sequence_length: int = 16   # 序列长度
prev_frames: int = 4        # 前一帧数量

# 音频参数
audio_sample_rate: int = 16000  # 音频采样率
wav2vec_sec: float = 0.64      # wav2vec 处理长度
```

## 训练流程

1. **数据预处理**: 加载视频和音频，提取特征
2. **模型初始化**: 创建 FLOAT 模型和 FMT
3. **损失计算**: 使用 Rectified Flow 损失
4. **优化更新**: 反向传播和参数更新
5. **验证评估**: 定期验证模型性能
6. **检查点保存**: 保存训练状态和最佳模型

## 监控和日志

### 训练指标
- 训练损失 (train/loss)
- 验证损失 (val/loss)  
- 梯度范数 (train/grad_norm)
- 学习率 (train/lr)
- GPU 内存使用

### 日志系统
- 控制台输出
- 文件日志 (train.log)
- Weights & Biases (可选)

## 性能优化

### 内存优化
- 混合精度训练 (AMP)
- 梯度累积
- 检查点 (Gradient Checkpointing)

### 计算优化  
- 分布式训练 (DDP)
- 数据并行加载
- GPU 内存管理

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 batch_size
   - 启用梯度累积
   - 使用混合精度训练

2. **训练不收敛**
   - 检查学习率设置
   - 验证数据预处理
   - 调整损失函数参数

3. **数据加载错误**
   - 检查数据路径
   - 验证数据格式
   - 确认文件权限

### 调试模式

```bash
# 使用小配置快速调试
python training/train.py --config small --num-epochs 1
```

## 扩展和定制

### 添加新的损失函数

```python
# 在 rectified_flow.py 中添加
class CustomLoss(nn.Module):
    def forward(self, model_output, target):
        # 自定义损失逻辑
        return loss
```

### 添加新的数据增强

```python
# 在 dataset.py 中的 __getitem__ 方法添加
def __getitem__(self, idx):
    # 数据加载
    # 应用增强
    return augmented_data
```

### 自定义回调函数

```python
# 在训练循环中添加自定义逻辑
def custom_callback(trainer, epoch, metrics):
    # 自定义处理逻辑
    pass
```

## 参考资料

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Rectified Flow: A Marginal Preserving Approach to Optimal Transport](https://arxiv.org/abs/2209.14577)
- [FLOAT: Factorized Learning of Object Attributes for Improved Multi-object Multi-part Scene Parsing](相关论文链接)
