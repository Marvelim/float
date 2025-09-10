# FLOAT Rectified Flow 训练系统

本目录包含了基于 Rectified Flow 方法的 FLOAT 模型训练代码。

## 文件结构

```
training/
├── __init__.py              # 包初始化文件
├── rectified_flow.py        # Rectified Flow 核心实现
├── dataset.py               # 数据加载器
├── train.py                 # 主训练脚本
├── utils.py                 # 训练工具函数
├── train_rectified_flow.sh  # 训练启动脚本
├── example_config.json      # 示例配置文件
└── README.md                # 说明文档

注意：主要的配置文件位于项目根目录的 options/ 文件夹中
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

### 3. 配置系统

配置文件位于项目根目录的 `options/` 文件夹中：

- **BaseOptions**: 基础配置类，定义所有训练和模型参数
- **example_config.json**: 示例配置文件，展示完整的配置结构

主要配置类别：
- **模型配置**: 模型架构相关参数（维度、层数、注意力头等）
- **训练配置**: 训练过程相关参数（学习率、批次大小、优化器等）
- **数据配置**: 数据处理相关参数（路径、采样率、序列长度等）
- **音频配置**: 音频处理相关参数（wav2vec、情感识别等）

配置文件通过命令行参数或 JSON 文件进行加载和覆盖。

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

# 指定参数
./training/train_rectified_flow.sh --batch-size 4 --num-epochs 50 --learning-rate 1e-4
```

### 2. Python 直接调用

```bash
# 使用默认配置
python training/train.py

# 指定参数
python training/train.py --batch-size 16 --learning-rate 5e-5 --fmt-depth 12
```

### 3. 从检查点恢复训练

```bash
python training/train.py --resume /path/to/checkpoint.pth
```

### 4. 使用配置文件

```bash
# 使用示例配置文件
python training/train.py --config-file training/example_config.json

# 使用自定义配置文件
python training/train.py --config-file /path/to/your/config.json
```

### 5. 配置参数覆盖

```bash
# 通过命令行参数覆盖配置文件中的设置
python training/train.py --config-file training/example_config.json --batch-size 32 --learning-rate 2e-4
```

## 配置参数

配置参数通过 `options/base_options.py` 中的 `BaseOptions` 类定义，主要分为以下几个类别：

### 模型配置

```python
# 核心维度
input_size: int = 512        # 输入图像尺寸
dim_w: int = 512            # 面部维度
dim_h: int = 1024           # 隐藏层维度
dim_a: int = 512            # 音频维度
dim_e: int = 7              # 情感维度
dim_m: int = 20             # 正交基维度

# FMT Transformer 参数
fmt_depth: int = 8          # FMT 层数
num_heads: int = 8          # 注意力头数
mlp_ratio: float = 4.0      # MLP 比例
num_prev_frames: int = 10   # 前一帧数量
```

### 训练配置

```python
# 基础参数
batch_size: int = 8         # 批次大小
learning_rate: float = 1e-4 # 学习率
weight_decay: float = 1e-5  # 权重衰减

# 优化器
optimizer: str = "adamw"    # 优化器类型
beta1: float = 0.9          # Adam beta1
beta2: float = 0.999        # Adam beta2

# 学习率调度
lr_scheduler: str = "cosine" # 学习率调度器
lr_warmup_epochs: int = 5    # 预热轮数
lr_min: float = 1e-6         # 最小学习率
```

### 音频配置

```python
# 音频处理
sampling_rate: int = 16000  # 音频采样率
wav2vec_sec: float = 2.0    # wav2vec 窗口长度
audio_marcing: int = 2      # 相邻帧数量
attention_window: int = 2   # 注意力窗口大小

# 模型路径
wav2vec_model_path: str     # wav2vec 模型路径
audio2emotion_path: str     # 情感识别模型路径
```

### 视频配置

```python
# 视频参数
fps: float = 25.0           # 视频帧率
input_nc: int = 3           # 输入图像通道数
```

### ODE 求解器配置

```python
# ODE 求解器
nfe: int = 10               # 函数评估次数
torchdiffeq_ode_method: str = "euler"  # ODE 求解方法
ode_atol: float = 1e-5      # 绝对容差
ode_rtol: float = 1e-5      # 相对容差

# 分类器自由引导
a_cfg_scale: float = 2.0    # 音频引导尺度
e_cfg_scale: float = 1.0    # 情感引导尺度
r_cfg_scale: float = 1.0    # 参考引导尺度
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
