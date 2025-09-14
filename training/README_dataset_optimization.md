# FLOAT 数据集优化版本

## 概述

`dataset_test.py` 是 `dataset.py` 的优化版本，主要解决了原版本中 `__getitem__` 方法执行缓慢的问题。

## 主要优化

### 1. 预处理阶段移动模型计算
- **音频编码器推理**: `audio_encoder.inference()` 移到预处理阶段
- **运动潜在表示提取**: `_extract_motion_latent()` 移到预处理阶段  
- **参考运动提取**: 参考帧的运动特征提取移到预处理阶段

### 2. 缓存系统
- 预处理结果保存到磁盘缓存
- 支持训练集和测试集分别缓存
- 避免重复预处理，大幅提升后续加载速度

### 3. 快速 `__getitem__`
- 只进行简单的张量切片操作
- 无模型推理计算
- 显著提升数据加载速度

## 使用方法

### 基本使用

```python
from dataset_test import create_dataloader_optimized

# 创建优化的数据加载器
dataloader = create_dataloader_optimized(
    data_root="/path/to/data",
    batch_size=8,
    train=True,
    opt=opt,  # 包含模型配置的选项
    force_preprocess=False  # 首次运行会自动预处理
)

# 使用数据加载器
for batch in dataloader:
    # 训练代码
    pass
```

### 缓存管理

```python
from dataset_test import check_cache_status, clear_cache

# 检查缓存状态
check_cache_status("/path/to/data")

# 清理缓存（如果数据或模型发生变化）
clear_cache("/path/to/data")

# 强制重新预处理
dataloader = create_dataloader_optimized(
    data_root="/path/to/data",
    opt=opt,
    force_preprocess=True  # 强制重新预处理
)
```

### 性能测试

```python
# 运行性能测试脚本
python test_dataset_performance.py
```

## 文件结构

```
training/
├── dataset.py                    # 原始数据集
├── dataset_test.py              # 优化数据集
├── test_dataset_performance.py  # 性能测试脚本
└── README_dataset_optimization.md  # 说明文档
```

## 预期性能提升

根据优化内容，预期可以获得以下性能提升：

- **首次运行**: 需要预处理时间，可能比原版本慢
- **后续运行**: `__getitem__` 速度提升 **5-20倍**
- **整体训练**: 数据加载不再是瓶颈

## 注意事项

### 1. 首次运行
- 首次运行需要预处理所有数据，可能需要较长时间
- 预处理过程中会显示进度条
- 预处理完成后会自动保存缓存

### 2. 存储空间
- 缓存文件会占用额外的磁盘空间
- 缓存大小取决于数据集大小和特征维度
- 建议确保有足够的磁盘空间

### 3. 数据一致性
- 如果原始数据发生变化，需要重新预处理
- 如果模型配置发生变化，需要重新预处理
- 使用 `force_preprocess=True` 强制重新预处理

### 4. 内存使用
- 预处理阶段会加载模型到GPU
- 预处理完成后模型会被释放
- `__getitem__` 阶段不使用GPU

## 兼容性

- 输出数据格式与原版本完全兼容
- 可以直接替换原版本使用
- 支持相同的配置参数

## 故障排除

### 缓存损坏
```python
# 清理缓存并重新预处理
clear_cache("/path/to/data")
dataloader = create_dataloader_optimized(
    data_root="/path/to/data",
    opt=opt,
    force_preprocess=True
)
```

### 内存不足
```python
# 减少批次大小
dataloader = create_dataloader_optimized(
    data_root="/path/to/data",
    batch_size=4,  # 减少批次大小
    opt=opt
)
```

### CUDA错误
- 确保在预处理阶段有足够的GPU内存
- 如果GPU内存不足，可以考虑使用CPU进行预处理

## 开发建议

1. **开发阶段**: 使用小数据集测试，确保功能正常
2. **生产阶段**: 使用完整数据集，享受性能提升
3. **调试阶段**: 可以临时使用原版本进行对比

## 更新日志

- **v1.0**: 初始版本，实现基本预处理和缓存功能
- 支持音频编码器和运动编码器的预处理
- 实现磁盘缓存系统
- 提供性能测试工具
