# 使用优化版本数据集进行训练

## 概述

`train.py` 已经更新为使用优化版本的数据集 (`dataset_test.py`)，这将显著提升训练时的数据加载速度。

## 主要变化

### 1. 数据集替换
- ✅ 使用 `FLOATDatasetOptimized` 替代 `FLOATDataset`
- ✅ 使用 `create_dataloader_optimized` 创建数据加载器
- ✅ 移除了数据集中模型组件的管理代码

### 2. 新增命令行参数
- `--force_preprocess`: 强制重新预处理数据集（忽略缓存）
- `--cache_dir`: 指定缓存目录路径

### 3. 预处理阶段
- 首次运行时会自动进行预处理
- 预处理结果会缓存到磁盘
- 后续运行会快速加载缓存

## 使用方法

### 基本训练命令

```bash
# 基本训练（首次运行会自动预处理）
python train.py --data_root /path/to/data --steps 50000

# 指定缓存目录
python train.py --data_root /path/to/data --cache_dir /path/to/cache --steps 50000

# 强制重新预处理（当数据或模型发生变化时）
python train.py --data_root /path/to/data --force_preprocess --steps 50000
```

### 完整训练命令示例

```bash
python train.py \
    --data_root /home/mli374/float/data \
    --steps 50000 \
    --batch_size 8 \
    --lr 1e-4 \
    --mixed_precision fp16 \
    --num_workers 4 \
    --wandb_project float-optimized \
    --wandb_name experiment_1 \
    --cache_dir /tmp/float_cache \
    --force_preprocess
```

## 性能对比

### 原版本 vs 优化版本

| 阶段 | 原版本 | 优化版本 | 提升 |
|------|--------|----------|------|
| 首次运行 | 正常速度 | 需要预处理时间 | 可能较慢 |
| 后续运行 | 每次 `__getitem__` 都有模型推理 | 只有简单的张量切片 | **5-20x** |
| 整体训练 | 数据加载是瓶颈 | 数据加载不再是瓶颈 | 显著提升 |

## 注意事项

### 1. 首次运行
- 首次运行需要预处理时间，请耐心等待
- 预处理过程会显示进度条
- 预处理完成后会自动保存缓存

### 2. 存储空间
- 缓存文件会占用额外磁盘空间
- 建议确保有足够的磁盘空间

### 3. 数据一致性
- 如果原始数据发生变化，使用 `--force_preprocess`
- 如果模型配置发生变化，使用 `--force_preprocess`

### 4. 缓存管理

```bash
# 检查缓存状态
python -c "
from training.dataset_test import check_cache_status
check_cache_status('/path/to/data')
"

# 清理缓存
python -c "
from training.dataset_test import clear_cache
clear_cache('/path/to/data')
"
```

## 故障排除

### 问题：预处理失败
**解决方案**：
1. 检查数据路径是否正确
2. 检查模型路径是否存在
3. 确保有足够的GPU内存进行预处理

### 问题：缓存损坏
**解决方案**：
```bash
python train.py --data_root /path/to/data --force_preprocess
```

### 问题：磁盘空间不足
**解决方案**：
1. 清理旧的缓存文件
2. 使用 `--cache_dir` 指定其他磁盘位置

## 开发建议

### 开发阶段
- 使用小数据集进行测试
- 使用 `--force_preprocess` 确保数据一致性

### 生产阶段
- 使用完整数据集
- 让系统自动管理缓存

### 调试阶段
- 可以临时切换回原版本进行对比
- 使用性能测试脚本验证提升效果

## 监控和日志

### Wandb 集成
优化版本会在 wandb 中记录以下额外信息：
- `dataset_version`: "optimized"
- `force_preprocess`: 是否强制重新预处理
- `cache_dir`: 缓存目录路径

### 控制台输出
- 预处理进度条
- 数据集大小信息
- 缓存状态信息

## 兼容性

- ✅ 输出数据格式与原版本完全兼容
- ✅ 支持所有原有的训练参数
- ✅ 支持分布式训练
- ✅ 支持混合精度训练

## 更新日志

- **v1.0**: 集成优化数据集到训练脚本
- 添加缓存管理参数
- 更新 wandb 配置
- 移除不必要的模型组件管理代码
