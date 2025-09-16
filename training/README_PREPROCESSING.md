# FLOAT 数据集预处理指南

## 概述

为了提高训练效率，我们将数据集预处理从训练脚本中分离出来。现在你可以先预处理数据集，然后再进行训练。

## 使用流程

### 1. 清除现有缓存并预处理数据集

```bash
# 清除所有缓存并预处理训练集和测试集
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --clear_cache_first

# 只预处理训练集
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --clear_cache_first --train_only

# 只预处理测试集
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --clear_cache_first --test_only
```

### 2. 检查缓存状态

```bash
# 检查缓存状态
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --check_status
```

### 3. 运行训练

预处理完成后，运行训练脚本：

```bash
python training/train.py --data_root datasets/ravdess_processed
```

## 命令行参数

### preprocess_dataset.py 参数

- `--data_root`: 数据根目录路径（必需）
- `--cache_dir`: 自定义缓存目录路径（可选，默认为 data_root/cache）
- `--force_preprocess`: 强制重新预处理，忽略现有缓存
- `--clear_cache_first`: 预处理前先清除所有相关缓存
- `--train_only`: 只预处理训练集
- `--test_only`: 只预处理测试集
- `--check_status`: 只检查缓存状态，不进行预处理

### train.py 的变化

训练脚本现在假设缓存已经预处理好了，不会强制重新预处理。如果需要重新预处理，请使用独立的预处理脚本。

## 优势

1. **分离关注点**: 预处理和训练分开，便于调试和管理
2. **节省时间**: 预处理一次，可以多次训练
3. **灵活性**: 可以选择只预处理训练集或测试集
4. **缓存管理**: 提供完整的缓存清理和状态检查功能

## 注意事项

1. **GPU 内存**: 预处理需要足够的 GPU 内存来加载模型
2. **存储空间**: 缓存文件可能很大，确保有足够的磁盘空间
3. **数据变化**: 如果数据或模型发生变化，需要重新预处理

## 故障排除

### 内存不足
如果遇到 GPU 内存不足的问题：
1. 确保没有其他程序占用 GPU
2. 考虑分批预处理（先训练集，再测试集）

### 缓存损坏
如果缓存文件损坏：
```bash
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --clear_cache_first --force_preprocess
```

### 检查预处理结果
```bash
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --check_status
```

## 示例工作流

```bash
# 1. 清除所有缓存
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --clear_cache_first --check_status

# 2. 预处理数据集
python training/preprocess_dataset.py --data_root datasets/ravdess_processed

# 3. 检查预处理结果
python training/preprocess_dataset.py --data_root datasets/ravdess_processed --check_status

# 4. 开始训练
python training/train.py --data_root datasets/ravdess_processed --steps 50000
```

## 快速开始

你也可以使用提供的示例脚本：

```bash
# 运行交互式预处理示例
./training/run_preprocessing_example.sh
```

## 文件说明

- `preprocess_dataset.py`: 独立的数据集预处理脚本
- `train.py`: 修改后的训练脚本（假设缓存已预处理）
- `run_preprocessing_example.sh`: 交互式预处理示例脚本
- `README_PREPROCESSING.md`: 本文档
