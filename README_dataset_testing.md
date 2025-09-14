# 数据集测试脚本使用指南

本目录包含了用于测试 FLOAT 数据集加载功能的多个测试脚本。这些脚本可以帮助你验证数据集配置是否正确，数据加载是否正常工作。

## 🔧 配置系统说明

**重要**: 项目主要使用 `options/base_options.py` 中的 `BaseOptions` 类管理所有配置。所有参数都有默认值，你可以通过命令行参数覆盖这些默认值。

- ✅ **主要配置**: `options/base_options.py` (BaseOptions 类)
- ⚠️ **可选配置**: `training/example_config.json` (仅作为示例和高级用法)

## 📁 测试脚本概览

### 1. `simple_dataset_test.py` - 简单测试脚本 ⭐⭐⭐ **推荐首选**
**用途**: 基于 BaseOptions 的简单快速测试，无需 JSON 配置文件
**推荐**: ⭐⭐⭐ 最简单直接的测试方式

```bash
# 基本测试（使用默认配置）
python simple_dataset_test.py

# 指定数据根目录
python simple_dataset_test.py --data_root /path/to/your/data

# 指定其他参数
python simple_dataset_test.py --data_root /path/to/data --batch_size 4
```

### 2. `test_dataset_config.py` - 配置检查脚本
**用途**: 检查数据集配置是否正确，包括路径、参数、依赖库等
**推荐**: ⭐⭐ 需要详细配置检查时使用

```bash
# 基本配置检查
python test_dataset_config.py

# 指定数据根目录
python test_dataset_config.py --data_root /path/to/your/data

# 创建示例配置文件（可选）
python test_dataset_config.py --create_sample
```

### 3. `quick_test_dataset.py` - 快速数据集测试
**用途**: 快速验证数据集是否能正常加载和工作
**推荐**: ⭐⭐ 需要更多测试选项时使用

```bash
# 快速测试
python quick_test_dataset.py

# 指定数据根目录
python quick_test_dataset.py --data_root /path/to/your/data

# 检查数据目录结构
python quick_test_dataset.py --check_structure

# 使用最小配置测试
python quick_test_dataset.py --custom_config
```

### 4. `test_dataset.py` - 完整数据集测试
**用途**: 全面测试数据集功能，包括性能、错误处理等
**推荐**: ⭐⭐ 深入测试时使用

```bash
# 完整测试
python test_dataset.py

# 使用自定义配置
python test_dataset.py --config your_config.json

# 指定数据根目录
python test_dataset.py --data_root /path/to/your/data
```

## 🚀 推荐的测试流程

### 🥇 最简单方式 (推荐新手)
```bash
# 一步到位的简单测试
python simple_dataset_test.py --data_root /path/to/your/data
```

### 🥈 标准流程 (推荐有经验用户)

#### 步骤 1: 简单测试
```bash
python simple_dataset_test.py
```
这会检查：
- ✅ BaseOptions 配置解析
- ✅ 数据路径是否存在
- ✅ 模型文件是否存在
- ✅ 数据集创建和加载

#### 步骤 2: 详细配置检查（可选）
```bash
python test_dataset_config.py
```
这会检查：
- ✅ 依赖库是否安装
- ✅ 参数配置是否合理
- ✅ 路径配置详细检查

#### 步骤 3: 完整测试（可选）
```bash
python test_dataset.py
```
这会进行：
- ✅ 批量数据加载测试
- ✅ 性能测试
- ✅ 错误处理测试
- ✅ 数据完整性检查

## 🔧 常见问题和解决方案

### 问题 1: 数据目录不存在
```
❌ 数据根目录不存在: /path/to/data
```
**解决方案**:
- 确保数据目录存在
- 使用 `--data_root` 参数指定正确路径
- 运行 `python quick_test_dataset.py --check_structure` 查看可用目录

### 问题 2: 缺少模型文件
```
❌ Wav2Vec2模型路径不存在: ./checkpoints/wav2vec2-base-960h
```
**解决方案**:
- 下载预训练模型到指定路径
- 或修改配置文件中的模型路径

### 问题 3: 依赖库缺失
```
❌ 缺少依赖库: OpenCV (cv2)
```
**解决方案**:
```bash
pip install opencv-python librosa transformers torch
```

### 问题 4: 数据集为空
```
⚠️ 警告: 数据集为空
```
**解决方案**:
- 检查数据目录结构是否正确
- 确保有 `ravdess_processed` 或 `ravdess_raw` 子目录
- 确保目录中包含视频和音频文件

## 📊 测试输出说明

### 成功输出示例
```
🎉 所有测试通过！数据集加载功能正常。
📊 数据集大小: 1440
✅ 数据项包含所有必要的键
⏱️ 平均加载时间: 1.23秒
```

### 失败输出示例
```
❌ 数据集创建失败: FileNotFoundError
💥 部分测试失败，请检查数据集配置和数据文件。
```

## 🛠️ 自定义配置

### 创建自定义配置文件
```bash
python test_dataset_config.py --create_sample
```

### 配置文件示例
```json
{
  "model": {
    "input_size": 512,
    "dim_w": 512,
    "dim_m": 20,
    "wav2vec_model_path": "./checkpoints/wav2vec2-base-960h"
  },
  "data": {
    "data_root": "/your/data/path",
    "num_workers": 4
  }
}
```

## 📝 测试结果解读

### 配置检查结果
- **✅ 通过**: 配置项正确
- **⚠️ 警告**: 配置可能有问题但不影响运行
- **❌ 错误**: 必须修复的问题

### 数据加载测试结果
- **数据集大小**: 可用的数据项数量
- **加载时间**: 单个数据项的平均加载时间
- **数据形状**: 各个数据项的张量形状
- **数据类型**: 数据的类型检查结果

## 🎯 性能优化建议

### 如果加载速度慢
1. 减少 `num_workers` 参数
2. 使用 SSD 存储数据
3. 预处理数据到更快的格式
4. 减少 `input_size` 参数

### 如果内存不足
1. 减少 `batch_size`
2. 减少 `num_prev_frames`
3. 使用更小的 `input_size`

## 📞 获取帮助

如果测试脚本运行遇到问题：

1. 首先运行配置检查: `python test_dataset_config.py`
2. 查看错误信息和建议的解决方案
3. 检查数据目录结构和文件完整性
4. 确保所有依赖库已正确安装

测试脚本会提供详细的错误信息和修复建议，请仔细阅读输出信息。
