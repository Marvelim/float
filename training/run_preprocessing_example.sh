#!/bin/bash

# FLOAT 数据集预处理示例脚本
# 这个脚本演示了如何使用新的预处理工具

echo "FLOAT 数据集预处理示例"
echo "======================"

# 设置数据路径
DATA_ROOT="datasets/ravdess_processed"

echo "步骤 1: 检查当前缓存状态"
python training/preprocess_dataset.py --data_root $DATA_ROOT --check_status

echo ""
echo "步骤 2: 清除现有缓存"
python training/preprocess_dataset.py --data_root $DATA_ROOT --clear_cache_first --check_status

echo ""
echo "步骤 3: 预处理训练集（这可能需要一些时间...）"
echo "注意: 如果你想先测试，可以添加 --train_only 参数只预处理训练集"
read -p "是否继续预处理？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python training/preprocess_dataset.py --data_root $DATA_ROOT --train_only
    
    echo ""
    echo "步骤 4: 检查预处理结果"
    python training/preprocess_dataset.py --data_root $DATA_ROOT --check_status
    
    echo ""
    echo "预处理完成！现在你可以运行训练脚本："
    echo "python training/train.py --data_root $DATA_ROOT --steps 1000"
else
    echo "预处理已取消"
fi
