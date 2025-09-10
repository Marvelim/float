#!/bin/bash

# FLOAT Rectified Flow 训练脚本
# FLOAT Rectified Flow Training Script

set -e

# 默认参数
CONFIG="default"
OUTPUT_DIR="/root/autodl-tmp/float/experiments"
BATCH_SIZE=8
LEARNING_RATE=1e-4
NUM_EPOCHS=100
DEVICE="cuda"
NUM_WORKERS=4

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --config CONFIG          配置名称 (default: default)"
            echo "  --output-dir DIR         输出目录 (default: /root/autodl-tmp/float/experiments)"
            echo "  --batch-size SIZE        批次大小 (default: 8)"
            echo "  --learning-rate LR       学习率 (default: 1e-4)"
            echo "  --num-epochs EPOCHS      训练轮数 (default: 100)"
            echo "  --device DEVICE          设备 (default: cuda)"
            echo "  --resume CHECKPOINT      从检查点恢复训练"
            echo "  --help                   显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "FLOAT Rectified Flow 训练"
echo "=========================================="
echo "配置: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "训练轮数: $NUM_EPOCHS"
echo "设备: $DEVICE"
if [ ! -z "$RESUME" ]; then
    echo "恢复训练: $RESUME"
fi
echo "=========================================="

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 设置 Python 路径
export PYTHONPATH="/root/autodl-tmp/float:$PYTHONPATH"

# 构建训练命令
TRAIN_CMD="python /root/autodl-tmp/float/training/train.py"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG"
TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --learning-rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --num-epochs $NUM_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --device $DEVICE"

if [ ! -z "$RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

echo "执行命令: $TRAIN_CMD"
echo "=========================================="

# 执行训练
eval $TRAIN_CMD
