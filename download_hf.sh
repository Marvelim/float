#!/bin/bash

# 设置要下载的模型仓库 ID
# Set the repository ID of the model to be downloaded
MODEL_ID="r-f/wav2vec-english-speech-emotion-recognition"

# 设置你希望将模型文件保存到的本地目录
# Set the local directory where you want to save the model files
# 注意：如果目录不存在，huggingface-cli 会自动创建它
# Note: huggingface-cli will create the directory if it doesn't exist
LOCAL_DIR="./checkpoints/wav2vec-english-speech-emotion-recognition"

# 打印开始信息
# Print start message
echo "===================================================================="
echo "正在开始下载模型: ${MODEL_ID}"
echo "模型将被保存到: ${LOCAL_DIR}"
echo "===================================================================="

# 使用 huggingface-cli 下载模型
# Use huggingface-cli to download the model
# --repo_type model 是默认选项，但明确写出更清晰
# --repo_type model is the default, but it's clearer to be explicit
# --local-dir 指定了本地存储路径
# --local-dir specifies the local storage path
huggingface-cli download \
    ${MODEL_ID} \
    --repo-type model \
    --local-dir ${LOCAL_DIR} \
    --local-dir-use-symlinks False

# 检查上一条命令的退出码，判断是否下载成功
# Check the exit code of the last command to determine if the download was successful
if [ $? -eq 0 ]; then
    echo "===================================================================="
    echo "模型 ${MODEL_ID} 已成功下载到 ${LOCAL_DIR}"
    echo "目录内容:"
    ls -l ${LOCAL_DIR}
    echo "===================================================================="
else
    echo "===================================================================="
    echo "错误：模型下载失败。请检查网络连接或错误信息。"
    echo "===================================================================="
fi