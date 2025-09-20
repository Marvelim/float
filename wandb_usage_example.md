# FLOAT 训练脚本 wandb 集成使用说明

## 概述

已成功在 `train.py` 中集成了 wandb 实验跟踪功能，可以记录训练指标、学习率、生成的样本视频等。

## 新增的 wandb 选项

### 在 `base_options.py` 中添加的选项：

- `--use_wandb`: 是否启用 wandb 跟踪（布尔标志）
- `--wandb_project`: wandb 项目名称（默认: 'float-training'）
- `--wandb_entity`: wandb 实体名称（可选）
- `--wandb_run_name`: wandb 运行名称（可选，会自动生成时间戳）
- `--wandb_tags`: wandb 标签列表（可选）

## 使用方法

### 1. 基本使用（启用 wandb）

```bash
python training/train.py --use-wandb
```

### 2. 指定项目名称

```bash
python training/train.py --use-wandb --wandb-project "my-float-experiment"
```

### 3. 指定运行名称和标签

```bash
python training/train.py --use-wandb \
    --wandb-project "float-training" \
    --wandb-run-name "experiment-v1" \
    --wandb-tags "baseline" "512x512" "flow-matching"
```

### 4. 完整示例

```bash
python training/train.py \
    --data-root ../datasets/ravdess_processed \
    --batch-size 4 \
    --steps 10000 \
    --lr 1e-4 \
    --use-wandb \
    --wandb-project "float-experiments" \
    --wandb-run-name "baseline-experiment" \
    --wandb-tags "baseline" "ravdess" "flow-matching"
```

## 记录的内容

### 训练指标
- `loss/total`: 总损失
- `loss/prev`: 前一帧损失
- `loss/current`: 当前帧损失
- `learning_rate`: 学习率
- `time/elapsed`: 经过时间
- `time/step`: 每步时间

### 样本视频
- `samples/video`: 生成的样本视频（在 `sample_step` 间隔时记录）

### 配置信息
- 所有训练参数会自动记录到 wandb 配置中

## 注意事项

1. **wandb 安装**: 确保已安装 wandb：
   ```bash
   pip install wandb
   ```

2. **wandb 登录**: 首次使用需要登录：
   ```bash
   wandb login
   ```

3. **可选功能**: 如果未安装 wandb，程序会显示警告但继续正常运行，只是不会记录到 wandb。

4. **视频记录**: 样本视频会保存到 `../results/` 目录，同时上传到 wandb。

5. **内存管理**: wandb 记录不会显著影响训练性能，但建议在生成样本后清理内存。

## 故障排除

- 如果遇到 wandb 相关错误，可以添加 `--no-use-wandb` 或直接移除 `--use-wandb` 标志来禁用 wandb
- 检查网络连接，确保可以访问 wandb 服务
- 确保 wandb API 密钥正确配置
