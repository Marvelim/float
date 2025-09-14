#!/usr/bin/env python3

import torch

print("测试 PyTorch repeat 操作")
print("=" * 50)

# 模拟修复后的情况：推理时r_s已经有正确的维度
r_s_original = torch.randn(1, 512)  # 原始推理时的r_s
r_s_fixed = r_s_original.unsqueeze(1)  # 修复：添加维度 [1, 512] -> [1, 1, 512]

# 模拟CFG后的张量形状
wr = torch.cat([r_s_fixed, r_s_fixed, r_s_fixed], dim=0)  # CFG: [3, 1, 512]
wa = torch.randn(3, 60, 512)  # CFG后的音频特征
we = torch.randn(3, 1, 7)  # CFG后的情感特征

print("原始张量形状:")
print(f"  wr.shape: {wr.shape}")
print(f"  wa.shape: {wa.shape}")
print(f"  we.shape: {we.shape}")
print(f"  wa.shape[1] (seq_len): {wa.shape[1]}")

print("\n执行 repeat 操作:")
print("  wr.repeat(1, wa.shape[1], 1)")
print("  we.repeat(1, wa.shape[1], 1)")

# 执行repeat操作
wr_repeated = wr.repeat(1, wa.shape[1], 1)
we_repeated = we.repeat(1, wa.shape[1], 1)

print("\nrepeat后的张量形状:")
print(f"  wr_repeated.shape: {wr_repeated.shape}")
print(f"  we_repeated.shape: {we_repeated.shape}")

print("\n期望的形状:")
print(f"  wr应该是: [3, 60, 512]")
print(f"  we应该是: [3, 60, 7]")

print("\n结果分析:")
if wr_repeated.shape == torch.Size([3, 60, 512]):
    print("  ✅ wr repeat操作正确")
else:
    print(f"  ❌ wr repeat操作错误: 期望[3, 60, 512], 实际{wr_repeated.shape}")

if we_repeated.shape == torch.Size([3, 60, 7]):
    print("  ✅ we repeat操作正确")
else:
    print(f"  ❌ we repeat操作错误: 期望[3, 60, 7], 实际{we_repeated.shape}")

print("\n测试torch.cat操作:")
try:
    c = torch.cat([wr_repeated, wa, we_repeated], dim=-1)
    print(f"  ✅ torch.cat成功: {c.shape}")
except Exception as e:
    print(f"  ❌ torch.cat失败: {e}")

print("\n" + "=" * 50)
print("PyTorch版本:", torch.__version__)
