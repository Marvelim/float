#!/usr/bin/env python3
"""
ensure 函数使用示例
展示如何在训练过程中使用 ensure 函数保存 data_out 为视频
"""

import torch
from training.train import ensure, save_data_out_as_video
from options.base_options import BaseOptions

def example_usage():
    """使用示例"""
    
    # 1. 解析选项
    opt = BaseOptions().parse()
    
    # 2. 模拟模型和批次数据
    model = None  # 这里应该是实际的 FLOAT 模型
    new_batch = {
        'reference_motion': torch.randn(1, 512),  # 参考运动特征
        'reference_feat': torch.randn(1, 512),    # 参考特征
        'motion_latent_cur': torch.randn(50, 512) # 当前运动潜在表示
    }
    
    # 3. 使用 ensure 函数（需要实际的模型）
    # data_out = ensure(model, new_batch, opt, step=1000)
    
    # 4. 或者直接使用保存函数（如果有现成的 data_out）
    # 模拟 data_out 张量 (batch_size=1, channels=3, height=512, width=512, frames=50)
    data_out = torch.randn(1, 3, 50, 512, 512)  # 模拟视频张量
    
    # 保存为视频
    output_path = save_data_out_as_video(data_out, opt, step=1000)
    print(f"视频已保存到: {output_path}")

def training_integration_example():
    """训练集成示例"""
    
    # 在训练循环中的使用方式：
    """
    def train_step_with_ensure(model, rectified_flow, batch_data, optimizer, device, opt, step):
        # 执行训练步骤
        loss_dict = train_step(model, rectified_flow, batch_data, optimizer, device)
        
        # 每隔一定步数保存 data_out 视频
        if step % 1000 == 0:  # 每1000步保存一次
            try:
                # 使用 ensure 函数保存视频
                data_out = ensure(model, batch_data, opt, step)
                print(f"步骤 {step}: data_out 视频已保存")
            except Exception as e:
                print(f"步骤 {step}: 保存 data_out 视频时出错: {e}")
        
        return loss_dict
    """

if __name__ == "__main__":
    print("ensure 函数使用示例")
    print("=" * 50)
    
    # 运行示例
    example_usage()
    
    print("\n训练集成示例:")
    print("在训练循环中，可以这样使用:")
    print("""
    # 每隔一定步数调用
    if step % 1000 == 0:
        data_out = ensure(model, new_batch, opt, step)
    """)
    
    print("\n注意事项:")
    print("1. ensure 函数需要模型支持 decode_latent_into_image 方法")
    print("2. 批次数据需要包含 reference_motion, reference_feat, motion_latent_cur")
    print("3. 视频会保存到 ../results/ 目录")
    print("4. 文件名包含时间戳和步数信息")
