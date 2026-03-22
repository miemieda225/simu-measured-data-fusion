# generate_latent.py
import torch
from .vae import load_model

def gen_vector(model_path, input_data, device):
    """
    生成潜在向量 z。
    
    Parameters:
    - model_path: str, 模型权重文件的路径
    - input_data: torch.Tensor, 输入的数据（如图像）
    - device: torch.device, 运行设备（CPU 或 GPU）
    
    Returns:
    - z: torch.Tensor, 潜在向量
    """
    # 加载模型
    model = load_model(model_path, device)

    # 获取编码器的输出（mu, logvar）
    mu, logvar = model.encode(input_data)
    
    # 使用重参数化技巧得到潜在向量 z
    z = model.reparameterize(mu, logvar)
    
    return z
