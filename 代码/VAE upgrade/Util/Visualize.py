import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def Plot(model, dataloader, device, index=5):
    """
    可视化原始数据和重建数据的热图
    :param model: 训练好的模型
    :param dataloader: 数据加载器
    :param device: 设备 (cpu or cuda)
    :param index: 用于从 dataloader 中获取数据的索引
    """
    model.eval()  # 将模型设置为评估模式
    x = next(iter(dataloader))[index]  # 获取数据
    x = x.float().unsqueeze(0)  # 调整形状
    x = x.to(device)  # 将数据转移到指定设备
    x_output = model.forward(x)  # 获取模型输出
    x2d = x_output[0]
    # 重建数据
    x2d = x2d.squeeze().detach().cpu().numpy()  # 从GPU取回并转换为numpy
    x = x.squeeze().detach().cpu().numpy()  # 原始数据
    
    # 设置全局最大最小值，用于统一热图的显示范围
    global_min = min(x.min(), x2d.min())
    global_max = max(x.max(), x2d.max())

    # 绘制热图
    plt.figure(figsize=(8, 8))  # 整体画布
    
    frequencies = np.linspace(0, 2500, 513)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.subplot(1, 2, 1)  # 1行2列的第1个
    sns.heatmap(x, cmap='viridis', cbar=True, vmin=global_min, vmax=global_max)
    plt.title('Original')
    plt.yticks(np.arange(0, 513, step=50), labels=np.round(frequencies[::50], 2))
    plt.xlabel('测点', fontsize=12)
    plt.ylabel('频率 (Hz)', fontsize=12)

    plt.subplot(1, 2, 2)  # 1行2列的第2个
    sns.heatmap(x2d, cmap='viridis', cbar=True, vmin=global_min, vmax=global_max)
    plt.title('Rebuilt')
    plt.yticks(np.arange(0, 513, step=50), labels=np.round(frequencies[::50], 2))
    plt.xlabel('测点', fontsize=12)
    plt.ylabel('频率 (Hz)', fontsize=12)
    
    plt.tight_layout()  # 自动调整布局
    plt.show()
