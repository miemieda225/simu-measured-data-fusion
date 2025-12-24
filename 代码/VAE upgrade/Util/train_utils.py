# utils/train_utils.py

import torch
import torch.nn.functional as F

def vae_loss(x_recon, x, mu, logvar, beta=0.01):
    # 计算重建损失
    recon_loss_sum = F.mse_loss(x_recon, x, reduction='sum')

    B = x.size(0)
    num_elements = x.size(1) * x.size(2) * x.size(3) 
    
    # 计算 KL 散度损失
    recon_loss = recon_loss_sum / (B * num_elements)
    kl_loss_sum = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss_sum / B
    
    # 总损失 (重建损失 + KL 损失)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_vae(model, train_loader, val_loader, optimizer, num_epochs, device, beta, save_path):
    # 用于存储训练和验证历史记录
    losses = []
    recon = []
    kl = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = 0
        recon_l_total = 0
        kl_l_total = 0
        model.train()  # 设置模型为训练模式
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            recon_data, mu, logvar = model(data)
            
            # 计算损失
            current_beta = min(epoch / 300, 1.0) * beta
            loss, recon_l, kl_l = vae_loss(recon_data, data, mu, logvar, current_beta)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_l_total += recon_l.item()
            kl_l_total += kl_l.item()

        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon = recon_l_total / len(train_loader.dataset)
        avg_kl = kl_l_total / len(train_loader.dataset)
        losses.append(avg_loss)
        recon.append(avg_recon)
        kl.append(avg_kl)

        # -------------------- 验证阶段 --------------------
        model.eval()  # 设置模型为评估模式
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device)
                recon_data, mu, log_var = model(data)
                loss, _, _ = vae_loss(recon_data, data, mu, log_var, current_beta)
                epoch_val_loss += loss.item() * data.size(0)
                
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if epoch % 100 == 0:
            print(f'====> Epoch: {epoch}/{num_epochs} Average loss: {avg_loss:.4f}')
            
    # 保存模型权重
    torch.save(model.state_dict(), save_path)
    print(f"-> 模型权重已保存到: {save_path}")

    return model, losses, recon, kl, val_losses

def plot_curve(losses, recon, kl, val_losses, log_scale=True):
    """
    绘制损失曲线，可以选择绘制原始值和对数值，或者同时绘制两者
    :param losses: 总损失
    :param recon: 重建损失
    :param kl: KL 散度损失
    :param val_losses: 验证损失
    :param log_scale: 是否绘制对数曲线，默认为 True
    """
    plt.figure(figsize=(10, 6))  # 设置图像大小
    
    if log_scale:
        # 绘制对数损失曲线
        plt.plot(np.log(losses), label="Total Loss (Log scale)")
        plt.plot(np.log(recon), label="Reconstruction Loss (Log scale)")
        plt.plot(np.log(kl), label="KL Loss (Log scale)")
        plt.plot(np.log(val_losses), label="Validation Loss (Log scale)")
        plt.ylabel('Log(Loss)')
    else:
        # 绘制原始损失曲线
        plt.plot(losses, label="Total Loss")
        plt.plot(recon, label="Reconstruction Loss")
        plt.plot(kl, label="KL Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.ylabel('Loss')
    
    # 设置X轴标签、标题和图例
    plt.xlabel('Epoch')
    plt.title('Loss Value')
    plt.legend()
    plt.tight_layout()
    plt.show()
