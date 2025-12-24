import torch
import torch.nn as nn

Z_DIM = 32  # 潜在分布的维度
COMPLEX_CHANNELS = 1  # 输出通道数

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # --- 编码器 (Encoder) ---
        self.encoder_conv = nn.Sequential(
            # 1. Input: 1 x 513 x 125
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(4, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),  # 使用 LeakyReLU
            
            # 2. Output: 32 x 257 x 32
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride = (4, 2), padding = (1,0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 3. Output: 64 x 128 x 8
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
            #Final Feature Map Size: 128 x 16 x 20 (H x W)
        )
        
        self.flat_size = 128*8*8
        self.fc_mu = nn.Linear(self.flat_size, Z_DIM)
        self.fc_logvar = nn.Linear(self.flat_size, Z_DIM)

        # # --- 解码器 (Decoder) ---
        self.fc_decode = nn.Linear(Z_DIM, self.flat_size)
        
        self.decoder_conv = nn.Sequential(
            # 1. Input: 128 x 16 x 20
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2)),
            #nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 2), output_padding=(0, 0)), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 2. Output: 64 x 32 x 8
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(4, 2), padding=(0, 0), output_padding=(0, 0)), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # # 3. Final Output: 1 x 513 x 125 (关键：使用 output_padding 匹配维度)
            nn.ConvTranspose2d(32, 1, kernel_size=(5, 5), stride=(4, 2), padding=(1, 1), output_padding=(2, 0)),
            
            # 关键：不使用任何激活函数 (线性激活) 以保留数据尺度
        )
        #self._get_conv_output_size()

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_decode(z)
        # 重新整形到反卷积层输入维度
        h = h.view(h.size(0), 128, 8, 8) 
        return self.decoder_conv(h)

    def _get_conv_output_size(self):
        """
        通过一次前向传递计算卷积层输出的尺寸，并更新self.flat_size
        """
        dummy_input = torch.zeros(1, 1, 513, 125)  # 模拟一个输入
        #output = self.encoder_conv(dummy_input)
        #print(output.shape)
        output = self.forward(dummy_input)
        print(output[0].shape)

def load_model(model_path, device):
    model = VAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置模型为评估模式
    return model