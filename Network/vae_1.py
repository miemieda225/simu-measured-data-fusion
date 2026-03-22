import torch
import torch.nn as nn

Z_DIM = 16  # 潜在分布的维度
COMPLEX_CHANNELS = 1  # 输出通道数

class VAE_41(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 编码器 (Encoder) ---
        self.encoder_conv = nn.Sequential(
            # 1. Input: 1 x 513 x 125
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),  # 使用 LeakyReLU
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 20x20
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 10x10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        
        
        self.flat_size = 128*5*5
        self.fc_mu = nn.Linear(self.flat_size, Z_DIM)
        self.fc_logvar = nn.Linear(self.flat_size, Z_DIM)

        # # --- 解码器 (Decoder) ---
        self.fc_decode = nn.Linear(Z_DIM, self.flat_size)
        
        self.decoder_conv = nn.Sequential(
            # 1. Input: 128 x 16 x 20
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            # 2. Output: 64 x 32 x 8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),# 20x20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1), # 39*39
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride =1, padding = 1),
            nn.Sigmoid()  
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
        h = h.view(h.size(0), 128, 5, 5) 
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