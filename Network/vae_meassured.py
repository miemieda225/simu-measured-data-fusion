import torch
import torch.nn as nn

class VAE20(nn.Module):
    def __init__(self, latent_dim=16):
        super(VAE20, self).__init__()

        # ===== Encoder =====
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),   # (1,20,20) -> (8,20,20)
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8, 16, kernel_size=4, stride=2,padding=1),  # -> (16,10,10)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1), # 5x5
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)# -> (16,5,5)
        )
        self.flat_size = 64*5*5
        self.fc_mu = nn.Linear(self.flat_size, 16)
        self.fc_logvar = nn.Linear(self.flat_size, 16)

        # ===== Decoder =====
        self.decoder_fc = nn.Linear(latent_dim, 64*5*5)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,padding=1),  # -> (8,10,10)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16,1,kernel_size=3, stride=1, padding=1)
            # -> (1,20,20)
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encode(self, x):
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)  # Flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder_fc(z)
        # 重新整形到反卷积层输入维度
        h = h.view(h.size(0), 64, 5, 5) 
        return self.decoder_conv(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def _get_conv_output_size(self):
        """
        通过一次前向传递计算卷积层输出的尺寸，并更新self.flat_size
        """
        dummy_input = torch.zeros(1, 1, 513, 125)  # 模拟一个输入
        #output = self.encoder_conv(dummy_input)
        #print(output.shape)
        output = self.forward(dummy_input)
        print(output[0].shape)

