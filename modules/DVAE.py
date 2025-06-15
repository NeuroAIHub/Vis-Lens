import torch
import torch.nn as nn
import torch.nn.functional as F
"""
class ResBlock(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, up_rate=None, residual=False):
        super().__init__()
        self.down_rate = down_rate
        self.up_rate = up_rate
        self.residual = residual
        self.conv1 = nn.Conv2d(in_width, middle_width, 1)
        self.conv2 = nn.Conv2d(middle_width, middle_width, 3, padding=1)
        self.conv3 = nn.Conv2d(middle_width, middle_width, 3, padding=1)
        self.conv4 = nn.Conv2d(middle_width, out_width, 1)

    def forward(self, x):
        x_res = self.conv1(F.gelu(x))
        x_res = self.conv2(F.gelu(x_res))
        x_res = self.conv3(F.gelu(x_res))
        x_res = self.conv4(F.gelu(x_res))

        if self.residual and x.shape == x_res.shape:
            x_out = x + x_res
        else:
            x_out = x_res

        if self.down_rate:
            x_out = F.avg_pool2d(x_out, kernel_size=self.down_rate)

        if self.up_rate:
            x_out = F.interpolate(x_out, scale_factor=self.up_rate, mode='nearest')

        return x_out

# Encoder: Reduce spatial size, increase channels
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 64, 3, padding=1)  # Input is (3, 224, 224)
        self.block_mod = nn.Sequential(
            ResBlock(64, 64, 128, down_rate=2, residual=True),   # 224 -> 112
            ResBlock(128, 64, 128, down_rate=2, residual=True),  # 112 -> 56
            ResBlock(128, 128, 256, down_rate=2, residual=True), # 56 -> 28
            ResBlock(256, 128, 256, down_rate=2, residual=True), # 28 -> 14
            ResBlock(256, 128, 512, down_rate=2, residual=True), # 14 -> 7
        )
        self.mu = nn.Conv2d(512, latent_dim, 1)
        self.logvar = nn.Conv2d(512, latent_dim, 1)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block_mod(x)
        return self.mu(x), self.logvar(x)

# Decoder: Increase spatial size, decrease channels
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.block_mod = nn.Sequential(
            ResBlock(latent_dim, 512, 512, up_rate=2, residual=True),   # 7 -> 14
            ResBlock(512, 256, 256, up_rate=2, residual=True),          # 14 -> 28
            ResBlock(256, 128, 256, up_rate=2, residual=True),          # 28 -> 56
            ResBlock(256, 128, 128, up_rate=2, residual=True),          # 56 -> 112
            ResBlock(128, 64, 64, up_rate=2, residual=True),            # 112 -> 224
        )
        self.final_conv = nn.Conv2d(64, 3, 3, padding=1)  # Output back to (3, 224, 224)

    def forward(self, z):
        z = self.block_mod(z)
        z = self.final_conv(z)
        return torch.sigmoid(z)

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
"""
