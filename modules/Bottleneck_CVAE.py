import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Convolutional Variational Autoencoder architecture
class Bottleneck_CVAE(nn.Module):
    def __init__(self, latent_dim=200, bottleneck_dim=100):
        super(Bottleneck_CVAE, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        # Use stride = 2 to reduce dimensions
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        self.fc_mu = nn.Linear(64 * 8 * 8, bottleneck_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, bottleneck_dim)
        
        self.fc_bn_mu = nn.Linear(bottleneck_dim, latent_dim)
        self.fc_bn_logvar = nn.Linear(bottleneck_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 64 * 8 * 8)
        # self.deconv1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        # self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)

    def encode(self, x):
        # Get a compressed feature map
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        
        # (64 * 8 * 8) -> bottleneck_dim -> latent dim, for both mu and logvar
        pre_mu = self.fc_mu(x)
        pre_logvar = self.fc_logvar(x)
        
        post_mu = self.fc_bn_mu(pre_mu)
        post_logvar = self.fc_bn_logvar(pre_logvar)
        
        return pre_mu, pre_logvar, post_mu, post_logvar

    def reparameterize(self, mu, logvar):
        # Use the reparameterization trick to sample from N(mu, var) from N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_decode(z))
        x = x.view(x.size(0), 64, 8, 8)  # Reshape to feature map
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Ensure output is in (0, 1)
        # x = F.relu(self.deconv4(x))
        # x = F.relu(self.deconv5(x))
        
        return x

    def forward(self, x):
        pre_mu, pre_logvar, post_mu, post_logvar = self.encode(x)
        z = self.reparameterize(post_mu, post_logvar)
        return self.decode(z), pre_mu, pre_logvar, post_mu, post_logvar