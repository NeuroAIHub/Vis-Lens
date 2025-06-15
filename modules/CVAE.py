import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Convolutional Variational Autoencoder architecture
import torch
import torch.nn as nn

class CVAE(nn.Module):

    def __init__(self, latent_dim=1024):
        super().__init__()

        self.latent_dim = latent_dim
        self.shape = 32  # 128 * 128

        # Encode
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*(self.shape-1)**2,2*self.latent_dim)
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(torch.tensor([0.0]))

        # Decode
        self.fc2 = nn.Linear(self.latent_dim, (self.shape ** 2) * 32)
        self.conv3 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        mean, logvar = torch.split(x, self.latent_dim, dim=1)
        return mean, logvar

    def decode(self, eps):
        x = self.relu(self.fc2(eps))
        x = x.view(-1, 32, self.shape, self.shape)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar