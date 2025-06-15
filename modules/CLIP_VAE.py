import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPtoCLIPVAE(nn.Module):
    def __init__(self, latent_dim=1024, input_dim=1024):
        super(CLIPtoCLIPVAE, self).__init__()

        # --------- Encoder: CLIP → Latent ---------
        """
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            
            nn.GELU(),
            nn.LayerNorm(2048),

            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, latent_dim * 2)  # Output: mu + logvar
        )
        """
        self.encoder = nn.Linear(input_dim, latent_dim * 2)
        self.decoder = nn.Linear(latent_dim, input_dim)
        # --------- Decoder: Latent → CLIP ---------
        """
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),

            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),

            nn.Linear(2048, input_dim)
        )
        """

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :h.shape[1] // 2], h[:, h.shape[1] // 2:]
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
        out = self.decode(z)
        return out, mu, logvar, z