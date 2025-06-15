import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, activation_loss=None, recon_weight=1.0, KLD_weight=1.0, KLD_form='default', units_num=1, lamb=1):
    recon = F.mse_loss(recon_x, x, reduction='sum')
    if KLD_form == 'default':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif KLD_form == 'one_mode':
        activation_loss = activation_loss.unsqueeze(1).repeat(1, mu.shape[1])
        KLD = KLD_activation_reguarlized_one_mode(mu, logvar, activation_loss, prior_mu=0, lamb=1)
    elif KLD_form == 'two_mode':
        KLD = KLD_activation_reguarlized_two_modes(mu, logvar, activation_loss, lamb)
    elif KLD_form == 'multi':
        KLD = KLD_activation_reguarlized_two_modes(mu, logvar, activation_loss, lamb, 'multi')
    else:
        raise ValueError('KLD form is not defined')
    return recon_weight * recon + KLD_weight * KLD, recon, KLD

def KLD_activation_reguarlized_one_mode(mu, logvar, activation_loss, prior_mu=0, lamb=1):
    prior_std = torch.abs(lamb / (activation_loss + 1e-6))
    prior_mu = torch.ones_like(mu) * prior_mu
    eps = 1e-6
    KLD = torch.sum(torch.log(prior_std) - logvar / 2 + (torch.exp(logvar) + (mu - prior_mu).pow(2)) / (2 * prior_std.pow(2) + eps) - 0.5)
    return KLD

def KLD_activation_reguarlized_two_modes(mu, logvar, activation_loss, lamb=1, units='single'):
    if units == 'single':
        activation_loss = activation_loss.unsqueeze(1).repeat(1, mu.shape[1])
    elif units == 'multi':
        activation_loss = activation_loss.repeat(1, int(mu.shape[1] / activation_loss.shape[1]))
    else:
        raise ValueError('units constrained type is not defined')
    pos_index = torch.where(activation_loss > 0)
    neg_index = torch.where(activation_loss < 0)
    KLD = 0
    if len(pos_index[0]) > 0:
        KLD_pos = KLD_activation_reguarlized_one_mode(mu[pos_index], logvar[pos_index], activation_loss[pos_index], prior_mu=1)
        KLD += KLD_pos
    if len(neg_index[0]) > 0:
        KLD_neg = KLD_activation_reguarlized_one_mode(mu[neg_index], logvar[neg_index], activation_loss[neg_index], prior_mu=-1)
        KLD += KLD_neg
    return KLD

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=128, lr=1e-4, alpha=1.0):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr
        self.alpha = alpha

        # Encoding layers (from SimpleVAE) adjusted for input size 256x256
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Adjust the dimensions for the linear layers
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, latent_dim)

        # Decoder layers adjusted accordingly
        self.fc_decode = nn.Linear(latent_dim, 512 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 256, 256]
            nn.Sigmoid()  # Output between [0, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 512, 8, 8)
        x = self.decoder(x)
        return x

    def forward(self, z):
        # Used for generating images from sampled latent vectors
        return self.decode(z)

    def forward_recons(self, x):
        # For generating reconstructions during inference
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x, activation_loss = batch  # Assuming batch returns (input, activation_loss)
        # Encoder
        mu, logvar = self.encode(x)
        # Reparameterization Trick
        z = self.reparameterize(mu, logvar)
        # Decoder
        decoder_out = self.decode(z)
        # Compute loss
        vae_loss, recons_loss, kl_loss = loss_function(
            decoder_out, x, mu, logvar,
            activation_loss=activation_loss, KLD_form='multi'
        )

        self.log("Reconstruction Loss", recons_loss, prog_bar=True)
        self.log("KL Divergence", kl_loss, prog_bar=True)
        total_loss = vae_loss
        self.log("Total Loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

