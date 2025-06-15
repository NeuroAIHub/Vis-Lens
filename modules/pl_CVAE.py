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
        KLD = KLD_activation_regularized_one_mode(mu, logvar, activation_loss, prior_mu=0, lamb=1)
    elif KLD_form == 'two_mode':
        KLD = KLD_activation_regularized_two_modes(mu, logvar, activation_loss, lamb)
    elif KLD_form == 'multi':
        KLD = KLD_activation_regularized_two_modes(mu, logvar, activation_loss, lamb, 'multi')
    else:
        raise ValueError('KLD form is not defined')
    return recon_weight * recon + KLD_weight * KLD, recon, KLD

def KLD_activation_regularized_one_mode(mu, logvar, activation_loss, prior_mu=0, lamb=1):
    prior_std = torch.abs(lamb / (activation_loss + 1e-6))
    prior_mu = torch.ones_like(mu) * prior_mu
    eps = 1e-6
    KLD = torch.sum(
        torch.log(prior_std) - logvar / 2 +
        (torch.exp(logvar) + (mu - prior_mu).pow(2)) / (2 * prior_std.pow(2) + eps) - 0.5
    )
    return KLD

def KLD_activation_regularized_two_modes(mu, logvar, activation_loss, lamb=1, units='single'):
    if units == 'single':
        activation_loss = activation_loss.unsqueeze(1).repeat(1, mu.shape[1])
    elif units == 'multi':
        activation_loss = activation_loss.repeat(1, int(mu.shape[1] / activation_loss.shape[1]))
    else:
        raise ValueError('Units constrained type is not defined')
    pos_index = torch.where(activation_loss > 0)
    neg_index = torch.where(activation_loss < 0)
    KLD = 0
    if len(pos_index[0]) > 0:
        KLD_pos = KLD_activation_regularized_one_mode(
            mu[pos_index], logvar[pos_index], activation_loss[pos_index], prior_mu=1
        )
        KLD += KLD_pos
    if len(neg_index[0]) > 0:
        KLD_neg = KLD_activation_regularized_one_mode(
            mu[neg_index], logvar[neg_index], activation_loss[neg_index], prior_mu=-1
        )
        KLD += KLD_neg
    return KLD

class VAE(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=1e-4, alpha=1.0):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr
        self.alpha = alpha

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)   # [B, 16, 32, 32]
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # [B, 32, 16, 16]
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # [B, 64, 8, 8]
        # Optional additional layers
        # self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        # self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Adjust the dimensions based on the input image size
        self.img_size = 32  # Assuming input images are 32x32 (CIFAR-10)
        fc_input_dim = 64 * (self.img_size // 4) * (self.img_size // 4)
        self.fc_mu = nn.Linear(fc_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(fc_input_dim, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, fc_input_dim)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1)

    def encode(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        # Optional additional layers
        # x = F.tanh(self.conv4(x))
        # x = F.tanh(self.conv5(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.tanh(self.fc_decode(z))
        x = x.view(z.size(0), 64, self.img_size // 4, self.img_size // 4)
        x = F.tanh(self.deconv1(x))
        x = F.tanh(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Output between [0, 1]
        return x

    def forward(self, z):
        return self.decode(z)

    def forward_recons(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decode(z)
        return decoder_out

    def training_step(self, batch, batch_idx):
        x, activation_loss = batch
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


if __name__ == "__main__":

    batch_size = 32
    x = torch.randn(batch_size, 3, 32, 32)
    # Dummy activation_loss (e.g., random values)
    activation_loss = torch.randn(batch_size, 100)  # Adjust the size as needed

    # Instantiate the VAE model
    vae = VAE(latent_dim=100, lr=1e-4, alpha=1.0)

    # Simulate a training step
    loss = vae.training_step((x, activation_loss), batch_idx=0)
    print(f"Training loss: {loss.item()}")