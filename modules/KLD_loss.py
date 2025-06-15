import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_function(recon_x, x, mu, logvar, device, activation_loss = None, recon_weight = 1.0, KLD_weight = 0.1, KLD_form = 'default', mixed_weight = 1, lamb = 1):
    recon = F.mse_loss(recon_x, x, reduction='sum')
    if KLD_form == 'default':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif KLD_form == 'one_mode':
        activation_loss = activation_loss.unsqueeze(1).repeat(1, mu.shape[1])
        KLD = KLD_activation_reguarlized_one_mode(mu, logvar, activation_loss, device, prior_mu = 0, lamb = 1.0)
    elif KLD_form == 'two_mode':
        KLD = KLD_activation_reguarlized_two_modes(mu, logvar, activation_loss, device, lamb)
    elif KLD_form == 'multi':
        KLD = KLD_activation_reguarlized_two_modes(mu, logvar, activation_loss, device, lamb, 'multi')
    elif KLD_form == 'mixed':
        split = int(mu.shape[1] / 2)
        KLD = mixed_weight * -0.5 * torch.sum(1 + logvar[:, :split] - mu[:, :split].pow(2) - logvar[:, :split].exp()) +  KLD_activation_reguarlized_two_modes(mu[:, split:], logvar[:, split:], activation_loss, device, lamb)
    else:
        raise ValueError('KLD form is not defined')
    return recon_weight * recon + KLD_weight * KLD, recon, KLD

def KLD_activation_reguarlized_one_mode(mu, logvar, activation_loss, device, prior_mu = 0, lamb = 1.0):
    prior_std = torch.abs(lamb/(activation_loss + 1e-3))
    prior_mu = (torch.ones_like(mu) * prior_mu).to(device)
    eps = 1e-6
    KLD = torch.sum(torch.log(prior_std)-logvar/2 + (torch.exp(logvar) + torch.pow(mu - prior_mu, 2))/(2*torch.pow(prior_std, 2) + eps) - 1/2)
    return KLD

def KLD_activation_reguarlized_two_modes(mu, logvar, activation_loss, device, lamb = 1.0, units = 'single'):
    if units == 'single':
        activation_loss = activation_loss.unsqueeze(1).repeat(1, mu.shape[1])
    elif units == 'multi':
        activation_loss = activation_loss.repeat(1, int(mu.shape[1]/activation_loss.shape[1]))
    else:
        raise ValueError('units constrained type is not defined')
    pos_index = torch.where(activation_loss>0)
    neg_index = torch.where(activation_loss<0)
    
    KLD_pos = KLD_activation_reguarlized_one_mode(mu[pos_index], logvar[pos_index], activation_loss[pos_index], device = device, prior_mu = 1, lamb = lamb)
    if len(neg_index[0]) > 0:
        KLD_neg = KLD_activation_reguarlized_one_mode(mu[neg_index], logvar[neg_index], activation_loss[neg_index], device = device, prior_mu = -1, lamb = lamb)
        KLD = KLD_pos + KLD_neg
    else:
        KLD = KLD_pos

    return KLD