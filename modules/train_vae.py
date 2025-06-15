import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import gc
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
import numpy as np

import sys 
sys.path.insert(0,'../models')
# from CVAE import CVAE
from ResVAE import ResVAE
from losses import loss_function
from ConvNet import ConvNet
from test_activation import test_input_for_unit_activation

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_val_split_ratio = 0.8
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# Split the trainset into trainset and validation set
train_size = int(train_val_split_ratio * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Check for GPU availability and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load ConvNet
net = ConvNet().to(device)
model_name = '../checkpoints/cifar_10_two_layers_0717.pt'
net.load_state_dict(torch.load(model_name))
net.to(device)

# Initialize VAE, loss function, and optimizer
vae = ResVAE(channel_in=1,
             ch=64, 
             blocks=(1, 2, 4, 8), 
             latent_channels=128).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5) 

def train(num_epochs, layer_name = None, unit_idx = None, activation_loss = None, KLD_form = 'default'):
    vae.train()
    losses_dict = {}
    train_losses = []
    train_recon_losses = []
    train_KLDs = []
    for epoch in range(num_epochs):
        train_loss = 0
        train_recon_loss = 0
        train_KLD = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            if KLD_form != 'default':
                activation_loss = test_input_for_unit_activation(net.to(device),
                                                      layer_name,
                                                      unit_idx,
                                                      data)
            loss, recon, KLD = loss_function(recon_batch, 
                                             data, 
                                             mu, 
                                             logvar,
                                             device = device,
                                             activation_loss = activation_loss,
                                             KLD_form = KLD_form)
            loss.backward()
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_KLD += KLD.item()
            optimizer.step()

        scheduler.step()

        print('====> Epoch: {} Average loss: {:.4f}, recon loss: {:.4f}, KLD: {:.4f}'.format(
            epoch, 
            train_loss / len(trainloader.dataset),
            train_recon_loss / len(trainloader.dataset),
            train_KLD / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))
        train_recon_losses.append(train_recon_loss / len(trainloader.dataset))
        train_KLDs.append(train_KLD / len(trainloader.dataset))
    
    losses_dict['loss'] = train_losses
    losses_dict['recon_loss'] = train_recon_losses
    losses_dict['KLD'] = train_KLDs
    
    return losses_dict


num_epochs = 100
layer_name = 'conv2' 
unit_idx = (3,5,5)

losses_dict = train(num_epochs = num_epochs, 
                    layer_name = layer_name,
                    unit_idx = unit_idx,
                    KLD_form = 'default')

vae_name = '../checkpoints/resvae_base_128_KLD_w_10_0903.pt'
torch.save(vae.state_dict(), vae_name)

np.save('losses_dict_resvae_base_128KLD_w_10_0903.npy',losses_dict)