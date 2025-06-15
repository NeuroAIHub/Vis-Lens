# train_vae_activation_regularized

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import gc
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
import numpy as np

import sys
sys.path.insert(0,'../models')
from Bottleneck_CVAE import Bottleneck_CVAE
#from CVAE import CVAE
# from ResVAE import ResVAE
from losses import loss_function
from ConvNet import ConvNet
from test_activation import test_input_for_unit_activation
from customize_dataset import CustomCIFAR10

# Check for GPU availability and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load ConvNet
net = ConvNet().to(device)
model_name = '../checkpoint/cifar_10_two_layers_color_1201.pt'
net.load_state_dict(torch.load(model_name))

num_feature_maps = 16
num_epochs = 100
val_epoch_freq = 10
layer_name = 'conv2'
constrained_pos = 'pre'
bottleneck_dim = 96
latent_dim = 96
sparsity_weight = 1
KLD_form = 'default'
#KLD_form = 'two_mode'

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_val_split_ratio = 0.8
trainset = CustomCIFAR10(root='./data', 
                         train=True, 
                         download=True, 
                         transform=transform,
                         preprocessing_network = net.to(device),
                         layer_name = 'conv2',
                         unit_idx = [(i,5,5) for i in range(num_feature_maps)],
                         device = device)

# Split the trainset into trainset and validation set
train_size = int(train_val_split_ratio * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

testset = CustomCIFAR10(root='./data', 
                         train=False, 
                         download=True, 
                         transform=transform,
                         preprocessing_network = net.to(device),
                         layer_name = 'conv2',
                         unit_idx = [(i,5,5) for i in range(num_feature_maps)],
                         device = device)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

z_dim = 96
#vae = CVAE(latent_dim=z_dim).to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

def train(num_epochs = 100, layer_name = None, activation_loss = None, KLD_form = 'default', sparsity_weight = 0, val_epoch_freq = 10):
    vae.train()
    losses_dict = {}
    train_losses = []
    train_recon_losses = []
    train_KLDs = []
    train_L1_sparsity_losses = []
    train_epochs = []
    
    val_losses = []
    val_recon_losses = []
    val_KLDs = []
    val_L1_sparsity_losses = []
    val_epochs = []
    
    for epoch in range(num_epochs):
        train_loss = 0
        train_recon_loss = 0
        train_KLD = 0
        train_L1_sparsity_loss = 0
        
        #for batch_idx, (data, activation_loss) in enumerate(trainloader):
        for batch_idx, (data, _, activation_loss) in enumerate(trainloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            #print(activation_loss.shape)
            #print(torch.max(activation_loss))
            #print(torch.min(activation_loss))
            loss_recon_KLD, recon, KLD = loss_function(recon_batch, 
                                             data, 
                                             mu, 
                                             logvar,
                                             device = device,
                                             activation_loss = activation_loss.to(device),
                                             KLD_form = KLD_form)
            total_l1_norm = sum(torch.norm(param, p=1) for param in vae.fc_mu.parameters())
            total_numel = sum(param.numel() for param in vae.fc_mu.parameters())
            loss_sparsity = total_l1_norm
            loss = loss_recon_KLD + sparsity_weight * loss_sparsity
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm = 1)
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_KLD += KLD.item()
            train_L1_sparsity_loss +=  loss_sparsity.item()
            optimizer.step()

        scheduler.step()

        print('====> Epoch: {} Average loss: {:.4f}, recon loss: {:.4f}, KLD: {:.4f}, sparsity L1: {:.4f}'.format(
            epoch, 
            train_loss / len(trainloader.dataset),
            train_recon_loss / len(trainloader.dataset),
            train_KLD / len(trainloader.dataset),
            train_L1_sparsity_loss / len(trainloader.dataset)))
        
        train_losses.append(train_loss / len(trainloader.dataset))
        train_recon_losses.append(train_recon_loss / len(trainloader.dataset))
        train_KLDs.append(train_KLD / len(trainloader.dataset))
        train_L1_sparsity_losses.append(train_L1_sparsity_loss / len(trainloader.dataset))
        train_epochs.append(epoch)
    
        if int(epoch % val_epoch_freq) == 0:
            val_loss = 0
            val_recon_loss = 0
            val_KLD = 0
            val_L1_sparsity_loss = 0
            for batch_idx, (data, _, activation_loss) in enumerate(trainloader):
                data = data.to(device)
                recon_batch, mu, logvar = vae(data)
                loss_recon_KLD, recon, KLD = loss_function(recon_batch, 
                                             data, 
                                             mu, 
                                             logvar,
                                             device = device,
                                             activation_loss = activation_loss.to(device),
                                             KLD_form = KLD_form)
                total_l1_norm = sum(torch.norm(param, p=1) for param in vae.fc_mu.parameters())
                total_numel = sum(param.numel() for param in vae.fc_mu.parameters())
                loss_sparsity = total_l1_norm
                loss = loss_recon_KLD + sparsity_weight * loss_sparsity

                val_loss += loss.item()
                val_recon_loss += recon.item()
                val_KLD += KLD.item()
                val_L1_sparsity_loss += loss_sparsity.item()
                
            
            print('====> Val epoch: {} Average val loss: {:.4f}, recon val loss: {:.4f}, val KLD: {:.4f}, sparsity L1: {:.4f}'.format(
                epoch, 
                val_loss / len(valloader.dataset),
                val_recon_loss / len(valloader.dataset),
                val_KLD / len(valloader.dataset),
                val_L1_sparsity_loss / len(valloader.dataset)))

            val_losses.append(val_loss / len(valloader.dataset))
            val_recon_losses.append(val_recon_loss / len(valloader.dataset))
            val_KLDs.append(val_KLD / len(valloader.dataset))
            val_L1_sparsity_losses.append(val_L1_sparsity_loss / len(valloader.dataset))
            val_epochs.append(epoch)
                
    losses_dict['loss'] = train_losses
    losses_dict['recon_loss'] = train_recon_losses
    losses_dict['KLD'] = train_KLDs
    losses_dict['sparsity_L1'] = train_L1_sparsity_losses
    losses_dict['train_epochs'] = train_epochs
    
    losses_dict['val_loss'] = val_losses
    losses_dict['val_recon_loss'] = val_recon_losses
    losses_dict['val_KLD'] = val_KLDs
    losses_dict['val_epochs'] = val_epochs
    
    return losses_dict


losses_dict = train(num_epochs = num_epochs, 
                    layer_name = layer_name,
                    KLD_form = KLD_form,
                    val_epoch_freq = val_epoch_freq,
                    sparsity_weight = sparsity_weight)


vae_name = f"../checkpoint/cvae_base_bottleneck_multi_constrain.pt"
torch.save(vae.state_dict(), vae_name)