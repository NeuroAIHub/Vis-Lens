import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import gc
import torch.nn.functional as F
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import math

from torchmetrics.image.fid import FrechetInceptionDistance

import sys 
sys.path.insert(0,'../modules')
from Bottleneck_CVAE import Bottleneck_CVAE
from KLD_loss import loss_function
from data_preprocess import calculate_mean_std
from customize_dataset import CustomCIFAR10

# 1. Define the ConvNet Model
# 3 Layers
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load and preprocess the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Check for GPU availability and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize the ConvNet, loss function, and optimizer
net = ConvNet().to(device)


model_name = '../checkpoints/cifar_10_three_layers_color.pt'
net.load_state_dict(torch.load(model_name))
net.to(device)

train_val_split_ratio = 0.8
num_features = 32
trainset = CustomCIFAR10(root='../datasets', 
                         train=True, 
                         download=True, 
                         transform=transform,
                         preprocessing_network = net.to(device),
                         layer_name = 'conv1',
                         unit_idx = [(i,5,5) for i in range(num_features)],
                         device = device)

# Split the trainset into trainset and validation set
train_size = int(train_val_split_ratio * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)

testset = CustomCIFAR10(root='../datasets', 
                         train=False, 
                         download=True, 
                         transform=transform,
                         preprocessing_network = net.to(device),
                         layer_name = 'conv1',
                         unit_idx = [(i,5,5) for i in range(num_features)],
                         #unit_idx = [(3,5,5), (3,1,1), 3],
                         device = device)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


bottleneck_dim = 192
latent_dim = 192
vae = Bottleneck_CVAE(bottleneck_dim = bottleneck_dim, latent_dim = latent_dim).to(device)


#vae_name = '../checkpoint/cvae_base_bottleneck_default.pt'
vae_name = '../checkpoints/exp_cvae_base_bottleneck_multi_constrain_three_layers_conv_1_32_features_lamb_0_1.pt'
vae.load_state_dict(torch.load(vae_name))
vae.to(device)

# Define Optimization Process based on Gradient Descent

def to_np(tensor_input):
    return tensor_input.detach().cpu().numpy()

torch.manual_seed(42)
np.random.seed(42)

from torch.nn import Parameter
import torch.optim as optim
import math

def KLD_loss(mu, logvar, device, prior_mu):
    prior_var = (torch.ones(1, mu.size()[1]) * 0.01).to(device)  # prior_std=0.1, prior_var=0.01
    prior_logvar = torch.log(prior_var)
    prior_mu = (torch.ones_like(mu) * prior_mu).to(device)

    # Compute the KL divergence
    KLD = 0.5 * torch.mean(
        prior_logvar - logvar
        + (torch.exp(logvar) + (mu - prior_mu) ** 2) / prior_var
        - 1
    )
    return KLD

def optimize_activation_with_latent(net,
                                        vae,
                                        layer_name, 
                                        unit_idx,
                                        per_feature_num_latent = 6,
                                        num_features = 16,
                                        num_iterations=5000, 
                                        KLD_weight = 1.0,
                                        target = "max",
                                        mode = "reg",
                                        learning_rate=0.1, 
                                        l2_reg=1e-3, 
                                        log_reg = 1e-3,
                                        tv_reg=1e-3, 
                                        noise_coef = 0.1, 
                                        energy_scale = 1.0, 
                                        decay_steps = 100, 
                                        decay_factor = 0.9,
                                        init = 'normal', # ['uniform', 'data']
                                        init_data = None,
                                        use_regularization = False,
                                        use_projection = False,
                                        use_likelihood = False,
                                        use_energy = False):
    net.eval()
    vae.eval()
    
    device = next(net.parameters()).device
    #print("Energy Scale is: ", energy_scale)
    if init == 'normal':
        input_latent = torch.randn((1, 100), requires_grad=True, device=device) # Change to single channel
    elif init == 'uniform':
        input_latent = (torch.rand((1, 100), requires_grad=True, device=device) - 0.5) / 0.5 # Change to single channel
    elif init == 'data':
        pre_mu, logvar, _, _ = vae.encode(init_data)

        
        if target == "max":
            for ith_latent in range(per_feature_num_latent):
                pre_mu[:,ith_latent*num_features+unit_idx[0]] = 1
        else:
            for ith_latent in range(per_feature_num_latent):
                pre_mu[:,ith_latent*num_features+unit_idx[0]] = -1      
                
        input_latent = Parameter(torch.empty(pre_mu.size(), device=device), requires_grad=True)
        input_latent.data.copy_(pre_mu)
        pre_mu_init = Parameter(torch.empty(pre_mu.size(), device=device), requires_grad=False)
        pre_mu_init.data.copy_(pre_mu)
        '''
        input_latent = Parameter(torch.empty((pre_mu.size()[0], per_feature_num_latent), device=device), requires_grad=True)
        # print(pre_mu.shape)
        input_latent_learn_params = []
        for ith_latent in range(per_feature_num_latent):
            input_latent_learn_params += list(pre_mu[:,ith_latent*num_features+unit_idx[0]])
        # print(torch.stack(input_latent_learn_params, dim = -1).unsqueeze(0).shape)
        input_latent.data.copy_(torch.stack(input_latent_learn_params, dim = -1).unsqueeze(0))
        '''
    else:
        raise ValueError("init type is not supported")
    
    optimizer = optim.Adam([input_latent], lr=learning_rate)

    optimizer.zero_grad()
    
    target_activation = [None]
    
    def hook(module, input, output):
        nonlocal target_activation
        if isinstance(unit_idx, tuple):
            #target_activation[0] = output[:, unit_idx[0], unit_idx[1], unit_idx[2]]
            target_activation[0] = output[:, unit_idx[0]].mean(-1).mean(-1)
        elif isinstance(unit_idx, list):
            unit_idx_item = unit_idx[0]
            target_activation[0] = output[0, unit_idx_item[0], unit_idx_item[1], unit_idx_item[2]]
            for unit_idx_item in unit_idx[1:]:
                target_activation[0] += output[0, unit_idx_item[0], unit_idx_item[1], unit_idx_item[2]]
        else:
            target_activation[0] = output[0, unit_idx]

    # layers = {'conv1': net.conv1, 'conv2': net.conv2, 'conv3': net.conv3, 'conv4': net.conv4}
    layers = {'conv1': net.conv1, 'conv2': net.conv2}
    layer = layers[layer_name]
    hook_handle = layer.register_forward_hook(hook)
    
    mask = torch.zeros_like(pre_mu)
    for ith_latent in range(per_feature_num_latent):
        mask[:,ith_latent*num_features+unit_idx[0]] = 1

    
    step_size = learning_rate
    for i in range(num_iterations):
        
        if i % decay_steps == 0 and i > 0:
            step_size *= decay_factor
        
        # pre_mu_new_params = []
        # old_index = 0
        # for ith_latent in range(per_feature_num_latent):
        #     pre_mu_new_params.append(pre_mu[:,old_index:ith_latent*num_features+unit_idx[0]])
        #     # print(pre_mu[:,old_index:ith_latent*num_features+unit_idx[0]].shape)
        #     pre_mu_new_params.append(input_latent[:,ith_latent].unsqueeze(-1))
        #     # print(input_latent[:,ith_latent].unsqueeze(-1).shape)
        #     old_index = ith_latent*num_features+unit_idx[0]+1
        # pre_mu_new_params.append(pre_mu[:,old_index:])
        # print(torch.cat(pre_mu_new_params, dim = 1).shape)
        # post_mu = vae.fc_bn_mu(torch.cat(pre_mu_new_params, dim = -1))
        # import pdb; pdb.set_trace()
        
        # post_mu = vae.fc_bn_mu(input_latent)
        
        post_mu = vae.fc_bn_mu(pre_mu_init * (1 - mask) + input_latent * mask)
        
        input_image = vae.decode(post_mu)

        _ = net(input_image)

        # When constrained:
        if mode == "reg":
            if target == "max":
                KLD = KLD_loss(mu = input_latent[:,[i * num_features + unit_idx[0] for i in range(per_feature_num_latent)]],
                       logvar = logvar[:,[i * num_features + unit_idx[0] for i in range(per_feature_num_latent)]].detach(),
                       device = device,
                       prior_mu = 1)
            else:
                KLD = KLD_loss(mu = input_latent[:,[i * num_features + unit_idx[0] for i in range(per_feature_num_latent)]],
                       logvar = logvar[:,[i * num_features + unit_idx[0] for i in range(per_feature_num_latent)]].detach(),
                       device = device,
                       prior_mu = -1)
        
        else:
            logvar = logvar.detach()
            mu = input_latent
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if target == "max":
            loss = -target_activation[0].mean() + KLD_weight * KLD
        else:
            loss = target_activation[0].mean() + KLD_weight * KLD
        if i == 0:
            pass
            #print("Init Activation score: ", target_activation[0].mean())
        
        if (i + 1) % 200 == 0:
            pass
            #print("Activation score: ", target_activation[0].mean())
            #print("KLD loss: ", KLD_weight * KLD)
        
        if use_regularization:
            l2_loss = l2_reg * torch.sum(input_image**2)
            loss += l2_loss
            
            tv_loss = tv_reg * (torch.sum(torch.abs(input_image[:, :, :-1, :] - input_image[:, :, 1:, :])) +
                            torch.sum(torch.abs(input_image[:, :, :, :-1] - input_image[:, :, :, 1:])))
            loss += tv_loss
        

        loss.backward()
        
        if use_energy:
            with torch.no_grad():
                # Update latent by descending along the gradient and adding Gaussian noise
                input_latent -= energy_scale * step_size * input_latent.grad
                #print(input_latent.grad)
                input_latent += step_size * 2 * torch.randn_like(input_latent)
            # optimizer.step()

            # Zero the gradients before the next step
            input_latent.grad.zero_()
        else:
            optimizer.step()
            optimizer.zero_grad()

        if use_projection:
            # projection step
            input_latent.data.clamp_(min=-1, max=1)
    
    scores_all = []
    
    scores_all.append(to_np(target_activation[0].mean()))
    input_image = input_image.detach()
    
    print('activation_score_LD_optimize:', to_np(target_activation[0].mean()))
    
    ######################################
    
    _ = net(init_data)
    
    scores_all.append(to_np(target_activation[0].mean()))
    
    print('activation_score_init:', to_np(target_activation[0].mean()))
    
    ######increase prior
    
    pre_mu_increase = pre_mu.clone()
    for ith_latent in range(per_feature_num_latent):
        pre_mu_increase[:,ith_latent*num_features+unit_idx[0]] = 1
    
    post_mu_increase = vae.fc_bn_mu(pre_mu_increase)
    input_image_increase = vae.decode(post_mu_increase)
    
    _ = net(input_image_increase)
    
    scores_all.append(to_np(target_activation[0].mean()))
    
    print('activation_score_increase_prior:', to_np(target_activation[0].mean()))
    
    input_image_increase = input_image_increase.detach()
    input_image_increase = (input_image_increase - input_image_increase.min()) / (input_image_increase.max() - input_image_increase.min())
    
    ######decrease prior
    
    pre_mu_decrease = pre_mu.clone()
    for ith_latent in range(per_feature_num_latent):
        pre_mu_decrease[:,ith_latent*num_features+unit_idx[0]] = -1
    
    post_mu_decrease = vae.fc_bn_mu(pre_mu_decrease)
    input_image_decrease = vae.decode(post_mu_decrease)
    
    _ = net(input_image_decrease)
    
    scores_all.append(to_np(target_activation[0].mean()))
    
    print('activation_score_decrease_prior:', to_np(target_activation[0].mean()))
    
    input_image_decrease = input_image_decrease.detach()
    input_image_decrease = (input_image_decrease - input_image_decrease.min()) / (input_image_decrease.max() - input_image_decrease.min())
        
    hook_handle.remove()

    return input_image, input_image_increase, input_image_decrease, scores_all


# increase activation
# decrease activation

print("Start optimizing for maximized activation with regularizated VAE")
use_regularization = False
use_projection = False
use_likelihood = False
use_energy = False
num_initializations = 4
optimize_object = 'latent'
#optimize_object = 'input'
init = 'data'
per_feature_num_latent = 6
num_features = 32
#idx_list = [1,530,220,269]
learning_rate = 0.01
num_iterations = 1000
layer_name = 'conv1'
unit_idx = (23,5,5)
KLD_weight = 0.001
target = "max" #["max", "min"]
mode = "reg" #["reg", "default"]
#energy_scale = 0.01
images = []
max_opt_images = []
decrease_images = []
increase_images = []
for i in range(100):
    max_opt_image, increase_image, decrease_image, scores = optimize_activation_with_latent(
                net = net.to(device),
                vae = vae,
                layer_name = layer_name, 
                unit_idx = unit_idx, 
                per_feature_num_latent = per_feature_num_latent,
                num_features = num_features,
                learning_rate = learning_rate, 
                num_iterations = num_iterations,
                KLD_weight = KLD_weight,
                target = target,
                mode = mode,
                use_projection = use_projection,
                use_likelihood = use_likelihood,
                use_energy = use_energy,
                use_regularization = use_regularization,
                init = init,
                init_data = testset[i][0].unsqueeze(0).to(device))
    images.append(testset[i][0].unsqueeze(0).to(device))
    max_opt_images.append(max_opt_image)
    decrease_images.append(decrease_image)
    increase_images.append(increase_image)
    print("Finish opt for image: " + str(i))




print("Start optimizing for minimized activation with regularizated VAE")
use_regularization = False
use_projection = False
use_likelihood = False
use_energy = False
num_initializations = 4
optimize_object = 'latent'
#optimize_object = 'input'
init = 'data'
per_feature_num_latent = 6
num_features = 32
#idx_list = [1,530,220,269]
learning_rate = 0.01
num_iterations = 1000
layer_name = 'conv1'
unit_idx = (23,5,5)
KLD_weight = 0.001
target = "min" #["max", "min"]
mode = "reg" #["reg", "default"]
#energy_scale = 0.01
min_opt_images = []
for i in range(100):
    min_opt_image, increase_image, decrease_image, scores = optimize_activation_with_latent(
                net = net.to(device),
                vae = vae,
                layer_name = layer_name, 
                unit_idx = unit_idx, 
                per_feature_num_latent = per_feature_num_latent,
                num_features = num_features,
                learning_rate = learning_rate, 
                num_iterations = num_iterations,
                KLD_weight = KLD_weight,
                target = target,
                mode = mode,
                use_projection = use_projection,
                use_likelihood = use_likelihood,
                use_energy = use_energy,
                use_regularization = use_regularization,
                init = init,
                init_data = testset[i][0].unsqueeze(0).to(device))
    min_opt_images.append(min_opt_image)
    print("Finish opt for image: " + str(i))



vae_name = '../checkpoints/cvae_base_bottleneck_default.pt'
vae.load_state_dict(torch.load(vae_name))
vae.to(device)



print("Start optimizing for maximized activation with default VAE")
use_regularization = False
use_projection = False
use_likelihood = False
use_energy = False
num_initializations = 4
optimize_object = 'latent'
#optimize_object = 'input'
init = 'data'
per_feature_num_latent = 6
num_features = 32
#idx_list = [1,530,220,269]
learning_rate = 0.01
num_iterations = 1000
layer_name = 'conv1'
unit_idx = (23,5,5)
KLD_weight = 0.001
target = "max" #["max", "min"]
mode = "default" #["reg", "default"]
#energy_scale = 0.01
default_max_opt_images = []
for i in range(100):
    max_opt_image, increase_image, decrease_image, scores = optimize_activation_with_latent(
                net = net.to(device),
                vae = vae,
                layer_name = layer_name, 
                unit_idx = unit_idx, 
                per_feature_num_latent = per_feature_num_latent,
                num_features = num_features,
                learning_rate = learning_rate, 
                num_iterations = num_iterations,
                KLD_weight = KLD_weight,
                target = target,
                mode = mode,
                use_projection = use_projection,
                use_likelihood = use_likelihood,
                use_energy = use_energy,
                use_regularization = use_regularization,
                init = init,
                init_data = testset[i][0].unsqueeze(0).to(device))
    default_max_opt_images.append(max_opt_image)
    print("Finish opt for image: " + str(i))




print("Start optimizing for minimized activation with default VAE")
use_regularization = False
use_projection = False
use_likelihood = False
use_energy = False
num_initializations = 4
optimize_object = 'latent'
#optimize_object = 'input'
init = 'data'
per_feature_num_latent = 6
num_features = 32
#idx_list = [1,530,220,269]
learning_rate = 0.01
num_iterations = 1000
layer_name = 'conv1'
unit_idx = (23,5,5)
KLD_weight = 0.001
target = "min" #["max", "min"]
mode = "default" #["reg", "default"]
#energy_scale = 0.01
default_neg_opt_images = []
for i in range(100):
    min_opt_image, increase_image, decrease_image, scores = optimize_activation_with_latent(
                net = net.to(device),
                vae = vae,
                layer_name = layer_name, 
                unit_idx = unit_idx, 
                per_feature_num_latent = per_feature_num_latent,
                num_features = num_features,
                learning_rate = learning_rate, 
                num_iterations = num_iterations,
                KLD_weight = KLD_weight,
                target = target,
                mode = mode,
                use_projection = use_projection,
                use_likelihood = use_likelihood,
                use_energy = use_energy,
                use_regularization = use_regularization,
                init = init,
                init_data = testset[i][0].unsqueeze(0).to(device))
    default_neg_opt_images.append(min_opt_image)
    print("Finish opt for image: " + str(i))

print("Start saving images")
np.save("../checkpoints/original_100_images.npy", np.array([images[i].detach().cpu() for i in range(100)]).squeeze(1))
np.save("../checkpoints/original_100_increase_images.npy", np.array([increase_images[i].detach().cpu() for i in range(100)]).squeeze(1))
np.save("../checkpoints/original_100_decrease_images.npy", np.array([decrease_images[i].detach().cpu() for i in range(100)]).squeeze(1))
np.save("../checkpoints/100_input_max_opt_images.npy", np.array([max_opt_images[i].detach().cpu() for i in range(100)]).squeeze(1))
np.save("../checkpoints/100_input_min_opt_images.npy", np.array([min_opt_images[i].detach().cpu() for i in range(100)]).squeeze(1))
np.save("../checkpoints/100_default_neg_opt_images.npy", np.array([default_neg_opt_images[i].detach().cpu() for i in range(100)]).squeeze(1))
np.save("../checkpoints/100_default_pos_opt_images.npy", np.array([default_max_opt_images[i].detach().cpu() for i in range(100)]).squeeze(1))
print("Finish saving images")