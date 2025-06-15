import math
import torch
import torch.optim as optim
from torch.nn import Parameter
from utils import to_np

# Define the parameters of the Gaussian distribution
# image_mu = mean[0]
# image_std = std[0]

def KLD_loss(mu, logvar, device, prior_mu):
    prior_std = torch.abs(1/ (torch.ones(1, mu.size()[1]) * 10)).to(device)
    prior_mu = (torch.ones_like(mu) * prior_mu).to(device)
    eps = 1e-6
    KLD = torch.sum(torch.log(prior_std)-logvar/2 + (torch.exp(logvar) + torch.pow(mu - prior_mu, 2))/(2*torch.pow(prior_std, 2) + eps) - 1/2)

    return KLD





def optimize_latent_for_unit_activation(net,
                                        vae,
                                        layer_name, 
                                        unit_idx,
                                        image_mu = None,
                                        image_std = None,
                                        per_feature_num_latent = 6,
                                        num_features = 16,
                                        num_iterations=5000, 
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
        
    if init == 'normal':
        input_latent = torch.randn((1, 100), requires_grad=True, device=device) # Change to single channel
    elif init == 'uniform':
        input_latent = (torch.rand((1, 100), requires_grad=True, device=device) - 0.5) / 0.5 # Change to single channel
    elif init == 'data':
        pre_mu, logvar, _, _ = vae.encode(init_data)

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
            target_activation[0] = output[0, unit_idx[0], unit_idx[1], unit_idx[2]]
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
        
        KLD = KLD_loss(mu = input_latent[:,[i * num_features + unit_idx[0] for i in range(per_feature_num_latent)]],
                       logvar = logvar[:,[i * num_features + unit_idx[0] for i in range(per_feature_num_latent)]].detach(),
                       device = device,
                       prior_mu = -1)
        

        # When default:
        """
        logvar = logvar.detach()
        mu = input_latent
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        """        

        
        loss = target_activation[0].mean() + 1.0 * KLD
        if i == 0:
            print("Init Activation score: ", target_activation[0].mean())
        
        if (i + 1) % 200 == 0:
            print("Activation score: ", target_activation[0].mean())
            print("KLD loss: ", 1.0 * KLD)
        
        if use_regularization:
            l2_loss = l2_reg * torch.sum(input_image**2)
            loss += l2_loss
            
            tv_loss = tv_reg * (torch.sum(torch.abs(input_image[:, :, :-1, :] - input_image[:, :, 1:, :])) +
                            torch.sum(torch.abs(input_image[:, :, :, :-1] - input_image[:, :, :, 1:])))
            loss += tv_loss
        
        if use_likelihood:
            # Compute Gaussian log likelihood
            log_likelihood = -0.5 * math.log(2 * math.pi * image_std**2) - (input_image - image_mu).pow(2).mean() / (2 * image_std**2)
            loss += log_reg * (-log_likelihood)

        loss.backward()
        
        if use_energy:
            with torch.no_grad():
                # Update latent by descending along the gradient and adding Gaussian noise
                input_latent -= energy_scale * step_size * input_latent.grad
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
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
    
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


def optimize_input_for_unit_activation(net,
                                       layer_name, 
                                       unit_idx,
                                       num_iterations=5000, 
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
    
    device = next(net.parameters()).device
        
    if init == 'normal':
        #input_image = torch.randn((1, 1, 64, 64), requires_grad=True, device=device) # Change to single channel
        input_image = torch.randn((1, 1, 32, 32), requires_grad=True, device=device) # Change to single channel
    elif init == 'uniform':
        input_image = (torch.rand((1, 1, 32, 32), requires_grad=True, device=device) - 0.5) / 0.5 # Change to single channel
    elif init == 'data':
        input_image = Parameter(torch.empty(init_data.size(), device=device), requires_grad=True)
        input_image.data.copy_(init_data)
    else:
        raise ValueError("init type is not supported")

    optimizer = optim.Adam([input_image], lr=learning_rate)

    optimizer.zero_grad()
    
    target_activation = [None]
    def hook(module, input, output):
        nonlocal target_activation
        if isinstance(unit_idx, tuple):
            target_activation[0] = output[0, unit_idx[0], unit_idx[1], unit_idx[2]]
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
    
    step_size = learning_rate
    for i in range(num_iterations):
        
        if i % decay_steps == 0 and i > 0:
            step_size *= decay_factor

        _ = net(input_image)

        loss = -target_activation[0].mean()

        if use_regularization:
            l2_loss = l2_reg * torch.sum(input_image**2)
            loss += l2_loss
            
            tv_loss = tv_reg * (torch.sum(torch.abs(input_image[:, :, :-1, :] - input_image[:, :, 1:, :])) +
                            torch.sum(torch.abs(input_image[:, :, :, :-1] - input_image[:, :, :, 1:])))
            loss += tv_loss
        
        if use_likelihood:
            # Compute Gaussian log likelihood
            log_likelihood = -0.5 * math.log(2 * math.pi * sigma**2) - (input_image - mu).pow(2).mean() / (2 * sigma**2)
            loss += log_reg * (-log_likelihood)

        loss.backward()
        
        if use_energy:
            with torch.no_grad():
                # Update image by descending along the gradient and adding Gaussian noise
                input_image -= energy_scale * step_size * input_image.grad
                input_image += step_size * 2 * torch.randn_like(input_image)
            # optimizer.step()

            # Zero the gradients before the next step
            input_image.grad.zero_()
        else:
            optimizer.step()
            optimizer.zero_grad()

        if use_projection:
            # projection step
            input_image.data.clamp_(min=-1, max=1)
    
    print('activation_loss:', target_activation[0].mean())
    hook_handle.remove()

    input_image = input_image.detach()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    return input_image