import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from optimize_activation import optimize_latent_for_unit_activation, optimize_input_for_unit_activation
from test_activation import test_input_for_unit_activation
import math


def to_np(tensor_input):
    return tensor_input.detach().cpu().numpy()

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize_features(net, input_image, layer_name):
    net = net.to("cpu")
    input_image = input_image.to("cpu")

    layer_outputs = []

    def hook(module, input, output):
        layer_outputs.append(output)

    # Register the hook for the layer of interest
    layers = {'conv1': net.conv1, 'conv2': net.conv2}
    layer = layers[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    # Pass the input image through the model up to the chosen layer
    _ = net(input_image.unsqueeze(0))
    hook_handle.remove()

    # Plot the activation maps of the chosen layer
    num_feature_maps = layer_outputs[0].shape[1]
    cols = 8
    rows = (num_feature_maps + cols - 1) // cols  # Round up the division


    fig = plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(num_feature_maps):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(layer_outputs[0][0, i].detach().numpy(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Feature map {i + 1}")

    plt.show()
    
def visualize_loss_landscape_unit(net, testloader, testset, device, layer_name, unit_idx, principal_components):
    losses = []
    with torch.no_grad():
        for data in testloader:
            images, labels, loss = data
            images, labels = images.to(device), labels.to(device) # Move data to the GPU
            losses.append(loss[:, unit_idx])
    losses_vis = torch.concat(losses)

    # Sample data:
    x = to_np(principal_components[:,0])
    y = to_np(principal_components[:,1])
    z = to_np(losses_vis)

    # Create a grid
    xi = np.linspace(min(x), max(x), 100)
    yi = np.linspace(min(y), max(y), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    print(z.shape)
    print(x.shape)
    print(y.shape)

    # Interpolate z values onto this grid
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Xi, Yi, Zi, cmap='jet')
    fig.colorbar(surf)

    ax.set_xlabel('test CIFAR10 PC 1')
    ax.set_ylabel('test CIFAR10 PC 2')
    ax.set_zlabel('unit activation loss')
    ax.set_title('unit activation loss over test images')

    # Optionally, plot the original points on the 3D surface
    ax.scatter(x, y, z, color='black', s=0.01)
    ax.view_init(elev=30, azim=45)  # adjust these values as needed

    fig.show()
    
    # Interpolate z values onto this grid
    Zi = griddata((x, y), z, (Xi, Yi), method='linear')

    plt.figure(figsize=(8,6))
    cp = plt.contourf(Xi, Yi, Zi, cmap='jet', levels=50)
    plt.colorbar(cp)
    plt.xlabel('test CIFAR10 PC 1')
    plt.ylabel('test CIFAR10 PC 2')
    plt.title('unit activation loss over test images')
    plt.scatter(x, y, color='black', s=0.05)  # Optionally, plot the original points
    plt.show()    
    
    sort_indexes = np.argsort(to_np(losses_vis))
    PCA_testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    images, labels, _ = next(iter(PCA_testloader))
    images = images / 2 + 0.5
    # Display top 20 images with lowest loss
    fig, axs = plt.subplots(2, 10, figsize=(20,6))
    for i in range(2):
        for j in range(10):
            idx = sort_indexes[i*10 + j]
            axs[i, j].imshow(images[idx].permute(1, 2, 0))
            axs[i, j].set_title(f"Lowest Loss: {to_np(losses_vis)[idx]:.2f}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()

    # Display top 20 images with highest loss
    fig, axs = plt.subplots(2, 10, figsize=(20,6))
    for i in range(2):
        for j in range(10):
            idx = sort_indexes[-(i*10 + j + 1)]
            axs[i, j].imshow(images[idx].permute(1, 2, 0))
            axs[i, j].set_title(f"Highest Loss: {to_np(losses_vis)[idx]:.2f}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()
    
def visualize_activation_loss_latent_dim_unit(net,
                                              testloader,
                                              device,
                                              layer_name,
                                              principal_components,
                                              feature_latent_factor = 6):
    
    from scipy.stats import norm
    losses = []
    with torch.no_grad():
        for data in testloader:
            images, labels, loss = data
            images, labels = images.to(device), labels.to(device) # Move data to the GPU
            losses.append(loss)
    losses_vis = torch.concat(losses).detach().cpu().numpy()
    
    # # Generating some example data
    fig, axes = plt.subplots(16, feature_latent_factor, figsize=(feature_latent_factor,16))

    for i in range(16):
        for j in range(feature_latent_factor):
            idx = j * 16 + i
            ax = axes[i, j]

            # Scatter plot
            ax.scatter(losses_vis[:,i], principal_components[:, idx], s=5, c='blue', alpha=0.5)
            # Compute correlation
            corr = np.corrcoef(losses_vis[:,i], principal_components[:, idx])[0, 1]
            ax.set_title(f"Corr: {corr:.2f}")

            # Hide axis labels for clarity
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    # Generating a random sample array for illustration
    samples = losses_vis

    # Calculate the distribution for the entire array
    hist, bins = np.histogram(samples, bins=50, density=True)

    # Separate the array into negative and positive parts
    negative_samples = samples[samples < 0]
    positive_samples = samples[samples > 0]

    # Calculate the distribution for the negative values
    neg_hist, neg_bins = np.histogram(negative_samples, bins=25)

    # Calculate the distribution for the positive values
    pos_hist, pos_bins = np.histogram(positive_samples, bins=25)

    # Calculate Gaussian distributions for negative and positive values
    neg_x = np.linspace(negative_samples.min(), positive_samples.max(), 100)
    neg_y = norm.pdf(neg_x, negative_samples.mean(), negative_samples.std())

    pos_x = np.linspace(negative_samples.min(), positive_samples.max(), 100)
    pos_y = norm.pdf(pos_x, positive_samples.mean(), positive_samples.std())
    
    com_y = (neg_y + pos_y)/2

    # Plotting
    plt.figure(figsize=(10, 4))

    # Overall Distribution
    plt.subplot(2, 2, 1)
    plt.bar(bins[:-1], hist, width=np.diff(bins), align="edge", alpha=0.7)
    plt.title("Overall Distribution")

    # Separated Distributions
    plt.subplot(2, 2, 2)
    plt.bar(neg_bins[:-1], neg_hist, width=np.diff(neg_bins), align="edge", color="red", alpha=0.7, label="Negative Values")
    plt.bar(pos_bins[:-1], pos_hist, width=np.diff(pos_bins), align="edge", color="green", alpha=0.7, label="Positive Values")
    plt.legend()
    
    # Gaussian
    plt.subplot(2, 2, 3)
    plt.plot(neg_x, neg_y, color="darkred", label="Neg Gaussian")
    plt.plot(pos_x, pos_y, color="darkgreen", label="Pos Gaussian")
    plt.plot(pos_x, com_y, color="darkblue", label="Combined two-mode Gaussian")
    plt.legend()
    
    
    plt.title("Separated Distributions with 2-mode Gaussian estimation")

    plt.tight_layout()
    plt.show()
    
def visualize_latent_2_latent_dim_unit(net,
                                       testloader,
                                       device,
                                       layer_name,
                                       principal_components,
                                       feature_latent_factor = 6):
    
    from scipy.stats import norm
    losses = []
    with torch.no_grad():
        for data in testloader:
            images, labels, loss = data
            images, labels = images.to(device), labels.to(device) # Move data to the GPU
            losses.append(loss)
    losses_vis = torch.concat(losses).detach().cpu().numpy()
    
    # # Generating some example data
    fig, axes = plt.subplots(16, feature_latent_factor, figsize=(feature_latent_factor,16))

    for i in range(16):
        for j in range(feature_latent_factor - 1):
            idx1 = j * 16 + i
            idx2 = (j + 1) * 16 + i
            ax = axes[i, j]

            # Scatter plot
            ax.scatter(principal_components[:, idx1], principal_components[:, idx2], s=5, c='blue', alpha=0.5)
            # Compute correlation
            corr = np.corrcoef(principal_components[:,idx1], principal_components[:, idx2])[0, 1]
            ax.set_title(f"Corr: {corr:.2f}")

            # Hide axis labels for clarity
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    
    
def visualize_multiple_optimized_inputs(net,
                                        testset,
                                        idx_list,
                                        vae,
                                        device,
                                        layer_name, 
                                        unit_idx,
                                        per_feature_num_latent,
                                        num_features,
                                        num_initializations, 
                                        num_iterations, 
                                        optimize_object = 'latent', 
                                        use_regularization = True, 
                                        use_projection = True, 
                                        use_likelihood = True, 
                                        use_energy = True, 
                                        energy_scale = 1.0, 
                                        init = 'normal', 
                                        learning_rate=0.1, 
                                        noise_coef = 0.1):

    cols = min(4, num_initializations)
    rows = math.ceil(num_initializations / cols)
    
    fig1, axes1 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes1 = axes1.flat
    
    fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes2 = axes2.flat
    
    fig3, axes3 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes3 = axes3.flat
    
    init_data = None
    if init == 'data':
        fig4, axes4 = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes4 = axes4.flat
        init_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
            
    optimized_inputs_all = []
    optimized_scores_all = []
    for i in range(num_initializations):
        optimized_inputs = []
        if init == 'data':
            init_data = torch.cat([testset[i][0].unsqueeze(0) for i in idx_list])
            init_data = init_data[i].unsqueeze(0)
            init_image = init_data.detach()
            init_image = ((init_image - init_image.min()) / (init_image.max() - init_image.min())).squeeze().cpu().numpy()
            axes4[i].imshow(init_image.transpose((1, 2, 0)))
            #axes4[i].set_title(f"initialized input {i + 1}")
            axes4[i].set_xticks([])
            axes4[i].set_yticks([])
            optimized_inputs.append(init_image)
            
        if optimize_object == 'latent':
            optimized_input, optimized_input_increase_prior, optimized_input_decrease_prior, scores = optimize_latent_for_unit_activation(
                net,
                vae,
                layer_name, 
                unit_idx, 
                per_feature_num_latent = per_feature_num_latent,
                num_features = num_features,
                learning_rate = learning_rate, 
                num_iterations = num_iterations,
                use_projection = use_projection,
                use_likelihood = use_likelihood,
                use_energy = use_energy,
                use_regularization = use_regularization,
                init = init,
                init_data = init_data.to(device),
                energy_scale = energy_scale)
            
            image_increase_prior = optimized_input_increase_prior.squeeze().cpu().numpy()
            axes2[i].imshow(image_increase_prior.transpose((1, 2, 0)))
           # axes2[i].set_title(f"prior increase input {i + 1}")
            axes2[i].set_xticks([])
            axes2[i].set_yticks([])
            optimized_inputs.append(image_increase_prior)
            
            image_decrease_prior = optimized_input_decrease_prior.squeeze().cpu().numpy()
            axes3[i].imshow(image_decrease_prior.transpose((1, 2, 0)))
            #axes3[i].set_title(f"prior decrease input {i + 1}")
            axes3[i].set_xticks([])
            axes3[i].set_yticks([])
            optimized_inputs.append(image_decrease_prior)
            
            optimized_scores_all.append(scores)
        
        else:
            optimized_input = optimize_input_for_unit_activation(
                net, 
                layer_name, 
                unit_idx, 
                learning_rate = learning_rate, 
                num_iterations = num_iterations,
                use_projection = use_projection,
                use_likelihood = use_likelihood,
                use_energy = use_energy,
                use_regularization = use_regularization,
                init = init,
                init_data = init_data,
                energy_scale = energy_scale)

        optimized_input = optimized_input.squeeze().cpu().numpy()
        axes1[i].imshow(optimized_input.transpose((1, 2, 0)))
        #axes1[i].set_title(f"optimized input {i + 1}")
        axes1[i].set_xticks([])
        axes1[i].set_yticks([])
        optimized_inputs.append(optimized_input)
        
        optimized_inputs_all.append(optimized_inputs)

    plt.show()
    
    return optimized_inputs_all, optimized_scores_all
    
def visualize_all_units(net, 
                        layer_names, 
                        num_initializations, 
                        num_units = None, 
                        use_regularization = True, 
                        use_projection = True, 
                        use_likelihood = True, 
                        use_energy = True, 
                        learning_rate = 0.1, 
                        noise_coef = 0.1, 
                        num_iterations = 5000):
    
    for layer_name in layer_names:
        if num_units is None:
            num_units = net._modules[layer_name].out_channels
        for unit_idx in range(num_units):
            print(f"Visualizing unit {unit_idx + 1} in layer {layer_name}:")
            visualize_multiple_optimized_inputs(net, layer_name, unit_idx, num_initializations, 
                                                use_regularization = use_regularization, 
                                                use_projection = use_projection, 
                                                use_likelihood = use_likelihood, 
                                                use_energy = use_energy, 
                                                learning_rate = learning_rate,
                                                noise_coef = noise_coef,
                                                num_iterations = num_iterations)
            print("\n")