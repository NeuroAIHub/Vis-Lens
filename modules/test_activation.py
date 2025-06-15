import torch
from torch.nn import Parameter

def test_input_for_unit_activation(net, 
                                   layer_name, 
                                   unit_idx, 
                                   input_image):
    net.eval()
    
    
    target_activation = [None]
    
    def hook(module, input, output):
        nonlocal target_activation
        if isinstance(unit_idx, tuple):
            target_activation[0] = output[:, unit_idx[0], unit_idx[1], unit_idx[2]]
        elif isinstance(unit_idx, list):
            unit_idx_item = unit_idx[0]
            target_activation[0] = output[:, unit_idx_item[0], unit_idx_item[1], unit_idx_item[2]]
            for unit_idx_item in unit_idx[1:]:
                target_activation[0] += output[:, unit_idx_item[0], unit_idx_item[1], unit_idx_item[2]]
        else:
            target_activation[0] = output[:, unit_idx].mean(1).mean(1)

    layers = {'conv1': net.conv1, 'conv2': net.conv2}
    layer = layers[layer_name]
    hook_handle = layer.register_forward_hook(hook)
    #This registers the hook function to be called every time the specified layer produces an output, This function is called when the specified layer outputs its results. It captures the activation based on unit_idx
    
    _ = net(input_image)
    
    # Minimize loss = maximize activation
    loss = target_activation[0]
    
    return loss