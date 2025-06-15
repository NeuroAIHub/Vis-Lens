import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from test_activation import test_input_for_unit_activation

class CustomCIFAR10(CIFAR10):
    def __init__(self, *args, preprocessing_network=None, layer_name = None, unit_idx = None, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.layer_name = layer_name
        self.unit_idx = unit_idx
        self.new_attributes = [None] * len(self)
        
        # Preprocess the dataset to compute and store the new attributes
        if preprocessing_network:
            self.new_attributes = self._preprocess_data(preprocessing_network)

    def _preprocess_data(self, net, batch_size = 2048):
        attributes = []
        data_batch = []
        # Disable gradient computation
        with torch.no_grad():
            i = 0
            # Process each batch through the network
            # Compute the activations of all units
            for data, _, _ in self:
                data_batch.append(data)
                if (i + 1) % batch_size == 0 or (i + 1) == len(self):
                    print('process datapoints:', i + 1)
                    data_batch = torch.stack(data_batch).to(self.device)
                    activation_loss_all_units = []
                    for unit_idx_item in self.unit_idx:
                        activation_loss = test_input_for_unit_activation(
                          net,
                          layer_name = self.layer_name,
                          unit_idx = unit_idx_item,
                          input_image = data_batch)
                        activation_loss_all_units.append(activation_loss.cpu())
                    attributes.extend(torch.stack(activation_loss_all_units).T)  
                    # Store activations, store them and return as processed dataset
                    data_batch = []
                i += 1
        return torch.stack(attributes, dim = 0)

    # Modify the __getitem__ method to return the new attribute alongside the data
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.new_attributes[index]
    
"""
class CustomImageNet(ImageNet):
    def __init__(self, *args, preprocessing_network=None, layer_name = None, unit_idx = None, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.device = device
        self.layer_name = layer_name
        self.unit_idx = unit_idx
        self.new_attributes = [None] * len(self)
        
        # Preprocess the dataset to compute and store the new attributes
        if preprocessing_network:
            self.new_attributes = self._preprocess_data(preprocessing_network)

    def _preprocess_data(self, net, batch_size = 2048):
        attributes = []
        data_batch = []
        # Disable gradient computation
        with torch.no_grad():
            i = 0
            # Process each batch through the network
            # Compute the activations of all units
            for data, _, _ in self:
                data_batch.append(data)
                if (i + 1) % batch_size == 0 or (i + 1) == len(self):
                    print('process datapoints:', i + 1)
                    data_batch = torch.stack(data_batch).to(self.device)
                    activation_loss_all_units = []
                    for unit_idx_item in self.unit_idx:
                        activation_loss = test_input_for_unit_activation(
                          net,
                          layer_name = self.layer_name,
                          unit_idx = unit_idx_item,
                          input_image = data_batch)
                        activation_loss_all_units.append(activation_loss.cpu())
                    attributes.extend(torch.stack(activation_loss_all_units).T)  
                    # Store activations, store them and return as processed dataset
                    data_batch = []
                i += 1
        return torch.stack(attributes, dim = 0)

    # Modify the __getitem__ method to return the new attribute alongside the data
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, self.new_attributes[index]
        
"""