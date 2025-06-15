import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

import torch
from torch.nn import Parameter
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
import torch.nn as nn

from torchvision import transforms
from tqdm import tqdm


if __name__ == "__main__":
    mp.set_start_method('spawn')

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



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


 
net = ConvNet()
net.load_state_dict(torch.load("/common/home/yz1635/DiffuseVAE/ANN_ckpt/cifar_10_two_layers_color_1201.pt"))
net.eval()

class NewCIFAR10Dataset(Dataset):
    def __init__(self, root, norm=True, transform=None, subsample_size=None, batch_size=100, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")

        if subsample_size is not None:
            assert isinstance(subsample_size, int)

        self.root = root
        self.norm = norm
        self.transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()])
        self.subsample_size = subsample_size

        # Load the dataset
        self.dataset = CIFAR10(
            self.root, train=True, download=True, transform=None, **kwargs
        )

        # Subsample the dataset (if enabled)
        if self.subsample_size is not None:
            self.dataset.data = self.dataset.data[:self.subsample_size]
            self.dataset.targets = self.dataset.targets[:self.subsample_size]

        # Precompute activation_loss for all images in batches
        """
        self.activation_loss = []
        batch_size = batch_size  
        print(f"Precomputing activation_loss for all images in batches of {batch_size}...")
        num_batches = (len(self.dataset) + batch_size - 1) // batch_size

        self.unit_idx = [(i, 5, 5) for i in range(16)]  

        for batch_idx in tqdm(range(num_batches), desc='Computing activation_loss'):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(self.dataset))
            batch_images = []
            for idx in range(batch_start, batch_end):
                img, _ = self.dataset[idx]
                img = self.transform(img)
                batch_images.append(img)

            input_images = torch.stack(batch_images, dim=0)  # Shape: [batch_size, C, H, W]
            
            # Compute activation_loss for each unit separately
            activation_loss_batch = []
            for unit_idx_item in self.unit_idx:
                activation_loss = test_input_for_unit_activation(
                    net,
                    layer_name='conv2',
                    unit_idx=unit_idx_item,
                    input_image=input_images
                )
                activation_loss_batch.append(activation_loss.detach().cpu())
            # Stack activation losses to get [num_units, batch_size], then transpose
            activation_loss_batch = torch.stack(activation_loss_batch, dim=0).T  # Shape: [batch_size, num_units]
            self.activation_loss.append(activation_loss_batch)

        # Concatenate activation losses from all batches
        self.activation_loss = torch.cat(self.activation_loss, dim=0)  # Shape: [num_images, num_units]
        print(self.activation_loss.shape)
        """

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0
        #return torch.tensor(img).permute(2, 0, 1).float(), self.activation_loss[idx]
        return torch.tensor(img).permute(2, 0, 1).float()
        
    def __len__(self):
        return len(self.dataset) if self.subsample_size is None else self.subsample_size  

if __name__ == "__main__":
    root = "/data/local/yusong/cifar-10/"
    dataset = NewCIFAR10Dataset(root)
    print(dataset[0].shape)
