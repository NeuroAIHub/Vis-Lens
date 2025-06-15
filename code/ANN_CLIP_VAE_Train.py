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
 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


import sys 
sys.path.insert(0,'../modules')
sys.path.append('../third_party/image_manipulation_code')
torch.manual_seed(42) 

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from KLD_loss import loss_function
#from nnfabrik.builder import get_data, get_model, get_trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import gc
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split

import sys 

import argparse
import os
from argparse import BooleanOptionalAction

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline

from nsd_datasets.nsd import NaturalScenesDataset
from nsd_datasets.nsd_clip import CLIPExtractor, NSDCLIPFeaturesDataset
from IPA.ip_adapter import IPAdapter
from methods.slerp import slerp
from methods.projection import project_modulation_vector
from methods.dino_encoder import EncoderModule
from CLIP_VAE import CLIPtoCLIPVAE  # CLIP-to-CLIP VAE model

from torchvision import transforms

sys.path.append("../")
import modules

# Load the main checkpoint
ckpt = torch.load('../checkpoints/resnet50_imagenet_bs2k_epochs600_linear.pth.tar')
ckpt['hparams'].encoder_ckpt = '../checkpoints/checkpoint-125100.pth.tar'
hparams = ckpt['hparams']


# Initialize the model
ANN_model = modules.ssl.SSLEval(hparams)
ANN_model.encoder.eval()
ANN_model.eval()

print("ANN Begin!")

from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

# ====== Command-line interface ==============================================
import argparse
parser = argparse.ArgumentParser(
    description="Extract ANN activations for a chosen layer/unit, "
                "pair them with CLIP features, and train a CLIP-to-CLIP VAE")

parser.add_argument("--data-dir",   type=str, help="Path to the dataset")
parser.add_argument("--layer-name", type=str,
                    default="layer2_2_conv2",
                    choices=["layer1_0_conv1", "layer1_0_conv2", "layer2_2_conv2"],
                    help="Name of the conv layer in the ResNet backbone")
parser.add_argument("--kernel-idx",   type=str,
                    default="(5,5)",
                    help='Unit index inside the feature map. '
                         'For spatial units give a tuple, e.g. "(h,w)"; '
                         'for channel units give an int, e.g. "128".')
parser.add_argument("--unit-idx",    type=int,   default=0,
                    help="Which column of activation_loss to constrain (KLD)")

# ---------------------------------------------------------------------------
# VAE / optimisation
# ---------------------------------------------------------------------------
parser.add_argument("--z-dim",      type=int,   default=2048)
parser.add_argument("--epochs",     type=int,   default=150)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--step-size",  type=int,   default=10)
parser.add_argument("--gamma",      type=float, default=0.9)

# ---------------------------------------------------------------------------
# Runtime / output
# ---------------------------------------------------------------------------
parser.add_argument("--device",     type=str,   default="auto",
                    help='Torch device ("cuda:0", "cpu", or "auto")')
parser.add_argument("--save-path",  type=str,
                    default="../checkpoints/"
                            "ANN_LW_CLIPtoCLIP_VAE_layer2_2_conv2_unit_69.pt")

args = parser.parse_args()
# ============================================================================


# ---------- Map CLI args to variables formerly hard-coded -------------------
data_dir        = args.data_dir

layer_name      = args.layer_name
kernal_idx        = eval(args.kernel_idx)     # "(5,5)" → (5, 5)  or "128" → 128
unit_idx  = args.unit_idx

z_dim           = args.z_dim
num_epochs      = args.epochs
lr = args.lr
if args.device == "auto":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

save_path       = args.save_path


def test_input_for_unit_activation(net, layer_name, unit_idx, input_images):
    net.eval()
    target_activation = [None]

    def hook(module, input, output):
        if isinstance(unit_idx, tuple):
            
            activation = torch.mean(output, dim=(2, 3)) 
            target_activation[0] = activation
        else:
            target_activation[0] = output[:, unit_idx].mean(1).mean(1)

    # Register the hook to capture activations
    layers = {
        'layer1_0_conv1': net.encoder.model.convnet.layer1[0].conv1,
        'layer1_0_conv2': net.encoder.model.convnet.layer1[0].conv2,
        'layer2_2_conv2': net.encoder.model.convnet.layer2[2].conv2,
    }
    layer = layers[layer_name]
    hook_handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = net.encoder(input_images) 
    hook_handle.remove()  
    return target_activation[0]


transform = transforms.Compose(
                [transforms.RandomResizedCrop(128, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]
                )
dataset = datasets.ImageFolder(root=data_dir + "/train", transform=transform)
data_loader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
with torch.no_grad():
	train_feature_maps = []
	images = []
	for batch in data_loader:
		input_image = batch[0].to(device)
		activation_loss = test_input_for_unit_activation(
            ANN_model.to(device),
            layer_name=layer_name,
            unit_idx=kernal_idx,
            input_images=input_image
        )
        # Gather activations across all GPUs
		train_feature_maps.append(activation_loss.cpu())
		images.append(input_image.cpu())
	
	# Concatenate feature maps after gathering from all GPUs
	train_feature_maps_tensor = torch.cat(train_feature_maps, dim=0)
	images = torch.cat(images, dim=0)

print("All images:", images.shape)
print("Activation loss shape:", train_feature_maps_tensor.shape)
print("Max loss:", torch.max(train_feature_maps_tensor))
print("Min loss:", torch.min(train_feature_maps_tensor))

print("ANN Activation Processed!")



import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

# ----------------------------
# CLIP Feature Extractor
# ----------------------------

class CLIPExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    @torch.no_grad()
    def forward(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
        output = self.clip(inputs).image_embeds[0]
        return output

    def extract_from_images(self, images, save_path=None):
        feats = []
        for img in tqdm(images, desc="Extracting CLIP features"):
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[0] in {1, 3}:
                    img = img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
                img = Image.fromarray((img * 255).astype(np.uint8)) if img.dtype == np.float32 else Image.fromarray(img)
            feat = self(img).detach().cpu().numpy()
            feats.append(feat)
        features = np.stack(feats, axis=0).astype(np.float32)
        if save_path is not None:
            np.save(save_path, features)
        return features

def compute_or_load_clip_features(images, save_dir="./", filename="clip_features.npy"):
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        print(f"[INFO] Loading CLIP features from {save_path}")
        return np.load(save_path)
    print("[INFO] CLIP features not found. Computing...")
    extractor = CLIPExtractor()
    features = extractor.extract_from_images(images, save_path=save_path)
    print("[INFO] CLIP feature extraction complete.")
    return features


# ----------------------------
# Custom Dataset with CLIP
# ----------------------------

class CustomDatasetWithCLIP(Dataset):
    def __init__(self, images, clip_features, activations, transform=None):
        self.data = images
        self.activations = activations
        self.clip_features = clip_features
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        activation_score = self.activations[idx]
        clip_feature = self.clip_features[idx]

        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
        elif image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)

        if self.transform:
            image = self.transform(image)

        return image, clip_feature, activation_score


# ----------------------------
# Example Usage
# ----------------------------

# Assuming: `images`, `activations`, `unit_losses` are available
# images: np.array or list of shape (N, H, W, C) or (N, C, H, W)
# activations: shape (N, D1)
# unit_losses: shape (N, D2)

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W) and scales to [0.0, 1.0]
])

clip_features_array = compute_or_load_clip_features(images, save_dir="../datasets/imagenet-mini/train_features", filename="clip_feats.npy")

# Initialize dataset and loader
dataset = CustomDatasetWithCLIP(images, clip_features_array, train_feature_maps_tensor,  transform=transform)
trainloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

print("ANN Dataset Ready!")

# Define Model, Optimizer, and Scheduler
vae = CLIPtoCLIPVAE(latent_dim=z_dim, input_dim=1024).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training Function
def train_clip_vae(num_epochs, dataloader):
    vae.train()
    losses_dict = {"loss": [], "recon_loss": [], "KLD": []}

    for epoch in range(num_epochs):
        train_loss = 0
        train_recon_loss = 0
        train_KLD = 0

        for batch_idx, (_, clip_embedding, activation_loss) in enumerate(dataloader):  # Use CLIP embeddings + NSD activations
            clip_embedding, activation_loss = clip_embedding.to(device), activation_loss.to(device)
            optimizer.zero_grad()

            # Forward Pass
            recon_clip, mu, logvar, _ = vae(clip_embedding)
            # Compute VAE Loss using Your Loss Function
            loss, recon, KLD = loss_function(
                recon_clip, clip_embedding, mu, logvar,
                device=device, activation_loss=activation_loss[:, unit_idx], recon_weight = 1.0, KLD_weight = 1.0, lamb = 1.0, KLD_form='two_mode')
        

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_KLD += KLD.item()

        scheduler.step()

        # Print Epoch Loss
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KLD: {train_KLD:.4f}')
        
        losses_dict["loss"].append(train_loss / len(dataloader.dataset))
        losses_dict["recon_loss"].append(train_recon_loss / len(dataloader.dataset))
        losses_dict["KLD"].append(train_KLD / len(dataloader.dataset))

    return losses_dict

# Run Training
num_epochs = num_epochs
print("Start Training!")
losses_dict = train_clip_vae(num_epochs=num_epochs, dataloader=trainloader)

# Save Model
vae_name = '../checkpoints/ANN_LW_CLIPtoCLIP_VAE_layer2_2_conv2_unit_69_Constrain_zdim_2048_KLD_1_lamb_1.pt'
torch.save(vae.state_dict(), vae_name)
print(f"Model saved at {vae_name}")