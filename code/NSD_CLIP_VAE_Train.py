import sys
sys.path.append('../modules')
sys.path.append('../third_party/image_manipulation_code')

import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from nsd_datasets.nsd import NaturalScenesDataset
from nsd_datasets.nsd_clip import NSDCLIPFeaturesDataset
from IPA.ip_adapter import IPAdapter
from methods.slerp import slerp
from methods.projection import project_modulation_vector
from methods.dino_encoder import EncoderModule

from CLIP_VAE import CLIPtoCLIPVAE  # CLIP-to-CLIP VAE model
from KLD_loss import loss_function  # Your original loss function

# Set device

import argparse

# ===== Command-line interface ===============================================
import argparse

parser = argparse.ArgumentParser(
    description="Train CLIP-to-CLIP VAE on NSD + CLIP features")

# Dataset / ROI
parser.add_argument("--nsd-root",   type=str,   default="../datasets/NSD_mix")
parser.add_argument("--roi-list",   type=str,
                    default="V1,V2,V3,V4,V8,LO1,LO2,V4t",
                    help="Comma-separated ROI names")
parser.add_argument("--roi-idx",    type=int,   default=7,
                    help="Column from activation_loss used as constraint")

# Optimisation
parser.add_argument("--epochs",     type=int,   default=150)
parser.add_argument("--batch-size", type=int,   default=100)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--step-size",  type=int,   default=10,
                    help="LR scheduler step (epochs)")
parser.add_argument("--gamma",      type=float, default=0.9,
                    help="LR scheduler decay")

# Model & checkpoint
parser.add_argument("--z-dim",      type=int,   default=2048)
parser.add_argument("--save-path",  type=str,
                    default="../checkpoints/"
                            "LW_CLIPtoCLIP_VAE_NSD_V4t_Constrain_"
                            "zdim_2048_KLD_01_lamb_01.pt")

# Runtime / hardware
parser.add_argument("--device",     type=str,   default="auto",
                    help='Torch device string ("cuda:1", "cpu", or "auto")')

args = parser.parse_args()
# ============================================================================

# ---------- Replace former literals with the parsed values ------------------
rois_list  = args.roi_list.split(",")
nsd_root   = args.nsd_root
roi_idx    = args.roi_idx
num_epochs = args.epochs
batch_size = args.batch_size
z_dim      = args.z_dim
save_path  = args.save_path

if args.device == "auto":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)
# ---------------------------------------------------------------------------

# Define NSDCLIPFeaturesDataset (Images + CLIP Features + NSD Activations)
subj1_lh_rois = ["aTL-faces", "EBA", "FBA-1", "FFA-1", "mfs-words",
                 "OFA", "OPA", "OWFA", "PPA", "RSC", "VWFA-1", "VWFA-2", "mTL-words"]
sub_1_lh_rois_unknown = ["IPS4", "SPL1", "TO1", "PHC2", "VO2", "PGi", "VVC", "TE2a", "STSdp", "VMV3"]
#sub_1_rh_rois_unknown = ["FBA-1", "FBA-2", "aTL-faces", "OFA"]
sub_1_lh_rois_unknown_ex = ["STSdp", "31a", "31pd", "DVT", "FST", "PGp", "PGs", "PH"]
sub_1_lh_rois_explored = ["OFA", "FFA-1", "EBA", "VWFA-1", "OPA", "PPA", "RSC", "OWFA"]
sub_1_lh_rois_unknown_ex_2 = ["TE2p", "PHT", "TPOJ1", "TPOJ2", "TPOJ3", "IP0", "IP1", "IP2"]
sub_1_lh_rois_mix = ["V1", "V2", "V3", "V4", "V8", "LO1", "LO2", "V4t"]



nsd_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print("NSD Begin!")
# Load NSD dataset
train_nsd = NaturalScenesDataset(
    root=nsd_root,
    subject=1,
    partition='train',
    hemisphere='lh',
    transform=nsd_transform,
    roi=rois_list,  # Load multiple ROIs
    return_average=True,
    tval_threshold=5
)

test_nsd = NaturalScenesDataset(
    root=nsd_root,
    subject=1,
    partition='test',
    hemisphere='lh',
    transform=nsd_transform,
    roi=rois_list,  # Load multiple ROIs
    return_average=True,
    tval_threshold=5
)

print("NSD Processed!")
# Convert to NSDCLIPFeaturesDataset
train_dataset = NSDCLIPFeaturesDataset(train_nsd)
test_dataset = NSDCLIPFeaturesDataset(test_nsd)

print("CLIP Processed!")
# DataLoader
trainloader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)

# Define Model, Optimizer, and Scheduler
z_dim = 2048
vae = CLIPtoCLIPVAE(latent_dim=z_dim, input_dim=1024).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
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
                device=device, activation_loss=activation_loss[:, roi_idx], recon_weight = 1.0, KLD_weight = 0.1, lamb = 0.1, KLD_form='two_mode'
            )

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
print("Start Training!")
losses_dict = train_clip_vae(num_epochs=num_epochs, dataloader=trainloader)

# Save Model
vae_name = save_path
torch.save(vae.state_dict(), vae_name)
print(f"Model saved at {vae_name}")