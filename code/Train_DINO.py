import sys
sys.path.append('../modules')
sys.path.append('../third_party/image_manipulation_code')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from KLD_loss import loss_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import gc
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.manual_seed(42) 

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
from torchvision import transforms

import argparse

parser = argparse.ArgumentParser(
    description="Train a DINO encoder to predict fMRI activations from NSD")

# ─── Dataset & ROI selection ────────────────────────────────────────────────
parser.add_argument("--nsd-root", type=str,
                    default="../datasets/NSD_mix")
parser.add_argument("--subject",  type=int,   default=1)
parser.add_argument("--roi-list", type=str,
                    default="V1,V2,V3,V4,V8,LO1,LO2,V4t",
                    help="Comma-separated ROI names")

# ─── Image / dataloader hyper-params ────────────────────────────────────────
parser.add_argument("--img-resize", type=int,   default=128,
                    help="Resize shorter edge to this before ToTensor()")
parser.add_argument("--dino-size",  type=int,   default=224,
                    help="Image size fed into DINO backbone")
parser.add_argument("--batch-size", type=int,   default=64)
parser.add_argument("--num-workers",type=int,   default=4)

# ─── Optimisation / training ────────────────────────────────────────────────
parser.add_argument("--epochs",     type=int,   default=15)
parser.add_argument("--lr",         type=float, default=1e-4)

# ─── Runtime & output ───────────────────────────────────────────────────────
parser.add_argument("--device",     type=str,   default="auto",
                    help='Torch device string ("cuda:0", "cpu", or "auto")')
parser.add_argument("--ckpt-dir",   type=str,   default="../checkpoints/",
                    help="Directory to save best checkpoints")
parser.add_argument("--file-name",  type=str,
                    default="dino_encoder-{hemi}-{rois}-{epoch:02d}-{val_loss:.2f}")

args = parser.parse_args()
# ════════════════════════════════════════════════════════════════════════════


# ─── Replace former literals with CLI values ────────────────────────────────
roi_list       = args.roi_list.split(",")
nsd_root       = args.nsd_root
subject        = args.subject

batch_size     = args.batch_size 
num_workers    = args.num_workers

num_epochs     = args.epochs
learning_rate  = args.lr

if args.device == "auto":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args.device)

file_name       = args.file_name
ckpt_dir        = args.ckpt_dir

# Define NSD dataset

subjects = [1,2,3,4,5,6,7,8]
subjects.remove(1)
modulation_vectors = []

nsd_transform = transforms.Compose([
    #transforms.Resize((32, 32)),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
    #transforms.Normalize((0.0,), (2,))
])


subj1_lh_rois = ["aTL-faces", "EBA", "FBA-1", "FFA-1", "mfs-words",
                 "OFA", "OPA", "OWFA", "PPA", "RSC", "VWFA-1", "VWFA-2", "mTL-words"]

sub_1_lh_rois_explored = ["OFA", "FFA-1", "EBA", "VWFA-1", "OPA", "PPA", "RSC", "OWFA"]

sub_1_lh_rois_unknown = ["IPS4", "SPL1", "TO1", "PHC2", "VO2", "PGi", "VVC", "TE2a", "STSdp", "VMV3"]
#sub_1_rh_rois_unknown = ["FBA-1", "FBA-2", "aTL-faces", "OFA"]
sub_1_lh_rois_unknown_ex = ["STSdp", "31a", "31pd", "DVT", "FST", "PGp", "PGs", "PH"]
sub_1_lh_rois_unknown_ex_2 = ["TE2p", "PHT", "TPOJ1", "TPOJ2", "TPOJ3", "IP0", "IP1", "IP2"]
sub_1_lh_rois_mix = ["V1", "V2", "V3", "V4", "V8", "LO1", "LO2", "V4t"]

trainset=NaturalScenesDataset(
                    root=nsd_root,
                    subject=1,
                    partition='train',
                    hemisphere='lh',
                    transform = nsd_transform,
                    roi=roi_list,
                    return_average=True,
                    tval_threshold=5
)

testset = NaturalScenesDataset(
                    root=nsd_root,
                    subject=1,
                    partition='test',
                    hemisphere='lh',
                    transform = nsd_transform,
                    roi=roi_list,
                    return_average=True,
                    tval_threshold=5
)

class BrainFMRI_Dataset(Dataset):
    def __init__(self, nsd_dataset, transform=None):
        self.nsd_dataset = nsd_dataset
        self.transform = transform

    def __len__(self):
        return len(self.nsd_dataset)

    def __getitem__(self, idx):
        img, activation = self.nsd_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(activation, dtype=torch.float32)

# Define DINO-compatible image transformations
transform_dino = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)),  # ✅ Handles both cases
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create DataLoaders
train_data = BrainFMRI_Dataset(trainset, transform=transform_dino)
test_data = BrainFMRI_Dataset(testset, transform=transform_dino)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(test_data, batch_size=64)




# **Initialize the DINO Encoder for fMRI Prediction**
encoder = EncoderModule(output_size=train_data[0][1].shape[0], learning_rate=1e-4)


from pytorch_lightning.callbacks import ModelCheckpoint

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dir,   
    filename=file_name, 
    monitor="val_loss",  
    save_top_k=1,  
    mode="min" 
)

# **Train using PyTorch Lightning Trainer**
trainer = pl.Trainer(
    max_epochs=num_epochs, 
    log_every_n_steps=1, 
    accelerator="gpu", 
    devices=1,
    callbacks=[checkpoint_callback]
)
trainer.fit(encoder, train_loader, val_loader)

