# ==============================================================
#      FID@1k  –  Structured‑VAE  vs.  BrainACTIV baseline
# ==============================================================
import sys
sys.path.append('../modules')
sys.path.append('../third_party/image_manipulation_code')

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from nsd_datasets.nsd         import NaturalScenesDataset
from nsd_datasets.nsd_clip    import NSDCLIPFeaturesDataset, CLIPExtractor
from IPA.ip_adapter           import IPAdapter
from diffusers                import StableDiffusionImg2ImgPipeline, DDIMScheduler

from CLIP_VAE         import CLIPtoCLIPVAE
from methods.dino_encoder     import EncoderModule  # used only to build mod‑vec
from methods.projection import project_modulation_vector
import argparse, torch, os, sys
import numpy as np

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Structured‑VAE vs. BrainACTIV ‑ FID / activation eval")

    p.add_argument("--subject",     type=int,   default=1,
                   help="NSD subject ID")
    p.add_argument("--roi",         type=str,   default="TE2p",
                   help="ROI name (exact string)")
    p.add_argument("--unit_idx",    type=int,   default=0,
                   help="DINO unit/channel to evaluate")
    p.add_argument("--alpha",       type=float, default=1.0,
                   help="µ‑interpolation weight (α)")
    p.add_argument("--gamma_sd",    type=float, default=1.0,
                   help="SDEdit strength γ")
    p.add_argument("--ridge_lamb",  type=float, default=10.0,
                   help="ridge‑regression λ")
    p.add_argument("--batch",       type=int,   default=32,
                   help="DataLoader batch size")
    p.add_argument("--seed",        type=int,   default=42,
                   help="random seed")
    p.add_argument("--nsd-root",    type=str,   default="../datasets/NSD_explored",
                   help="nsd data root dir")   
    p.add_argument("--vae-path",    type=str,   default="../checkpoints/LW_CLIPtoCLIP_VAE_NSD_TE2p_Constrain_zdim_2048_KLD_01_lamb_01.pt",
                   help="vae checkpoint path")       
    p.add_argument("--dino-path",   type=str,   default="../checkpoints/dino_encoder-lh-explored-8-ROI-ex-epoch=04-val_loss=0.11.ckpt",
                   help="dino checkpoint path")
    p.add_argument("--device",      type=str,   default=torch.device('cuda:0'),
                   help="'cpu', 'cuda:N' or 'auto'")

    return p.parse_args()

cfg = parse_args()

# ---------------- hyper‑parameters -----------------------------
SUBJECT     = cfg.subject
ROI         = cfg.roi
UNIT_IDX    = cfg.unit_idx
ALPHA       = cfg.alpha
GAMMA_SD    = cfg.gamma_sd
RIDGE_LAMB  = cfg.ridge_lamb
BATCH       = cfg.batch
SEED        = cfg.seed
NSD_ROOT    = cfg.nsd_root
VAE_PATH    = cfg.vae_path
DINO_PATH   = cfg.dino_path

if cfg.device == "auto":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(cfg.device)
# ---------------------------------------------------------------

# --- print run configuration ----------------------------------
print("\n╓────────────────  Run configuration  ────────────────╖")
print(f"║ Subject / ROI        : {SUBJECT:<3} / {ROI:<10}              ║")
print(f"║ Unit‑idx (DINO)      : {UNIT_IDX:<28}║")
print(f"║ α (µ‑interp)         : {ALPHA:<28}║")
print(f"║ γ (SDEdit strength)  : {GAMMA_SD:<28}║")
print(f"║ Ridge λ              : {RIDGE_LAMB:<28}║")
print(f"║ Batch size           : {BATCH:<28}║")
print(f"║ Random seed          : {SEED:<28}║")
print(f"║ Device               : {DEVICE!s:<28}║")
print("╙────────────────────────────────────────────────────╜\n")


# ==============================================================
# 1.  DATA  ––  NSD test split (≈1 k images)
# ==============================================================

tf_nsd = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
])

test_nsd = NaturalScenesDataset(
        root=NSD_ROOT,
        subject=SUBJECT,
        partition='test',
        hemisphere='lh',
        roi=ROI,
        transform=tf_nsd,
        return_average=True,
        tval_threshold=5.0
)


# ==============================================================
# 2.  MODELS  ––  CLIP extractor, VAE, IP‑Adapter
# ==============================================================
            # laion‑ViT‑H‑14

vae = CLIPtoCLIPVAE(latent_dim=2048, input_dim=1024).to(DEVICE)
vae.load_state_dict(torch.load(
        VAE_PATH,
        map_location=DEVICE))
vae.eval()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(DEVICE)
pipe.scheduler, pipe.safety_checker = (
        DDIMScheduler.from_config(pipe.scheduler.config), None)
ip = IPAdapter(pipe,
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "../third_party/image_manipulation_code/models/ip-adapter_sd15.bin",
        DEVICE)

ckpt_path_dino = DINO_PATH
encoder_dino = EncoderModule.load_from_checkpoint(ckpt_path_dino, strict=False).eval().to(DEVICE)

# ==============================================================
# 3.  BrainACTIV modulation vector (train split, same ROI)
# ==============================================================

train_nsd = NaturalScenesDataset(
        root=NSD_ROOT,
        subject=SUBJECT,
        partition='train',
        hemisphere='lh',
        roi=ROI,
        transform=None,            # raw PIL for CLIP features
        return_average=True,
        tval_threshold=5.0
)
train_clip = NSDCLIPFeaturesDataset(train_nsd)        # CLIP features
test_clip = NSDCLIPFeaturesDataset(test_nsd)  

dl_test = DataLoader(test_clip, batch_size=BATCH,
                     shuffle=False, num_workers=4, pin_memory=True)

X = torch.from_numpy(train_clip.features).float()
X = X / X.norm(dim=1, keepdim=True)
Y = torch.from_numpy(train_nsd.activations).float().unsqueeze(1)
C = X.T @ X + RIDGE_LAMB * torch.eye(X.shape[1])
W = torch.linalg.solve(C, X.T @ Y).squeeze(1).numpy()        # (1024,)
w_hat = W / np.linalg.norm(W)
w_hat = project_modulation_vector(train_clip.features, w_hat)  # projection
w_hat = torch.from_numpy(w_hat).unsqueeze(0).to(DEVICE)      


# ==============================================================
# 4.  GENERATION  ––  our µ‑pos   &   BrainACTIV
# ==============================================================
def slerp(val, low, high):
    """Spherical interpolation between low and high."""
    low_norm = low / low.norm(dim=-1, keepdim=True)
    high_norm = high / high.norm(dim=-1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(-1))
    so = torch.sin(omega)
    so = torch.where(so == 0, torch.ones_like(so), so)  # Avoid division by zero
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(-1) * low + (torch.sin(val * omega) / so).unsqueeze(-1) * high
    return res

@torch.no_grad()
def dino_mean(pil_list):
    tf = transforms.Compose([transforms.Resize(224),
                             transforms.ToTensor(),
                             transforms.Normalize([.485,.456,.406],
                                                  [.229,.224,.225])])
    batch = torch.stack([tf(p) for p in pil_list]).to(DEVICE)
    act   = encoder_dino(batch)[:, UNIT_IDX]
    return act.mean().item()

def to_pil512(t: torch.Tensor) -> Image.Image:
    """
    Tensor (C,H,W) in [0,1]  ->  PIL.Image RGB 512×512
    """
    if t.ndim == 3 and t.shape[0] in {1, 3}:          # tensor
        t = t.clamp(0, 1) * 255                       # [0,1] → [0,255]
        t = t.to(torch.uint8)                         # uint8
        # (C,H,W) → (H,W,C) for PIL
        arr = t.cpu().permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:                         # 1‑channel → 3‑channel
            arr = np.repeat(arr, 3, axis=2)
        pil = Image.fromarray(arr, mode="RGB")
    elif isinstance(t, Image.Image):
        pil = t.convert("RGB")
    else:
        raise TypeError(f"Unsupported type {type(t)}")

    return pil.resize((512, 512), Image.BICUBIC)
to_pil = transforms.ToPILImage()

def tensor_to_rgb_pil(t: torch.Tensor, size=(512, 512)) -> Image.Image:
    """
    Safe conversion: (C,H,W) tensor in [0,1]  →  PIL RGB  (uint8, size 512×512).
    Works for 1‑channel or 3‑channel input – nothing else.
    """
    # clamp & uint8
    t = t.clone().detach().cpu().float().clamp_(0, 1).mul_(255).byte()

    if t.dim() != 3 or t.shape[0] not in (1, 3):
        raise ValueError(f"Unexpected tensor shape {tuple(t.shape)}")

    if t.shape[0] == 1:                 # grey → repeat three times
        t = t.repeat(3, 1, 1)

    # C,H,W  →  H,W,C
    arr = t.permute(1, 2, 0).contiguous().numpy()

    pil = Image.fromarray(arr, mode="RGB")
    return pil.resize(size, Image.BICUBIC)


real, pos, neg, bmax, bmin = [], [], [], [], []

for imgs, clips, _ in tqdm(dl_test, total=len(dl_test), desc="Generating"):
    imgs_rgb = [tensor_to_rgb_pil(t) for t in imgs]   # *guaranteed* RGB uint8

    real.extend(imgs_rgb)

    clips = clips.to(DEVICE).float()              # (B,1024)

    # ours: μ_pos / μ_neg
    with torch.no_grad():
        mu, logvar = vae.encode(clips)
        mu_pos = slerp(ALPHA, mu,  torch.ones_like(mu))
        mu_neg = slerp(ALPHA, mu, -torch.ones_like(mu))
        clip_pos = vae.decode(mu_pos)
        clip_neg = vae.decode(mu_neg)

    pos_batch = ip.generate(clip_image_embeds=clip_pos,
                            image=imgs_rgb, strength=GAMMA_SD,
                            num_inference_steps=50, num_samples=1, seed=SEED)
    neg_batch = ip.generate(clip_image_embeds=clip_neg,
                            image=imgs_rgb, strength=GAMMA_SD,
                            num_inference_steps=50, num_samples=1, seed=SEED)
    pos.extend(pos_batch)
    neg.extend(neg_batch)

    # BrainACTIV: max / min
    endp  = clips.norm(dim=1, keepdim=True) * w_hat          # (B,1024)
    clip_max = slerp(ALPHA, clips,  endp)
    clip_min = slerp(ALPHA, clips, -endp)

    bmax_batch = ip.generate(clip_image_embeds=clip_max,
                             image=imgs_rgb, strength=GAMMA_SD,
                             num_inference_steps=50, num_samples=1, seed=SEED)
    bmin_batch = ip.generate(clip_image_embeds=clip_min,
                             image=imgs_rgb, strength=GAMMA_SD,
                             num_inference_steps=50, num_samples=1, seed=SEED)

    bmax.extend(bmax_batch)
    bmin.extend(bmin_batch)
    if len(real) >= 1000:     # 1 k originals  →  2 k synthetics per method
        break
# keep exactly 1 000 per split
real, pos, neg, bmax, bmin = [lst[:1000] for lst in
                              (real, pos, neg, bmax, bmin)]


# ==============================================================
#  DINO mean activations
# ==============================================================

print("\n─── DINO‑predicted mean activation (unit {}) ──".format(UNIT_IDX))
print(f"original : {dino_mean(real):7.3f}")
print(f"μ pos    : {dino_mean(pos):7.3f}")
print(f"μ neg    : {dino_mean(neg):7.3f}")
print(f"baseline max : {dino_mean(bmax):7.3f}")
print(f"baseline min : {dino_mean(bmin):7.3f}")

# ==============================================================
#  FID @ 2 k   (μ_pos+μ_neg)  vs  (max+min)
# ==============================================================

prep = transforms.Compose([transforms.Resize(299, Image.BICUBIC),
                           transforms.PILToTensor()])   # uint8

def to_uint8(img_list):                       # → (N,3,299,299) uint8
    return torch.stack([prep(p) for p in img_list])

real2k   = to_uint8(real + real)              # duplicate → 2 000
ours2k   = to_uint8(pos  + neg)               # 2 000
bact2k   = to_uint8(bmax + bmin)              # 2 000

def fid(a_real, a_fake):
    fid = FrechetInceptionDistance(feature=2048)
    fid.update(a_real , real=True)
    fid.update(a_fake , real=False)
    return fid.compute().item()

fid_ours = fid(real2k, ours2k)
fid_bact = fid(real2k, bact2k)

print("\n───  FID @ 2 k  ───────────────────────────────")
print(f"Ours  (μ_pos + μ_neg)   : {fid_ours:6.2f}")
print(f"Baseline (max + min)    : {fid_bact:6.2f}")
print("───────────────────────────────────────────────")
