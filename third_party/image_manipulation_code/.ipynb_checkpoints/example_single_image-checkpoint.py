import argparse
import os
from argparse import BooleanOptionalAction

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline

from datasets.nsd import NaturalScenesDataset
from datasets.nsd_clip import CLIPExtractor, NSDCLIPFeaturesDataset
from IPA.ip_adapter import IPAdapter
from methods.slerp import slerp
from methods.projection import project_modulation_vector
from methods.dino_encoder import EncoderModule
from torchvision import transforms


def main(cfg):

    print('#############################################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} ####')
    print('#############################################')

    folder_all = os.path.join(cfg.output_dir, f"{cfg.subject}_{cfg.roi}")

    dino_preds_max = []
    dino_preds_min = []

    # Load models
    diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(cfg.device)
    diffusion_pipeline.scheduler = DDIMScheduler.from_config(diffusion_pipeline.scheduler.config)
    diffusion_pipeline.safety_checker = None
    ip_model = IPAdapter(diffusion_pipeline, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", os.path.join(cfg.models_dir, "ip-adapter_sd15.bin"), cfg.device)
    clip_extractor = CLIPExtractor(cfg.device)
    ckpt_path = os.path.join(cfg.ckpt_dir, 'dino_vit', f'0{cfg.subject}_{cfg.roi}_0')
    ckpt_path = os.path.join(ckpt_path, sorted(list(os.listdir(ckpt_path)))[-1])
    dino_encoder = EncoderModule.load_from_checkpoint(ckpt_path, strict=False).to(cfg.device).eval()

    transform_dino = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    # Get modulation vectors from rest of subjects
    subjects = [1,2,3,4,5,6,7,8]
    subjects.remove(cfg.subject)
    modulation_vectors = []
    features = []
    for s in subjects:
        try:
            dataset_train = NSDCLIPFeaturesDataset(
                nsd=NaturalScenesDataset(
                    root=cfg.dataset_root,
                    subject=s,
                    partition='train',
                    hemisphere='both',
                    roi=cfg.roi,
                    return_average=True,
                    tval_threshold=5,
                )
            )
        except:
            continue
        modulation_vectors.append(dataset_train.get_modulation_vector())
        features.append(dataset_train.features)
    modulation_vectors = np.stack(modulation_vectors, axis=0)
    modulation_vector_max = modulation_vectors.mean(axis=0)
    modulation_vector_max = modulation_vector_max / np.linalg.norm(modulation_vector_max)
    modulation_vector_min = -modulation_vector_max
    features = np.concatenate(features, axis=0)

    if cfg.projected:
        modulation_vector_max = project_modulation_vector(features, modulation_vector_max)
        modulation_vector_min = project_modulation_vector(features, modulation_vector_min)

    # Set strengths and seeds
    ts = np.linspace(0, 1, cfg.num_frames+1)
    ts = 1 / np.exp(ts**10)[::-1]
    ts = ts - ts.min()
    ts = ts / ts.max() * cfg.t1_strength
    strengths = ts

    # Get source image
    image = Image.open(cfg.img_path)
    source_img_embeds = clip_extractor(image).detach().cpu().numpy()
    image = image.resize((512, 512))

    folder_max = os.path.join(folder_all, 'max')
    folder_min = os.path.join(folder_all, 'min')
    os.makedirs(folder_max, exist_ok=True)
    os.makedirs(folder_min, exist_ok=True)

    # Get embeddings
    endpoint_max = modulation_vector_max *  np.linalg.norm(source_img_embeds)
    endpoint_min = modulation_vector_min *  np.linalg.norm(source_img_embeds)
    target_max = torch.from_numpy(slerp(source_img_embeds, endpoint_max, cfg.num_frames+1, t1=cfg.t1_slerp)).unsqueeze(1).to(cfg.device)
    target_min = torch.from_numpy(slerp(source_img_embeds, endpoint_min, cfg.num_frames+1, t1=cfg.t1_slerp)).unsqueeze(1).to(cfg.device) 

    # Generate images
    for j, (st, tmax, tmin) in enumerate(zip(strengths, target_max, target_min)):
        
        # Maximization
        if j == 0:
            img = image
        else:
            img = ip_model.generate(
                clip_image_embeds=tmax,
                image=image,
                strength=st,
                num_samples=1,
                num_inference_steps=50, 
                seed=cfg.seed
            )[0]
        img.save(os.path.join(folder_max, f"{j}.png"))
        dino_pred = dino_encoder(transform_dino(img).to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        dino_preds_max.append(dino_pred)

        # Minimization
        if j ==0:
            img = image
        else:
            img = ip_model.generate(
                clip_image_embeds=tmin,
                image=image,
                strength=st,
                num_samples=1,
                num_inference_steps=50, 
                seed=cfg.seed
            )[0]
        img.save(os.path.join(folder_min, f"{j}.png"))
        dino_pred = dino_encoder(transform_dino(img).to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        dino_preds_min.append(dino_pred)

    np.save(os.path.join(folder_all, 'dino_preds_max.npy'), np.array(dino_preds_max))
    np.save(os.path.join(folder_all, 'dino_preds_min.npy'), np.array(dino_preds_min))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default="./data/NSD")
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument("--models_dir", type=str, default='./data/models')
    parser.add_argument("--img_path", type=str, default='bear.png')
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default="PPA")
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--t1_strength", type=float, default=0.6)
    parser.add_argument("--t1_slerp", type=float, default=1)
    parser.add_argument("--projected", action=BooleanOptionalAction, default=False)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    cfg = parser.parse_args()
    main(cfg)