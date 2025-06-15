import argparse
import os
from argparse import BooleanOptionalAction

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from nsd_datasets.nsd import NaturalScenesDataset
from nsd_datasets.nsd_clip import NSDCLIPFeaturesDataset
from IPA.ip_adapter import IPAdapter
from methods.subsets import subsets
from methods.slerp import slerp
from methods.projection import project_modulation_vector
from methods.dino_encoder import EncoderModule
from torchvision import transforms

def main(cfg):

    print('#############################################')
    print(f'### Subject {cfg.subject} ROI {cfg.roi} ####')
    print('#############################################')

    folder_all = os.path.join(cfg.output_dir, f"{cfg.subject}_{cfg.roi}_{cfg.t1_slerp}_{cfg.t1_strength}")
    os.makedirs(folder_all, exist_ok=True)

    subset_keys = ['wild_animals', 'vehicles', 'sports', 'food', 'birds', 'furniture']
    dino_preds_max = np.empty((len(subset_keys), cfg.num_images, cfg.num_frames+1), dtype=np.float32)
    dino_preds_min = np.empty((len(subset_keys), cfg.num_images, cfg.num_frames+1), dtype=np.float32)

    # Load models
    diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(cfg.device)
    diffusion_pipeline.scheduler = DDIMScheduler.from_config(diffusion_pipeline.scheduler.config)
    diffusion_pipeline.safety_checker = None
    ip_model = IPAdapter(diffusion_pipeline, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", os.path.join(cfg.models_dir, "ip-adapter_sd15.bin"), cfg.device)

    
    #ckpt_path = os.path.join(ckpt_path, sorted(list(os.listdir(ckpt_path)))[-1])
    ckpt_path = cfg.ckpt_dir
    dino_encoder = EncoderModule.load_from_checkpoint(ckpt_path, strict=False).to(cfg.device).eval()

    transform_dino = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get modulation vector (avg across all other subjects)
    subjects = [1,2,3,4,5,6,7,8]
    subjects.remove(cfg.subject)
    modulation_vectors = []
    features = []
    for s in subjects:
        dataset_train = NSDCLIPFeaturesDataset(
                nsd=NaturalScenesDataset(
                    root=cfg.dataset_root,
                    subject=s,
                    partition='train',
                    hemisphere='lh',
                    roi=cfg.roi,
                    return_average=True
                )
            )
        print("CLIP subject " + str(s) + " Processed!")
        modulation_vectors.append(dataset_train.get_modulation_vector())
        features.append(dataset_train.features)
    modulation_vectors = np.stack(modulation_vectors, axis=0)
    modulation_vector_max = modulation_vectors.mean(axis=0)
    modulation_vector_max = modulation_vector_max / np.linalg.norm(modulation_vector_max)
    modulation_vector_min = -modulation_vector_max
    features = np.concatenate(features, axis=0)

    if cfg.projected:
        print("Projected!")
        modulation_vector_max = project_modulation_vector(features, modulation_vector_max)
        modulation_vector_min = project_modulation_vector(features, modulation_vector_min)

    # Set strengths
    ts = np.linspace(0, 1, cfg.num_frames+1)
    ts = 1 / np.exp(ts**10)[::-1]
    ts = ts - ts.min()
    ts = ts / ts.max() * cfg.t1_strength
    strengths = ts

    # Set seeds
    seeds = torch.randint(-int(1e10), int(1e10), (cfg.num_images,), generator=torch.Generator().manual_seed(cfg.seed))

    # Loop over subsets
    for s_i, subset in enumerate(subset_keys):

        dataset = NSDCLIPFeaturesDataset(
            nsd=NaturalScenesDataset(
                root=cfg.dataset_root,
                subject=cfg.subject,
                partition='test',
                hemisphere='lh',
                roi=cfg.roi,
                return_average=True,
                subset=subset,
                tval_threshold=5.0
            )
        )

        mean = dataset.nsd.activations.mean()
        dists_to_mean = np.abs(dataset.nsd.activations - mean)
        sample = np.argsort(dists_to_mean)[:cfg.num_images]

        for i, idx in enumerate(sorted(sample)):

            image, source_img_embeds, _ = dataset[idx]
            _, _ = dataset.nsd[idx]
            image = image.resize((512, 512))

            folder_max = os.path.join(folder_all, subset, str(idx), 'max')
            folder_min = os.path.join(folder_all, subset, str(idx), 'min')
            folder_max_all = os.path.join(folder_all, 'max')
            folder_min_all = os.path.join(folder_all, 'min')
            folder_img_all = os.path.join(folder_all, 'img')
            os.makedirs(folder_max, exist_ok=True)
            os.makedirs(folder_min, exist_ok=True)
            os.makedirs(folder_max_all, exist_ok=True)
            os.makedirs(folder_min_all, exist_ok=True)
            os.makedirs(folder_img_all, exist_ok=True)
            endpoint_max = modulation_vector_max * np.linalg.norm(source_img_embeds)
            endpoint_min = modulation_vector_min * np.linalg.norm(source_img_embeds)
            """
            print(modulation_vector_max.shape)
            print(source_img_embeds.shape)
            print(endpoint_max.shape)
            """
            target_max = torch.from_numpy(slerp(source_img_embeds, endpoint_max, cfg.num_frames+1, t1=cfg.t1_slerp)).unsqueeze(1).to(cfg.device)
            target_min = torch.from_numpy(slerp(source_img_embeds, endpoint_min, cfg.num_frames+1, t1=cfg.t1_slerp)).unsqueeze(1).to(cfg.device)
            #print(target_max == torch.from_numpy(endpoint_max).to(cfg.device))
            for j, (st, tmax, tmin) in enumerate(zip(strengths, target_max, target_min)):
                
                # Maximize
                if j == 0:
                    img = image
                    img.save(os.path.join(folder_img_all, f"{idx}.png"))
                else:
                    img = ip_model.generate(
                        clip_image_embeds=tmax,
                        image=image,
                        strength=st,
                        num_samples=1,
                        num_inference_steps=50, 
                        seed=seeds[i].item()
                    )[0]
                if j == cfg.num_frames:
                    img.save(os.path.join(folder_max_all, f"{idx}.png"))
                img.save(os.path.join(folder_max, f"{j}.png"))
                dino_pred = dino_encoder(transform_dino(img).to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean()
                dino_preds_max[s_i, i, j] = dino_pred

                # Minimize
                if j == 0:
                    img = image
                else:
                    img = ip_model.generate(
                        clip_image_embeds=tmin,
                        image=image,
                        strength=st,
                        num_samples=1,
                        num_inference_steps=50, 
                        seed=seeds[i].item()
                    )[0]
                if j == cfg.num_frames:
                    img.save(os.path.join(folder_min_all, f"{idx}.png"))
                img.save(os.path.join(folder_min, f"{j}.png"))
                dino_pred = dino_encoder(transform_dino(img).to(cfg.device).unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean()
                dino_preds_min[s_i, i, j] = dino_pred

    np.save(os.path.join(folder_all, 'dino_preds_max.npy'), dino_preds_max)
    np.save(os.path.join(folder_all, 'dino_preds_min.npy'), dino_preds_min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str, default="../../datasets/NSD_mix")
    parser.add_argument("--models_dir", type=str, default='./models')
    parser.add_argument("--ckpt_dir", type=str, default="../../checkpoints/dino_encoder-epoch=03-val_loss=0.12.ckpt")
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default="V1")
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--t1_strength", type=float, default=1.0)  # From paper (SDEdit γ)
    parser.add_argument("--t1_slerp", type=float, default=1.0)     # From paper (interpolation α)
    parser.add_argument("--projected", action=BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = parser.parse_args()
    main(cfg)