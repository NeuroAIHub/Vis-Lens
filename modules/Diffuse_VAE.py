import copy
import torch
import torch.nn as nn
from DDPM import DDPMv2, DDPM
from unet_openai import SuperResModel, UNetModel
from ResNet152VAE import VAE
from BasicVAE import SimpleVAE
from spaced_diff import SpacedDiffusion
from spaced_diff_form2 import SpacedDiffusionForm2
import numpy as np


def space_timesteps(num_timesteps, desired_count, type="uniform"):
    if type == "uniform":
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired_count:
                return range(0, num_timesteps, i)
        raise ValueError(f"cannot create exactly {desired_count} steps with an integer stride")
    elif type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
        seq = [int(s) for s in list(seq)]
        return seq
    else:
        raise NotImplementedError


# Main DiffusionVAE Wrapper
class DiffusionVAE(nn.Module):
    def __init__(
        self,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super(DiffusionVAE, self).__init__()

        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        """"CIFAR-10 DDPM Decoder Configs are here"""
        ddpm_type = "form1"
        attn_resolutions = [16]
        dim_mults = [1, 2, 2, 2]

        decoder_cls = UNetModel if ddpm_type == "uncond" else SuperResModel
        decoder = decoder_cls(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=attn_resolutions,
            channel_mult=dim_mults,
            use_checkpoint=False,
            dropout=0.3,
            num_heads=8,
            z_dim=1024,
            use_scale_shift_norm=True,
            use_z=True,
        )

        # Create an EMA version of the decoder for the target network
        ema_decoder = copy.deepcopy(decoder)
        for p in ema_decoder.parameters():
            p.requires_grad = False

        # Initialize DDPM models (online and target)
        ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
        
        online_ddpm = ddpm_cls(
            decoder,
            beta_1=0.0001,
            beta_2=0.02,
            T=1000,
        )
        target_ddpm = ddpm_cls(
            ema_decoder,
            beta_1=0.0001,
            beta_2=0.02,
            T=1000,
        )

        # Initialize models 
        self.online_network = online_ddpm
        self.target_network = target_ddpm

        self.vae = SimpleVAE(
            latent_dim=1024
        )
        VAE_path = "../checkpoint/VAE_ANN_layer2_2_conv2_constrain.pt"
        self.vae.load_state_dict(torch.load(VAE_path))
        for q in self.vae.parameters():
            q.requires_grad = False
        # Model settings


        """"CIFAR-10 DDPM Train Configs are here"""
        lr = 2e-4
        cfd_rate=0.0 # Conditioning signal dropout rate as in Classifier-free guidance
        n_anneal_steps=5000 # number of warmup steps
        loss="l2"
        grad_clip = 1.0
        conditional = True
        z_cond = False

 
        # Training arguments
        
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents
        self.cfd_rate = cfd_rate
        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None


   
        self.z_cond = z_cond


    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )
    
        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        
        checkpoints=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def encode_with_vae(self, x):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_with_vae(self, z):
        return self.vae.decode(z)

    def predict(self, batch):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )
        
        #if self.eval_mode == "sample":
        if False:
            x_t = batch
            #recons = self.vae(z)
            recons = self.vae(x_t)
            recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
        
            img = batch
            recons, _, _ = self.vae(img)
            recons = 2 * recons - 1
        
            
            # DDPM encoder
            
            x_t = self.online_network.compute_noisy_input(
                2 * img - 1,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )
        

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z= None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )
