### Environment Setup
pip install -r requirements.txt

### Data preparation

- NSD dataset: change to third_party/image_manipulation_code dir, run dataset_setup.py
- Imagenet-mini/CIFAR-10 can be downloaded from web and put into datasets folder


### Train VAE on NSD Dataset 

```
python3 NSD_CLIP_VAE_Train.py \
    --nsd-root ../datasets/NSD_mix \
    --roi-list V1,V2,V3 \
    --roi-idx 2 \
    --epochs 200 \
    --batch-size 64 \
    --lr 5e-4 \
    --z-dim 1024 \
    --save-path ../checkpoints/models.pt
```

Note: 

- z-dim is the number of dimensions of VAE latent space 
- pass a list of rois, use unit_idx to select the roi that its activations are used to construct latent space(For example, when use roi-idx 2 for roi-list V1,V2,V3(seperated by comma), the activations of region V3 will be used to constrain VAE latent space)

### Train VAE on ANN Unit Activations

```
python3 ANN_CLIP_VAE_Train.py \
        --data-dir ../datasets/imagenet-mini \
        --device cuda:2 \
        --layer-name layer1_0_conv1 \
        --kernel-idx "(5,5)" \
        --unit-idx 27 \
        --epochs 200 \
        --lr 1e-4 \
        --z-dim 1024 \
        --save-path ../checkpoints/run_layer1_0_conv1_unit_27.pt
```

Note: 

- To train SIMCLR, refer to this one: https://github.com/google-research/simclr
- layer names are pre-defined names in SIMCLR

### Train the In-silico brain simulator.
```
python3 Train_DINO.py \
  --nsd-root ../datasets/NSD_mix \
  --roi-list V1,V2,V3 \
  --device cuda:2 \
  --ckpt-dir ../checkpoints/ \
  --file-name dino_encoder-lh-mix \
  --epochs 15
```

Note: 

- The DINO predictor will be trained to predict the ROI activations of input roi-list.
- For example, the DINO trained with above script will generate the predicted activations of regions V1, V2, V3, in order.

### Test on complete NSD testset
```
python3 NSD_Eval.py \
  --roi PPA \
  --unit_idx 5 \
  --device "cuda:1"\
  --nsd-root ../datasets/NSD_explored \
  --vae-path ../checkpoints/LW_CLIPtoCLIP_VAE_NSD_PPA_Constrain_zdim_2048_KLD_01_lamb_01.pt \
  --dino-path ../checkpoints/dino_encoder-lh-explored-8-ROI-ex-epoch=04-val_loss=0.11.ckpt
```
Note:

- Unit-idx is the idx of Regions in the ROIs list when trainning the DINO predictor.

### Run optimization
```
python3 Optimization.py 
```

Note:

- This will run activation optimization with organized landscape on CIFAR-10 datasets.

### Visualization
Run scripts in Visualization.ipynb for visualization of results.

### Run Baseline only:
Switch to third_party/image_manipulation_code
```
python3 main_experiment.py \
  --dataset_root "../../datasets/NSD_ex_1" \
  --roi "31pd" \
  --t1_slerp 1.0 \
  --device "cuda:0" \
  --output_dir "./output"
```