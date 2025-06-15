# BrainACTIV:  Identifying visuo-semantic properties driving cortical selectivity using diffusion-based image manipulation

We provide minimal code to run manipulations on a given image, as well as to run our main validation experiment. 

## Setup

1. Create environment

```
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

2. To download the Natural Scenes Dataset, please fill in the [dataset form](https://forms.gle/eT4jHxaWwYUDEf2i9) and accept terms and conditions. Then, run `dataset_setup.py` to setup the files locally.

3. Clone IP-Adapter repository and download IP-Adapter weights (`ip-adapter_sd15.bin`) into a folder `models` in the current directory.
```
git clone git@github.com:tencent-ailab/IP-Adapter.git IPA
```

5. Install the [Neural Encoding Dataset (NED) toolbox](https://github.com/gifale95/NED) and download the pretrained fMRI encoders.

## Manipulating a single image

To manipulate a single image, please use `example_single_image.py` by specifying the image path, as well as the target ROI and subject number.

```
python example_single_image.py --img_path "my_image.png" --subject 5 --roi "PPA"
```

## Running main experiment

We obtain manipulations for Section 4.2 and 4.3 using `main_experiment.py`.