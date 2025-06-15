import gc
import io
import json
import multiprocessing
import os
import shutil
import urllib.request
import zipfile

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import requests
from PIL import Image
from pycocotools.coco import COCO
from scipy.io import loadmat
from tqdm import tqdm

"""
NSD code adapted from Maggie Henderson: https://github.com/mmhenderson/modfit/blob/master/code/utils/nsd_utils.py
and Andrew Luo: https://github.com/aluo-x/BrainDiVE/tree/main/data_maker
"""

# Hardcoded paths; can be changed if needed
# temp_path is used to store temporary files downloaded from the internet; can be deleted after the data is processed
data_path = './data/NSD_new/'
temp_path = './data/NSD_temp/'
nsd_url = 'https://natural-scenes-dataset.s3-us-east-2.amazonaws.com'

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 
                'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 
                'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

# All ROIs we are interested in
rois = {
    'V1v': ('prf-visualrois', 1),    
    'V1d': ('prf-visualrois', 2),	   
    'V2v': ('prf-visualrois', 3),	   
    'V2d': ('prf-visualrois', 4),	   
    'V3v': ('prf-visualrois', 5),	    
    'V3d': ('prf-visualrois', 6),	    
    'hV4': ('prf-visualrois', 7),
    'EBA': ('floc-bodies', 1),
    'FBA-1': ('floc-bodies', 2),
    'FBA-2': ('floc-bodies', 3),
    'mTL-bodies': ('floc-bodies', 4),
    'OFA': ('floc-faces', 1),
    'FFA-1': ('floc-faces', 2),
    'FFA-2': ('floc-faces', 3),
    'mTL-faces': ('floc-faces', 4),
    'aTL-faces': ('floc-faces', 5),
    'OPA': ('floc-places', 1),
    'PPA': ('floc-places', 2),
    'RSC': ('floc-places', 3),
    'OWFA': ('floc-words', 1),
    'VWFA-1': ('floc-words', 2),
    'VWFA-2': ('floc-words', 3),
    'mfs-words': ('floc-words', 4),
    'mTL-words': ('floc-words', 5),
    'early': ('streams', 1),
    'midventral': ('streams', 2),
    'midlateral': ('streams', 3),
    'midparietal': ('streams', 4),
    'ventral': ('streams', 5),
    'lateral': ('streams', 6),
    'parietal': ('streams', 7),
}

trials_per_sess = 750
sess_per_subj = 40
# Hard-coded values based on sessions that are missing for some subjects
max_sess_each_subj = [40, 40, 32, 30, 40, 32, 40, 30]

def get_session_inds_full():
    session_inds = np.repeat(np.arange(0, sess_per_subj), trials_per_sess)
    return session_inds

def load_from_hdf5(hdf5_file, keyname=None):
    data_set = h5py.File(hdf5_file, 'r')
    if keyname is None:
        keyname = list(data_set.keys())[0]
    values = np.copy(data_set[keyname])
    data_set.close()    
    return values

def get_stim_info():
    stim_info_path = os.path.join('nsddata', 'experiments', 'nsd')
    stim_info_file = 'nsd_stim_info_merged.csv'
    online_path = os.path.join(nsd_url, stim_info_path, stim_info_file)
    local_path = os.path.join(temp_path, stim_info_path, stim_info_file)
    if not os.path.exists(local_path):
        os.makedirs(os.path.join(temp_path, stim_info_path), exist_ok=True)
        urllib.request.urlretrieve(online_path, local_path)
    stim_info = pd.read_csv(local_path)
    return stim_info

def get_noise_ceiling(subject, hemisphere):
    mask = get_visual_cortex_mask(subject, hemisphere)
    ncsnr_path = os.path.join('nsddata_betas', 'ppdata', f'subj{subject:02d}', 'nativesurface', 'betas_fithrf_GLMdenoise_RR')
    ncsnr_file = f'{hemisphere}.ncsnr.mgh'
    online_path = os.path.join(nsd_url, ncsnr_path, ncsnr_file)
    local_path = os.path.join(temp_path, ncsnr_path, ncsnr_file)
    if not os.path.exists(local_path):
        os.makedirs(os.path.join(temp_path, ncsnr_path), exist_ok=True)
        urllib.request.urlretrieve(online_path, local_path)
    ncsnr = nib.load(local_path).get_fdata()
    nc = ncsnr**2 / (ncsnr**2 + 1 / 3)
    nc = nc[mask].squeeze()
    return nc

def get_shared():
    stim_info = get_stim_info()
    shared1000_path = os.path.join('nsddata', 'stimuli', 'nsd')
    shared1000_file = 'shared1000.tsv'
    online_path = os.path.join(nsd_url, shared1000_path, shared1000_file)
    local_path = os.path.join(temp_path, shared1000_path, shared1000_file)
    if not os.path.exists(local_path):
        os.makedirs(os.path.join(temp_path, shared1000_path), exist_ok=True)
        urllib.request.urlretrieve(online_path, local_path)
    shared1000 = pd.read_csv(local_path, sep='\t', header=None)
    shared1000 = shared1000[0].values - 1
    shared1000 = stim_info.loc[shared1000]['cocoId'].values
    return shared1000

def get_exp_design():
    exp_design_path = os.path.join('nsddata', 'experiments', 'nsd')
    exp_design_file = 'nsd_expdesign.mat'
    online_path = os.path.join(nsd_url, exp_design_path, exp_design_file)
    local_path = os.path.join(temp_path, exp_design_path, exp_design_file)
    if not os.path.exists(local_path):
        os.makedirs(os.path.join(temp_path, exp_design_path), exist_ok=True)
        urllib.request.urlretrieve(online_path, local_path)
    exp_design = loadmat(local_path)
    return exp_design

def get_master_image_order():    
    """
    Gather the "ordering" information for NSD images.
    masterordering gives zero-indexed ordering of indices (matlab-like to python-like), same for all subjects. 
    consists of 30000 values in the range [0-9999], which provide a list of trials in order. 
    The value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    get the image for that trial.
    """
    exp_design = get_exp_design()
    image_order = exp_design['masterordering'].flatten() - 1
    return image_order

def get_visual_cortex_mask(subject, hemisphere):
    """Get a mask indicating the visual cortex for a given subject and hemisphere."""
    selected = []
    for roi_string in ["prf-visualrois.mgz", "floc-bodies.mgz", "floc-faces.mgz", "floc-places.mgz", "floc-words.mgz", "streams.mgz"]:    
        roi_string = f'{hemisphere}.{roi_string}'
        roi_path = os.path.join('nsddata', 'freesurfer', f'subj{subject:02d}', 'label')
        local_path = os.path.join(temp_path, roi_path, roi_string)
        online_path = os.path.join(nsd_url, roi_path, roi_string)
        if not os.path.exists(local_path):
            os.makedirs(os.path.join(temp_path, roi_path), exist_ok=True)
            urllib.request.urlretrieve(online_path, local_path)
        all_roi = nib.load(local_path).get_fdata()
        selected.append(all_roi > 0)
    mask = np.logical_or.reduce(selected).squeeze()
    return mask

def coco_crop(img, cropbox_in):
    img = np.array(img)
    if type(cropbox_in) is str:
        cropbox = eval(cropbox_in)
    else:
        cropbox = cropbox_in
    top = int(img.shape[0]*cropbox[0])
    bottom = int(img.shape[0]*(1-cropbox[1]))
    left = int(img.shape[1]*cropbox[2])
    right = int(img.shape[1]*(1-cropbox[3]))
    return Image.fromarray(img[top:bottom, left:right])

def download_coco_annotation_file():
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, "r")
    zip_file_object.extractall(path=data_path)

def download_images_chunk(chunk):
    target_path = os.path.join(data_path, 'images')
    os.makedirs(target_path, exist_ok=True)

    annot_file = {
        'train2017': os.path.join(data_path, "annotations", "captions_train2017.json"),
        'val2017': os.path.join(data_path, "annotations", "captions_val2017.json"),
    }
    coco = {k: COCO(v) for k, v in annot_file.items()}
    for coco_id, coco_split, crop_box in chunk[['cocoId', 'cocoSplit', 'cropBox']].values:
        img = coco[coco_split].loadImgs(coco_id)[0]
        img_data = requests.get(img['coco_url']).content
        img = Image.open(io.BytesIO(img_data))
        img = coco_crop(img, crop_box)
        img.save(os.path.join(target_path, f'{coco_id}.png'))

def download_images():
    print('Downloading images...')
    stim_info = get_stim_info()
    cpus = multiprocessing.cpu_count()
    stim_info_chunks = np.array_split(stim_info, cpus)
    with multiprocessing.Pool(cpus) as p:
        p.map(download_images_chunk, stim_info_chunks)

def load_betas(subject, hemisphere, sessions, zscore_betas_within_sess=True):
    """
    Load preprocessed voxel data for an NSD subject (beta weights).
    Always loading the betas with suffix 'fithrf_GLMdenoise_RR.
    Concatenate the values across multiple sessions.
    "sessions" is zero-indexed, add one to get the actual session numbers.
    """
    
    beta_subj_path = os.path.join('nsddata_betas', 'ppdata', f'subj{subject:02d}', 'nativesurface', 'betas_fithrf_GLMdenoise_RR')   
        
    n_trials = len(sessions)*trials_per_sess

    for ss, se in tqdm(enumerate(sessions), total=len(sessions), desc=f"Loading sessions..."):

        visual_cortex_mask = get_visual_cortex_mask(subject, hemisphere)
        
        fn2load = os.path.join(beta_subj_path, f'{hemisphere}.betas_session{(se+1):02d}.hdf5')
        online_path = os.path.join(nsd_url, fn2load)
        local_path = os.path.join(temp_path, fn2load)

        if not os.path.exists(local_path):
            os.makedirs(os.path.join(temp_path, beta_subj_path), exist_ok=True)
            urllib.request.urlretrieve(online_path, local_path)

        betas = load_from_hdf5(local_path)[:, visual_cortex_mask]

        # Divide by 300 to convert back to percent signal change
        betas = betas.astype(np.float32) / 300

        if zscore_betas_within_sess: 
            mb = np.mean(betas, axis=0, keepdims=True)
            sb = np.std(betas, axis=0, keepdims=True)
            betas = np.nan_to_num((betas - mb) / (sb + 1e-6))
            del mb, sb
            gc.collect()

        if ss==0:        
            n_vox = betas.shape[1]
            betas_full = np.zeros((n_trials, n_vox), dtype=np.single)

        betas_full[ss*trials_per_sess : (ss+1)*trials_per_sess, :] = betas

        del betas
        gc.collect()
        
    return betas_full

def average_image_repetitions(voxel_data, image_order):
    """Average voxel data over repetitions of the same image."""
    n_voxels = voxel_data.shape[1]
    unique_ims = np.unique(image_order)
    avg_dat_each_image = np.zeros((len(unique_ims), n_voxels))
    for uu, im in enumerate(unique_ims):
        inds = image_order==im
        avg_dat_each_image[uu,:] = np.mean(voxel_data[inds,:], axis=0)
    return avg_dat_each_image, unique_ims

def get_data(
    subject, 
    zscore_betas_within_sess=True, 
):
    """Gather voxel data and image order for one NSD subject."""

    # Load the experiment design file that defines full image order over 30,000 trials
    image_order = get_master_image_order()
    
    # We use all sessions available for a subject
    session_inds = get_session_inds_full()
    sessions = np.arange(max_sess_each_subj[subject-1])
    inds2use = np.isin(session_inds, sessions)
    session_inds = session_inds[inds2use]
    image_order = image_order[inds2use]   

    # Now load voxel data (preprocessed beta weights for each trial)
    lh_voxel_data = load_betas(subject, 'lh', sessions, zscore_betas_within_sess)
    rh_voxel_data = load_betas(subject, 'rh', sessions, zscore_betas_within_sess)
    assert (lh_voxel_data.shape[0]==len(image_order)) and (rh_voxel_data.shape[0]==len(image_order))
    
    # Average over repetitions of same image
    avg_dat_each_image, unique_ims = average_image_repetitions(lh_voxel_data, image_order)
    lh_voxel_data = avg_dat_each_image # use average data going forward
    avg_dat_each_image, unique_ims = average_image_repetitions(rh_voxel_data, image_order)
    rh_voxel_data = avg_dat_each_image
    image_order = unique_ims # now the unique image indices become new image order

    exp_design = get_exp_design()
    stim_info = get_stim_info()
    subject_idx_MATRIX = exp_design['subjectim']
    subject_df = stim_info.loc[subject_idx_MATRIX[subject - 1, :] - 1]
    COCO_ids_unorder = np.array(subject_df["cocoId"].tolist()).astype(np.int64)
    coco_ids = COCO_ids_unorder[image_order]

    return lh_voxel_data, rh_voxel_data, coco_ids

def setup_rois(subject):
    print('Setting up ROIs...')
    target_path = os.path.join(data_path, f'subj{subject:02d}', 'roi')
    os.makedirs(target_path, exist_ok=True)

    for hemisphere in ['lh', 'rh']:
        mask = get_visual_cortex_mask(subject, hemisphere)
        for roi, (roi_string, roi_number) in rois.items():

            roi_path = os.path.join('nsddata', 'freesurfer', f'subj{subject:02d}', 'label')
            mask_string = f'{hemisphere}.{roi_string}.mgz'
            local_path = os.path.join(temp_path, roi_path, mask_string)
            online_path = os.path.join(nsd_url, roi_path, mask_string)
            if not os.path.exists(local_path):
                os.makedirs(os.path.join(temp_path, roi_path), exist_ok=True)
                urllib.request.urlretrieve(online_path, local_path)
            roi_idx = nib.load(local_path).get_fdata()
            roi_mask = roi_idx.squeeze() == roi_number
            if not np.any(roi_mask):
                continue
            roi_mask = roi_mask.astype(bool)
            roi_mask = roi_mask[mask]
            np.save(os.path.join(target_path, f'{hemisphere}.{roi}_mask.npy'), roi_mask)

            if 'floc' in roi_string:
                if 'words' in roi_string: roi_string = 'flocword'
                tval_string = f"{hemisphere}.{roi_string.replace('-','')}tval.mgz"
                local_path = os.path.join(temp_path, roi_path, tval_string)
                online_path = os.path.join(nsd_url, roi_path, tval_string)
                if not os.path.exists(local_path):
                    os.makedirs(os.path.join(temp_path, roi_path), exist_ok=True)
                    urllib.request.urlretrieve(online_path, local_path)
                tvals = nib.load(local_path).get_fdata()
                tvals = tvals.squeeze()[mask]
                np.save(os.path.join(data_path, f'subj{subject:02d}', 'roi', f'{hemisphere}.{roi}_tval.npy'), tvals)

def setup_transform(subject):
    print('Setting up transforms...')
    target_path = os.path.join(data_path, f'subj{subject:02d}', 'transform')
    os.makedirs(target_path, exist_ok=True)
    for hemisphere in ['lh', 'rh']:
        mask = get_visual_cortex_mask(subject, hemisphere)
        transform_path = os.path.join('nsddata', 'ppdata', f'subj{subject:02d}', 'transforms')
        transform_string = f'{hemisphere}.white-to-fsaverage.mgz'
        local_path = os.path.join(temp_path, transform_path, transform_string)
        online_path = os.path.join(nsd_url, transform_path, transform_string)
        if not os.path.exists(local_path):
            os.makedirs(os.path.join(temp_path, transform_path), exist_ok=True)
            urllib.request.urlretrieve(online_path, local_path)
        transform = nib.load(local_path).get_fdata().squeeze()
        old_indices = np.arange(len(mask))[mask]
        new_indices = np.array([np.where(old_indices == i)[0][0] if i in old_indices else -1 for i in transform], dtype=int)
        np.save(os.path.join(target_path, f'{hemisphere}.fsaverage.npy'), new_indices)

def setup_data(subject):
    print('Setting up data...')
    target_path = os.path.join(data_path, f'subj{subject:02d}')
    os.makedirs(target_path, exist_ok=True)
    lh_voxel_data, rh_voxel_data, coco_ids = get_data(subject)
    np.save(os.path.join(target_path, 'lh.fmri_data.npy'), lh_voxel_data)
    np.save(os.path.join(target_path, 'rh.fmri_data.npy'), rh_voxel_data)
    np.save(os.path.join(target_path, 'coco_ids.npy'), coco_ids)

def setup_noise_ceilings(subject):
    print('Setting up noise ceiling...')
    target_path = os.path.join(data_path, f'subj{subject:02d}')
    os.makedirs(target_path, exist_ok=True)
    lh_nc = get_noise_ceiling(subject, 'lh')
    rh_nc = get_noise_ceiling(subject, 'rh')
    np.save(os.path.join(target_path, 'lh.noise_ceiling.npy'), lh_nc)
    np.save(os.path.join(target_path, 'rh.noise_ceiling.npy'), rh_nc)

def build_coco_category_search():

    stim_info = get_stim_info()
    annot_file = {
        'train2017': os.path.join(data_path, "annotations", "instances_train2017.json"),
        'val2017': os.path.join(data_path, "annotations", "instances_val2017.json"),
    }
    coco = {k: COCO(v) for k, v in annot_file.items()}

    coco_id2categories = {}
    category2coco_ids = {}

    for coco_id, coco_split, crop_box in tqdm(stim_info[['cocoId', 'cocoSplit', 'cropBox']].values, total=len(stim_info), desc=f"Building category search..."):
        
        annIds = coco[coco_split].getAnnIds(imgIds=coco_id, iscrowd=None)
        anns = coco[coco_split].loadAnns(annIds)
        
        # Initialize a mask for the whole image
        img_info = coco[coco_split].loadImgs(coco_id)[0]
        composite_mask = np.zeros((img_info['height'], img_info['width']))

        for ann in anns:
            # Generate segmentation mask for the current annotation
            mask = coco[coco_split].annToMask(ann)
            # Update the composite mask
            composite_mask = np.maximum(composite_mask, mask * ann['category_id'])

        # Crop the mask
        top, bottom, left, right = eval(crop_box)
        topCrop = int(round(composite_mask.shape[0] * top))
        bottomCrop = int(round(composite_mask.shape[0] * bottom))
        leftCrop = int(round(composite_mask.shape[1] * left))
        rightCrop = int(round(composite_mask.shape[1] * right))
        cropped_image_array = composite_mask[topCrop:composite_mask.shape[0]-bottomCrop, leftCrop:composite_mask.shape[1]-rightCrop]
        cropped_image_array = cropped_image_array.astype(np.uint8)

        cats, counts = np.unique(cropped_image_array, return_counts=True)
        categories = []
        for cat, count in zip(cats, counts):
            percent = (count / mask.size) * 100
            if cat != 0 and percent >= 0.5:
                categories.append(COCO_CLASSES[cat-1])
        coco_id2categories[coco_id] = categories
        for c in categories:
            category2coco_ids.setdefault(c, [])
            category2coco_ids[c].append(coco_id)
    
    f = os.path.join(data_path, "coco_id2categories.json")
    json.dump(coco_id2categories, open(f, "w"))
    f = os.path.join(data_path, "category2coco_ids.json")
    json.dump(category2coco_ids, open(f, "w"))

if __name__ == '__main__':

    os.makedirs(data_path, exist_ok=True)
    stim_info = get_stim_info()
    stim_info.to_csv(os.path.join(data_path, 'stim_info.csv'), index=False)
    del stim_info

    shared1000 = get_shared()
    np.save(os.path.join(data_path, 'shared1000.npy'), shared1000)
    del shared1000

    download_coco_annotation_file()
    download_images()
    build_coco_category_search()

    for subject in [1,2,3,4,5,6,7,8]:
        print(f"Setting up subject {subject}\n")
        setup_rois(subject)
        setup_transform(subject)
        setup_data(subject)
        setup_noise_ceilings(subject)
        shutil.rmtree(temp_path)
    