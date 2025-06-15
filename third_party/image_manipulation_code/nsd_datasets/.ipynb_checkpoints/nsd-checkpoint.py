import os
import pathlib

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

from methods.subsets import get_subset_indices


class NaturalScenesDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        subject: int,
        partition: str,
        transform: transforms.Compose = None,
        hemisphere: str = None,
        roi: str = None,
        tval_threshold: float = 2.0,
        return_average: bool = False,
        subset: str = None,
    ):
        super().__init__()
        assert partition in ["train", "test", "all"]
        assert hemisphere in [None, "lh", "rh", "both"]
        assert subject in range(1, 9)
        if roi is not None: assert hemisphere is not None
        self.return_activations = roi is not None

        self.root = pathlib.Path(root)
        self.subject = subject
        self.partition = partition
        self.transform = transform
        self.return_average = return_average
        self.subj_dir = os.path.join(self.root, f"subj{self.subject:02d}")

        self.coco_ids = np.load(os.path.join(self.subj_dir, 'coco_ids.npy'))
        partition_mask = self.load_partition_mask()
        subset_mask = self.load_subset_mask(subset)
        self.coco_id_mask = partition_mask & subset_mask
        self.coco_ids = self.coco_ids[self.coco_id_mask]

        if self.return_activations:
            if hemisphere == 'both':
                self.activations = np.concatenate(
                    [self.load_activations(roi, 'lh', tval_threshold),
                     self.load_activations(roi, 'rh', tval_threshold)],
                    axis=1
                )
            else:
                if isinstance(roi, list):
                    activation_list = []
                    for region in roi:
                        region_activation = self.load_activations(region, hemisphere, tval_threshold)
                        if return_average:
                            region_activation = region_activation.mean(axis = 1).reshape(-1, 1)
                        activation_list.append(region_activation)
                    # Concatenate activations along the feature axis (axis=1)
                    self.activations = np.concatenate(activation_list, axis=1)
                else:
                    self.activations = self.load_activations(roi, hemisphere, tval_threshold)
                    if return_average:
                        self.activations = self.activations.mean(axis=1)

    def __len__(self):
        return len(self.coco_ids)

    def __getitem__(self, idx):
        coco_id = self.coco_ids[idx]
        img = Image.open(os.path.join(self.root, 'images', f'{coco_id}.png')).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.return_activations:
            activation = self.activations[idx]
            return img, activation
        return img
    
    def load_partition_mask(self):
        shared1000 = np.load(os.path.join(self.root, 'shared1000.npy'))
        mask = np.isin(self.coco_ids, shared1000)
        if self.partition == 'train':
            mask = ~mask
        elif self.partition == 'all':
            mask = np.ones_like(mask).astype(bool)
        return mask
    
    def load_activations(self, roi, hemisphere, tval_threshold):
        activations = np.load(os.path.join(self.subj_dir, f'{hemisphere}.fmri_data.npy'))
        roi_mask = np.load(os.path.join(self.subj_dir, 'roi', f'{hemisphere}.{roi}_mask.npy'))
        tvals = np.load(os.path.join(self.subj_dir, 'roi', f'{hemisphere}.{roi}_tval.npy'))
        roi_mask = roi_mask & (tvals > tval_threshold)
        activations = activations[self.coco_id_mask][:, roi_mask]
        return activations
    
    def load_subset_mask(self, subset):
        if subset is None:
            return np.ones_like(self.coco_ids).astype(bool)
        subset_indices = get_subset_indices(self, subset)
        subset_mask = [idx in subset_indices for idx in self.coco_ids]
        return subset_mask
