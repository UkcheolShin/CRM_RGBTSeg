# Written by Ukcheol Shin, Jan. 24, 2023 using the following two repositories.
# PST900: https://github.com/ShreyasSkandanS/pst900_thermal_rgb
# Mask2Former: https://github.com/facebookresearch/Mask2Former

import cv2
import numpy as np
import os, torch
from imageio import imread
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset


class PST_dataset(Dataset):

    def __init__(self, data_dir, cfg, split):
        super(PST_dataset, self).__init__()

        assert split in ['train', 'val', 'test'], \
            'split must be "train"|"val"|"test"' 

        # read dataset list, all files have the same name across 'rgb', 'label', 'thermal', 'depth' folders
        self.data_list = os.listdir(os.path.join(data_dir, 'rgb')) 
        self.data_list.sort()

        self.data_dir  = os.path.join(data_dir, split)
        self.split     = split
        self.n_data    = len(self.data_list)
        self.size_divisibility = -1
        self.ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        
    def read_image(self, name, folder):
        file_path = os.path.join(self.data_dir, '%s/%s' % (folder, name))
        image     = imread(file_path).astype('float32')
        return image

    def __getitem__(self, index):
        name  = self.data_list[index]
        image_rgb = self.read_image(name, 'rgb')
        image_thr = np.expand_dims(self.read_image(name, 'thermal'), axis=2)
        image = np.concatenate((image_rgb,image_thr),axis=2)
        # depth = self.read_image(name, 'depth')

        sem_seg_gt = self.read_image(name, 'labels').astype("double")

        # Pad image and segmentation label here!
        image      = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image      = F.pad(image, padding_size, value=128).contiguous()
            sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Packing data
        result = {}
        result["name"]  = name
        result["image"] = image
        result["sem_seg_gt"] = sem_seg_gt.long()

        return result

    def __len__(self):
        return self.n_data
