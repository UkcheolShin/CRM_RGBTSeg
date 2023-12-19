# Modified by Ukcheol Shin, Jan. 24, 2023 from the following two repositories.
# SimMIM: https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py
# Mask2Former: https://github.com/facebookresearch/Mask2Former

import numpy as np
from PIL import Image
import torch
import random
import cv2

class MaskGenerator:
    def __init__(self, input_size=[256, 320], mask_patch_size=32, model_patch_size=4, \
                 mask_ratio=0.6, mask_type='patch', strategy='comp'):
        self.input_size = np.array(input_size)
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size[0] % self.mask_patch_size == 0
        assert self.input_size[1] % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        if mask_type == 'patch':
            self.gen_mask = self.gen_patch_mask
        elif mask_type == 'square':
            self.gen_mask = self.gen_square_mask
        else:
            raise AssertionError("Not valid mask type!")

        if strategy == 'comp':
            self.strategy = self.gen_comp_masks
        elif strategy == 'rand_comp':
            self.strategy = self.gen_rand_comp_masks
        elif strategy == 'indiv':
            self.strategy = self.gen_indiv_masks
        else:
            raise AssertionError("Not valid strategy!")
 
    def gen_patch_mask(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = np.expand_dims(mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1), axis=0)
        return mask

    def gen_square_mask(self):
        mask = np.zeros((self.input_size[0], self.input_size[1]), dtype=int)

        h1 = np.random.randint(0, self.input_size[0]*self.mask_ratio)
        w1 = np.random.randint(0, self.input_size[1]*self.mask_ratio)
        h2 = int(h1 + self.input_size[0]*self.mask_ratio)
        w2 = int(w1 + self.input_size[1]*self.mask_ratio)

        mask[w1:w2, h1:h2] = 1
        return np.expand_dims(mask, axis=0)

    def gen_comp_masks(self):
        mask = self.gen_mask()
        return mask, 1-mask

    def gen_rand_comp_masks(self):
        mask = self.gen_mask()
        nomask = np.zeros_like(mask)

        idx = random.randrange(3)
        if idx == 0:   return nomask, 1-mask
        elif idx == 1: return mask, nomask
        elif idx == 2: return mask, 1-mask

    def gen_indiv_masks(self):
        mask1 = self.gen_mask()
        mask2 = self.gen_mask()
        return mask1, mask2

    def __call__(self):
        return self.strategy()

# Modified from Mask2former ColorAugSSDTransform function to take 4 channel input
from fvcore.transforms.transform import Transform
class ColorAugSSDTransform(Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB", "RGBT"]
        self.is_rgb = img_format == "RGB"
        self.is_rgbt = img_format == "RGBT"
        del img_format
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgbt:
            img_r = img[:, :, [2, 1, 0]]
            img_t = img[:, :, -1]
            img_r = self.brightness(img_r)
            if random.randrange(2):
                img_r = self.contrast(img_r)
                img_r = self.saturation(img_r)
                img_r = self.hue(img_r)
                img_t = self.contrast(img_t)
            else:
                img_r = self.saturation(img_r)
                img_r = self.hue(img_r)
                img_r = self.contrast(img_r)
                img_t = self.contrast(img_t)
            img[:,:,:3]  = img_r[:, :, [2, 1, 0]]
            img[:,:,-1] = img_t
        else:
            if self.is_rgb:
                img = img[:, :, [2, 1, 0]]
            img = self.brightness(img)
            if random.randrange(2):
                img = self.contrast(img)
                img = self.saturation(img)
                img = self.hue(img)
            else:
                img = self.saturation(img)
                img = self.hue(img)
                img = self.contrast(img)
            if self.is_rgb:
                img = img[:, :, [2, 1, 0]]
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randrange(2):
            return self.convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta)
            )
        return img

    def contrast(self, img):
        if random.randrange(2):
            return self.convert(img, alpha=random.uniform(self.contrast_low, self.contrast_high))
        return img

    def saturation(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=random.uniform(self.saturation_low, self.saturation_high)
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randrange(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img
