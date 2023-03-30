# coding:utf-8
# Written by Ukcheol Shin, Jan. 24, 2023
# Email: shinwc159@gmail.com

import os
import copy
import itertools

import torch
import torch.nn as nn 
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from sklearn.metrics import confusion_matrix
import torchvision.utils as vutils

from util.util import compute_results, get_palette_MF, get_palette_PST, get_palette_KP, visualize_pred
from .registry import MODELS
from models.mask2former import RGBTMaskFormer
import cv2
import numpy as np

@MODELS.register_module(name='RGBTMaskFormer')
class Model_RGBT_Mask2Former(LightningModule):
    def __init__(self, cfg):
        super(Model_RGBT_Mask2Former, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        # Set our init args as class attributes
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        if self.num_classes == 9 : 
            self.label_list = ["unlabeled", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]
            self.palette = get_palette_MF()
        elif self.num_classes == 5 : 
            self.label_list = ["unlabeled", "fire_extinhuisher", "backpack", "hand_drill", "rescue_randy"]
            self.palette = get_palette_PST()
        else:
            self.label_list = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light", "traffic_sign",
                        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]#, "unlabeled"]
            self.palette = get_palette_KP()

        self.network = RGBTMaskFormer(cfg)

    def test_step(self, batch_data, batch_idx):
        images = [x["image"] for x in batch_data]
        labels = [x["sem_seg_gt"] for x in batch_data]

        # get network output
        logits = self.network(batch_data)
        logits = [x["sem_seg"] for x in logits]

        images = torch.stack(images) 
        labels = torch.stack(labels) 
        logits = torch.stack(logits) 

        # evaluate performance
        pred  = logits.argmax(1).squeeze().flatten() 
        label = labels.squeeze().flatten()

        # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
        label_list = [x for x in range(self.num_classes)]
        conf = confusion_matrix(y_true=label.cpu().detach(), y_pred=pred.cpu().detach(), labels=label_list)

        # save the results
        pred_vis  = visualize_pred(self.palette, logits.argmax(1).squeeze().detach().cpu())
        png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, self.cfg.SAVE.DIR_NAME, "{:05}.png".format(batch_idx))
        cv2.imwrite(png_path, cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR))

        if self.cfg.SAVE.FLAG_VIS_GT:
            # denormalize input images
            images = images.squeeze().detach().cpu().numpy().transpose(1,2,0)
            rgb_vis = images[:,:,:3].astype(np.uint8)
            thr_vis = np.repeat(images[:,:,[-1]], 3, axis=2).astype(np.uint8)
            label_vis = visualize_pred(self.palette, labels.squeeze().detach().cpu())

            # if thermal image has low visibility, use the 3 lines for better visibility.
            # vmax = np.percentile(thr_vis, 99.5)
            # vmin = np.percentile(thr_vis, 0.1)
            # thr_vis = np.clip((thr_vis - vmin) / (vmax-vmin), a_min=0.0, a_max=1.0)*255

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "rgb", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "thr", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(thr_vis, cv2.COLOR_RGB2BGR))

            png_path = os.path.join(self.cfg.SAVE.DIR_ROOT, "gt", "{:05}.png".format(batch_idx))
            cv2.imwrite(png_path, cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))

        return torch.tensor(conf) 

    def test_epoch_end(self, test_step_outputs):
        conf_total = torch.stack(test_step_outputs).sum(dim=0)
        precision, recall, IoU = compute_results(conf_total)

        logger = self.logger.experiment
        self.log('val/average_precision',precision.mean())
        self.log('val/average_recall', recall.mean())
        self.log('val/average_IoU', IoU.mean(), prog_bar=True)

        for i in range(len(precision)):
            self.log("val(class)/precision_class_%s" % self.label_list[i], precision[i])
            self.log("val(class)/recall_class_%s"% self.label_list[i], recall[i])
            self.log('val(class)/Iou_%s'% self.label_list[i], IoU[i])
