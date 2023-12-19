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
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
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
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.lr_decay = cfg.SOLVER.WEIGHT_DECAY

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
        self.optimizer = self.build_optimizer(cfg, self.network)
        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer)
        self.automatic_optimization = False

        self.w_mws = cfg.MODEL.CRMLOSS.MWS_WEIGHT
        self.w_sdc = cfg.MODEL.CRMLOSS.SDC_WEIGHT 
        self.w_sdn = cfg.MODEL.CRMLOSS.SDN_WEIGHT 

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def forward(self, x):
        logits = self.network(x)
        return logits.argmax(1).squeeze()

    def training_step(self, batch_data, batch_idx):
        # optimizers
        optim = self.optimizers()

        # get input & gt_label
        labels = [x["sem_seg_gt"] for x in batch_data]
        labels = torch.stack(labels) 

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        losses_dict = self.network(batch_data)
        loss = sum(losses_dict[0].values())
        loss += self.w_mws*sum(losses_dict[1].values()) # RGB
        loss += self.w_mws*sum(losses_dict[2].values()) # THR 

        loss += sum(losses_dict[3].values()) # Masked RGB-T
        loss += self.w_sdc*losses_dict[4] # self-distillation for complementary representation  
        loss += self.w_sdn*losses_dict[5] # self-distillation for non-local representation 

        # optimize network
        optim.zero_grad()
        self.manual_backward(loss)
        optim.step()

        # log
        self.log('train/total_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def validation_step(self, batch_data, batch_idx):
        # get input & gt_label
        images = [x["image"] for x in batch_data]
        labels = [x["sem_seg_gt"] for x in batch_data]

        # tensorboard logger
        logger = self.logger.experiment

        # get network output
        logits = self.network(batch_data)
        logits = [x["sem_seg"] for x in logits]

        images = torch.stack(images) 
        labels = torch.stack(labels) 
        logits = torch.stack(logits) 

        # evaluate performance
        pred  = logits.argmax(1).squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
        label = labels.squeeze().flatten()

        # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
        label_list = [x for x in range(self.num_classes)]
        conf = confusion_matrix(y_true=label.cpu().detach(), y_pred=pred.cpu().detach(), labels=label_list)

        if batch_idx%100 == 0 :
            rgb_vis  = (images[:,:3,...]).type(torch.uint8); thr_vis = (images[:,[-1],...]).type(torch.uint8); 
            scale    = max(1, 255//self.num_classes) # label (0,1,2..) is invisable, multiply a constant for visualization
            gt_label = labels.unsqueeze(1)*scale; pred_ = logits.argmax(1).unsqueeze(1)*scale

            thr_vis  = torch.cat((thr_vis, thr_vis, thr_vis), 1) 
            gt_vis   = torch.cat((gt_label, gt_label, gt_label), 1)  
            pred_vis = torch.cat((pred_, pred_, pred_),1) 

            input_rgb_grid  = vutils.make_grid(rgb_vis,  nrow=8, padding=10, pad_value=1.0) # can only display 3-channel images, so images[:,:3]
            input_thr_grid  = vutils.make_grid(thr_vis,  nrow=8, padding=10, pad_value=1.0) 
            gt_label_grid   = vutils.make_grid(gt_vis,   nrow=8, padding=10, pad_value=1.0) 
            prediction_grid = vutils.make_grid(pred_vis, nrow=8, padding=10, pad_value=1.0) 
            result_grid = torch.cat((input_rgb_grid, input_thr_grid, gt_label_grid, prediction_grid), dim=1)

            self.logger.experiment.add_image('val/result', result_grid.type(torch.float32)/255., self.global_step)

        return torch.tensor(conf) 

    def validation_epoch_end(self, validation_step_outputs):
        conf_total = torch.stack(validation_step_outputs).sum(dim=0)
        precision, recall, IoU = compute_results(conf_total)

        logger = self.logger.experiment
        self.log('val/average_precision',precision.mean())
        self.log('val/average_recall', recall.mean())
        self.log('val/average_IoU', IoU.mean(), prog_bar=True)

        for i in range(len(precision)):
            self.log("val(class)/precision_class_%s" % self.label_list[i], precision[i])
            self.log("val(class)/recall_class_%s"% self.label_list[i], recall[i])
            self.log('val(class)/Iou_%s'% self.label_list[i], IoU[i])

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
        # Here we just reuse the validation_epoch_end for testing
        return self.validation_epoch_end(test_step_outputs)