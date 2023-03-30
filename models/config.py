# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN

def add_CRM_config(cfg):
    """
    Add config for Complementary Random Maksing.
    """
    # data config
    cfg.MODEL.SWIN.SHARE_START_IDX = 4
    cfg.MODEL.FUSION = CN()
    cfg.MODEL.FUSION.AGGREGATION = "MAX"
    cfg.MODEL.FUSION.LAYER = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.CRMLOSS = CN()
    cfg.MODEL.CRMLOSS.MWS_WEIGHT = 1.0
    cfg.MODEL.CRMLOSS.SDC_WEIGHT = 1.0
    cfg.MODEL.CRMLOSS.SDN_WEIGHT = 1.0

    cfg.DATASETS.NAME = "MFdataset"
    cfg.DATASETS.DIR = "./datasets/MFdataset/"
    cfg.DATASETS.IMS_PER_BATCH = 8
    cfg.DATASETS.WORKERS_PER_GPU = 4

    # mask_former model config
    cfg.INPUT.MASK = CN()
    cfg.INPUT.MASK.ENABLED = True
    cfg.INPUT.MASK.SIZE = (256, 320)
    cfg.INPUT.MASK.PATCH_SIZE = 32
    cfg.INPUT.MASK.RATIO = 0.5
    cfg.INPUT.MASK.TYPE = 'patch'
    cfg.INPUT.MASK.STRATEGY = 'rand_comp'

    cfg.SAVE = CN()
    cfg.SAVE.DIR_ROOT = "./results"
    cfg.SAVE.DIR_NAME = "pred"
    cfg.SAVE.FLAG_VIS_GT = False
