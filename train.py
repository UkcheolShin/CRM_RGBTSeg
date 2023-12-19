# Written by Ukcheol Shin, Jan. 24, 2023
# Email: shinwc159@gmail.com

import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch

from models import MODELS
from dataloaders import build_dataset

# MaskFormer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from models.mask2former import add_maskformer2_config
from models.config import add_CRM_config
from util.RGBTCheckpointer import RGBTCheckpointer

def parse_args():
    parser = ArgumentParser(description='Training with DDP.')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument('--work_dir',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--name',
                        type=str)
    parser.add_argument('--seed',
                        type=int,
                        default=1024)
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()
    return args

def my_collate_fn(batch_dict):
    return batch_dict

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_CRM_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg

def main():
    # parse args
    args = parse_args()
    cfg  = setup(args)
    print(f'Now training with {args.config_file}...')

    # configure seed
    seed_everything(args.seed)

    # prepare data loader
    dataset = build_dataset(cfg)
    train_loader = DataLoader(dataset['train'], cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=cfg.DATASETS.WORKERS_PER_GPU, drop_last=True, collate_fn=my_collate_fn)
    val_loader   = DataLoader(dataset['test'], cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=cfg.DATASETS.WORKERS_PER_GPU, drop_last=True, collate_fn=my_collate_fn)

    # define model
    model = MODELS.build(name=cfg.MODEL.META_ARCHITECTURE, option=cfg)
    if cfg.MODEL.META_ARCHITECTURE in {"RGBTMaskFormer"}:
        RGBTCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)['state_dict'])

    # define trainer
    work_dir = osp.join(args.work_dir, args.name)
    checkpoint_callback = ModelCheckpoint(dirpath=work_dir,
                                          save_weights_only=True,
                                          monitor='val/average_IoU',
                                          mode='max',
                                          save_top_k=1,
                                          filename='checkpoint_{epoch:02d}_{step}')

    trainer = Trainer(strategy='ddp' if args.num_gpus > 1 else None,
                      default_root_dir=work_dir,
                      gpus=args.num_gpus,
                      num_nodes=1,
                      # max_epochs=cfg.SOLVER.total_epochs,
                      max_steps=cfg.SOLVER.MAX_ITER,
                      callbacks=[checkpoint_callback],
                      check_val_every_n_epoch=5)
                      # precision=16)

    # training
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()