# Written by Ukcheol Shin, Jan. 24, 2023
# Email: shinwc159@gmail.com

import os.path as osp
from argparse import ArgumentParser

from mmcv import Config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader

from models import MODELS
from dataloaders import build_dataset
from util.util import make_save_dir

# MaskFormer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.projects.deeplab import add_deeplab_config
from models.mask2former import add_maskformer2_config
from models.config import add_CRM_config
from util.RGBTCheckpointer import RGBTCheckpointer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--num-gpus', type=int) # number of gpus
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--name', type=str)
    parser.add_argument('--work_dir', type=str, default='checkpoints')
    return parser.parse_args()
 
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

def my_collate_fn(batch_dict):
    return batch_dict

if __name__ == '__main__':
    # parse args
    args = parse_args()
    cfg = setup(args)
    print('Now evaluating with {}...'.format(osp.basename(args.config_file)))
    
    # device
    device = torch.device('cuda:0')

    # prepare data loader
    dataset = build_dataset(cfg)
    test_loader   = DataLoader(dataset['test'], 1, shuffle=False, num_workers=cfg.DATASETS.WORKERS_PER_GPU, drop_last=False, collate_fn=my_collate_fn)

    # model
    model = MODELS.build(name=cfg.MODEL.META_ARCHITECTURE, option=cfg)
    # model.network.load_state_dict(torch.load(args.checkpoint)['model'])# for pre-trained weight of original RTF and MFNet 
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.to(device)
    model.eval()
    print('Successfully load weights from check point {}.'.format(args.checkpoint))

    # make save directory
    make_save_dir(path_root=cfg.SAVE.DIR_ROOT, pred_name=cfg.SAVE.DIR_NAME)

    # define trainer
    work_dir = osp.join(args.work_dir, args.name)
    trainer = Trainer(default_root_dir=work_dir,
                      gpus=args.num_gpus,
                      num_nodes=1)
                      # precision=16)

    # testing
    trainer.test(model, test_loader)
