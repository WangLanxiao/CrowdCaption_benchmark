"""
From original at https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/__init__.py
Original copyright of Facebook code below, modifications by Yehao Li, Copyright 2021.	
"""

# Copyright (c) Facebook, Inc. and its affiliates.
import warnings
warnings.filterwarnings('ignore')
import logging
import os
from collections import OrderedDict
import torch
import xmodaler.utils.comm as comm
from xmodaler.checkpoint import XmodalerCheckpointer
from xmodaler.config import get_cfg
from xmodaler.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, build_engine
from xmodaler.modeling import add_config
import csv
import shutil


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    print("Command Line Args:", args)
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = build_engine(cfg)
    trainer.resume_or_load(resume=args.resume)
    print(trainer.model)

    if args.eval_only:
        res = None
        epoch_id = cfg['OUTPUT_DIR'].split('_')[-1]
        #######val part
        if trainer.val_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.val_data_loader, trainer.val_evaluator, epoch=-1 , mode = 'val')
        if comm.is_main_process():
            print(res)
        csv_writer.writerow([epoch_id, version_csv, "val", res])
        #######test part
        if trainer.test_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader, trainer.test_evaluator, epoch=-1, mode = 'test')
        if comm.is_main_process():
            print(res)
        return res
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    print(cfg.OUTPUT_DIR)