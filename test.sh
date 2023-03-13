#!/usr/bin/env bash
#######  eval special epoch    ##########
CUDA_VISIBLE_DEVICES=0
python ./train_net.py --num-gpus 1 \
 --config-file ./configs/crowd_scenes_caption/crowdcaption/crowdcaption.yaml \
 --eval-only \
 MODEL.WEIGHTS ${subpth} \
 DECODE_STRATEGY.BEAM_SIZE 3

