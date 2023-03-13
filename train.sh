# XE
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_net.py --num-gpus 8 \
--resume --dist-url tcp://127.0.0.1:26124 \
--config-file ./configs/crowd_scenes_caption/crowdcaption/crowdcaption.yaml \
OUTPUT_DIR  ./work_dirs/crowdcaption

