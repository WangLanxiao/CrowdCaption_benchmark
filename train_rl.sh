# RL
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_net.py --num-gpus 8 \
 --dist-url tcp://127.0.0.1:26124 \
 --config-file ./configs/crowd_scenes_caption/crowdcaption/crowdcaption_rl.yaml \
 MODEL.WEIGHTS ${subpth} \
 DECODE_STRATEGY.BEAM_SIZE 3
