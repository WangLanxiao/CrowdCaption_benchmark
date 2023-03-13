# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
from .build import build_losses, build_rl_losses
from .cross_entropy import CrossEntropy
from .label_smoothing import LabelSmoothing
from .reward_criterion import RewardCriterion
from .crowd_bce_logits import Crowd_BCEWithLogits
