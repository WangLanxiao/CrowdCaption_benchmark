# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
import torch.nn as nn
from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class Crowd_BCEWithLogits(nn.Module):
    @configurable
    def __init__(self,alpha_object,alpha_action,alpha_status):
        super(Crowd_BCEWithLogits, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.alpha_object = alpha_object
        self.alpha_action = alpha_action
        self.alpha_status = alpha_status

    @classmethod
    def from_config(cls, cfg):
        return {
            "alpha_object": cfg.LOSSES.AERFA_OBJECT,
            "alpha_action": cfg.LOSSES.AERFA_ACTION,
            "alpha_status": cfg.LOSSES.AERFA_STATUS,
        }

    def forward(self, outputs_dict):
        ret  = {}
        targets1 = outputs_dict[kfg.G_OBJECT_IDS]
        targets2 = outputs_dict[kfg.G_BEHAVIOR_IDS]
        targets3 = outputs_dict[kfg.G_ATTRIBUTION_IDS]
        out1 = outputs_dict[kfg.CLASS_RESULTS][0]
        out2 = outputs_dict[kfg.CLASS_RESULTS][1]
        out3 = outputs_dict[kfg.CLASS_RESULTS][2]


        loss1 = self.criterion(out1, targets1.float()) * targets1.size(1)
        ret.update({ 'BCEWithLogits Loss OBJECT': loss1 * self.alpha_object })

        loss2 = self.criterion(out2, targets2.float()) * targets2.size(1)
        ret.update({'BCEWithLogits Loss BEHAVIOR': loss2 * self.alpha_action })

        loss3 = self.criterion(out3, targets3.float())  * targets3.size(1)
        ret.update({'BCEWithLogits Loss ATTRIBUTION': loss3 * self.alpha_status })

            
        return ret