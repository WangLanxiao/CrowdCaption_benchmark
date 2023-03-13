# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Yehao Li
@contact: yehaoli.sysu@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F

from xmodaler.config import configurable
from xmodaler.config import kfg
from .build import PREDICTOR_REGISTRY

__all__ = ["CrowdPredictor"]

@PREDICTOR_REGISTRY.register()
class CrowdPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,   # include <BOS>/<EOS>
        dropout: float
    ):
        super(CrowdPredictor, self).__init__()
        # self.vis_emb = nn.Linear(2*hidden_size, 256)
        self.logits = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "vocab_size": cfg.MODEL.VOCAB_SIZE,
            "dropout": cfg.MODEL.PRED_DROPOUT
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        if self.dropout:
            hidden_states = self.dropout(hidden_states)   #wlx
        logits = self.logits(hidden_states)
        # logits = F.softmax(logits)   #wlx

        return { kfg.G_LOGITS: logits }