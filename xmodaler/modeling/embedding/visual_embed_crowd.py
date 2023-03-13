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
from ..layers.create_act import get_act_layer
from .build import EMBEDDING_REGISTRY
from ..layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

__all__ = ["VisualEmbeddingCrowd"]

@EMBEDDING_REGISTRY.register()
class VisualEmbeddingCrowd(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        in_dim: int,
        out_dim: int,
        **kwargs
    ):
        super(VisualEmbeddingCrowd, self).__init__()
        self.embeddings = nn.Linear(in_dim, out_dim)
        self.embeddings_act = kwargs.pop("embeddings_act", None)
        self.embeddings_norm = kwargs.pop("embeddings_norm", None)
        self.embeddings_dropout = kwargs.pop("embeddings_dropout", None)
        self.embeddings_pos = kwargs.pop('embeddings_pos', None)

        # self.conv1 = nn.Conv2d(34, 37, kernel_size=1, stride=1, padding=0)

        self.flatten = nn.Flatten(2, 3)
        self.AvgPool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc_1 = nn.Linear(128 * 208, 2*out_dim)
        self.fc_relu_1 = nn.ReLU()
        self.fc_2 = nn.Linear(2 * out_dim, out_dim)
        self.fc_relu_2 = nn.ReLU()
        self.fc_3 = nn.Linear(out_dim, out_dim)



    @classmethod
    def from_config(cls, cfg):
        kwargs = {
            "in_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "out_dim": cfg.MODEL.VISUAL_EMBED.OUT_DIM
        }

        activation_name = (cfg.MODEL.VISUAL_EMBED.ACTIVATION).lower()
        if activation_name != "none":
            activation = get_act_layer(activation_name)
            assert activation is not None

            act_kwargs = {}
            if activation_name in { "elu", "celu" }:
                act_kwargs["alpha"] = cfg.MODEL.VISUAL_EMBED.ELU_ALPHA
            embeddings_act = activation(**act_kwargs)
            kwargs['embeddings_act'] = embeddings_act

        if cfg.MODEL.VISUAL_EMBED.DROPOUT > 0:
            embeddings_dropout = nn.Dropout(cfg.MODEL.VISUAL_EMBED.DROPOUT)
            kwargs['embeddings_dropout'] = embeddings_dropout

        if cfg.MODEL.VISUAL_EMBED.USE_NORM:
            embeddings_norm = nn.LayerNorm(cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_norm'] = embeddings_norm

        if cfg.MODEL.VISUAL_EMBED.LOCATION_SIZE > 0:
            embeddings_pos = nn.Linear(cfg.MODEL.VISUAL_EMBED.LOCATION_SIZE , cfg.MODEL.VISUAL_EMBED.OUT_DIM)
            kwargs['embeddings_pos'] = embeddings_pos

        return kwargs

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.ATT_FEATS]
        boxes = batched_inputs[kfg.ATT_FEATS_LOC] if kfg.ATT_FEATS_LOC in batched_inputs else None    #x y w h  position information

        embeddings = self.embeddings(feats)
        embeddings_pos = None
        if (self.embeddings_pos is not None) and (boxes is not None):
            embeddings_pos = self.embeddings_pos(boxes)

        if self.embeddings_act is not None:
            embeddings = self.embeddings_act(embeddings)
            if (boxes is not None):
                embeddings_pos = self.embeddings_act(embeddings_pos)

        if self.embeddings_norm is not None:
            embeddings = self.embeddings_norm(embeddings)
            if (boxes is not None):
                embeddings_pos = self.embeddings_norm(embeddings_pos)

        if self.embeddings_dropout is not None:
            embeddings = self.embeddings_dropout(embeddings)
            if (boxes is not None):
                embeddings_pos = self.embeddings_dropout(embeddings_pos)

        hrnet_feats = batched_inputs[kfg.HRNET_FEATS]
        # hrnet_feats = self.conv1(hrnet_feats)
        backbone_embed = self.flatten(hrnet_feats)
        hrnet_embeddings_l1 = self.fc_relu_1(self.fc_1(backbone_embed))
        hrnet_embeddings_l2 = self.fc_relu_2(self.fc_2(hrnet_embeddings_l1))
        hrnet_embeddings_l3 = self.embeddings_dropout(self.fc_3(hrnet_embeddings_l2))
        # if self.embeddings_norm is not None:
        #     hrnet_embeddings_l3 = self.embeddings_norm(hrnet_embeddings_l3)
        return { kfg.ATT_FEATS: embeddings, kfg.ATT_FEATS_LOC: embeddings_pos, kfg.HRNET_FEATS: hrnet_embeddings_l3}