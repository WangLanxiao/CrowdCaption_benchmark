# Copyright 2021 JD.com, Inc., JD AI
"""
@author: Jianjie Luo, Jingwen Chen
@contact: jianjieluo.sysu@gmail.com, chenjingwen.sysu@gmail.com
"""
import torch
from torch import nn
from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import ENCODER_REGISTRY
import torch.nn.functional as F

__all__ = ["Crowdcaption_encoder"]


# ------------------------------------------------------
# ------------ Soft Attention Mechanism ----------------
# ------------------------------------------------------
class SpacialAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(SpacialAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        nn.init.xavier_normal_(self.wh.weight)
        nn.init.xavier_normal_(self.wv.weight)
        nn.init.xavier_normal_(self.wa.weight)

    def forward(self, key, feats):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        # feats_pool = self.pool(feats.permute(0,2,1)).permute(0,2,1)
        v = self.wv(feats)
        inputs = self.wh(key).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats)
        return att_feats

class ChannelAttention(nn.Module):
    def __init__(self, feat_size, hidden_size, att_size):
        super(ChannelAttention, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.wh = nn.Linear(hidden_size, att_size)
        self.wv = nn.Linear(feat_size, att_size)
        self.wa = nn.Linear(att_size, hidden_size, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(1)
        nn.init.xavier_normal_(self.wh.weight)
        nn.init.xavier_normal_(self.wv.weight)
        nn.init.xavier_normal_(self.wa.weight)

    def forward(self, key, feats):
        '''
        :param feats: (batch_size, feat_num, feat_size)
        :param key: (batch_size, hidden_size)
        :return: att_feats: (batch_size, feat_size)
                 alpha: (batch_size, feat_num)
        '''
        key_pool = self.pool(key.permute(0,2,1)).permute(0,2,1)
        v = self.wv(feats)
        inputs = self.wh(key_pool).expand_as(v) + v
        alpha = torch.sigmoid(self.wa(torch.tanh(inputs)))
        att_feats = alpha * feats
        return att_feats

@ENCODER_REGISTRY.register()
class Crowdcaption_encoder(nn.Module):
    @configurable
    def __init__(
            self,
            *,
            hidden_size: int,
            embed_dim: int,
            class_attribution,
            class_behavior,
            class_object
    ):
        super(Crowdcaption_encoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.level0_s = SpacialAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level0_c = ChannelAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level0_f = nn.Linear(embed_dim * 2, embed_dim)

        self.level1_s = SpacialAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level1_c = ChannelAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level1_f = nn.Linear(embed_dim * 2, embed_dim)

        self.level2_s = SpacialAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level2_c = ChannelAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level2_f = nn.Linear(embed_dim * 2, embed_dim)

        self.level3_s = SpacialAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level3_c = ChannelAttention(feat_size=embed_dim, hidden_size=embed_dim, att_size=embed_dim)
        self.level3_f = nn.Linear(embed_dim * 2, embed_dim)

        self.class_fc1 = nn.Linear(embed_dim, class_object)
        self.class_fc2 = nn.Linear(embed_dim, class_behavior)
        self.class_fc3 = nn.Linear(embed_dim, class_attribution)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)
    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "embed_dim": cfg.MODEL.ENCODER_DIM,
            "class_attribution": cfg.MODEL.ATTRIBUTION_VOCAB_SIZE,
            "class_behavior": cfg.MODEL.BEHAVIOR_VOCAB_SIZE,
            "class_object": cfg.MODEL.OBJECT_VOCAB_SIZE
        }

    @classmethod
    def add_config(cls, cfg):
        pass


    def forward(self, batched_inputs, mode=None):
        ret = {}
        if mode == None or mode == 'v':
            att_mask = batched_inputs[kfg.ATT_MASKS]
            att_feats = batched_inputs[kfg.ATT_FEATS] # B * 37 * 1024
            batch_size = att_feats.size()[0]
            gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1) # B * 1 * 1024
            hrnet_feat = batched_inputs[kfg.HRNET_FEATS] # B * 34 * 1024
            # box_feat = batched_inputs[kfg.ATT_FEATS_LOC] # B * 37 * 1024

            global_feat_0 = self.level0_s(gv_feat.unsqueeze(1) * att_feats, att_feats)
            global_feat = self.level0_c(gv_feat.unsqueeze(1) * att_feats, global_feat_0)+gv_feat.unsqueeze(1)
            global_feat = self.proj(global_feat)
            global_feat = self.layer_norm(global_feat)
            att_feats_cat = torch.cat([global_feat.expand_as(att_feats), att_feats], dim=-1)
            att_feats = self.level0_f(att_feats_cat) + att_feats
            att_feats = self.layer_norm(att_feats)

            f1 = self.level1_s(global_feat * att_feats , att_feats)
            f1 = self.level1_c(global_feat * att_feats , f1)
            f1_cat = torch.cat([f1, global_feat], dim=-1)
            f1_final = self.level1_f(f1_cat) + global_feat
            class_object_feat = self.layer_norm(f1_final)       # B * 1 * 1024

            f2 = self.level2_s(class_object_feat * hrnet_feat, hrnet_feat)
            f2 = self.level2_c(class_object_feat * hrnet_feat, f2)
            f2_cat = torch.cat([f2, global_feat], dim=-1)
            f2_final = self.level2_f(f2_cat) + global_feat
            class_behavior_feat = self.layer_norm(f2_final)  # B * 1 * 1024

            f3 = self.level3_s(class_object_feat * class_behavior_feat, att_feats)
            f3 = self.level3_c(class_object_feat * class_behavior_feat, f3)
            f3_cat = torch.cat([f3, global_feat], dim=-1)
            f3_final = self.level3_f(f3_cat) + global_feat
            class_attribution_feat = self.layer_norm(f3_final)  # B * 1 * 1024

            class_att_feats = torch.cat([class_object_feat, class_behavior_feat, class_attribution_feat], dim=-1)

            results1 = torch.sigmoid(self.class_fc1(class_object_feat.squeeze(1)))
            results2 = torch.sigmoid(self.class_fc2(class_behavior_feat.squeeze(1)))
            results3 = torch.sigmoid(self.class_fc3(class_attribution_feat.squeeze(1)))
            results = [results1, results2, results3]
            # results=None

            ret.update({
                kfg.G_HIDDEN_STATES: [global_feat.squeeze(1),global_feat.squeeze(1)],
                kfg.G_CELL_STATES: [global_feat.squeeze(1)],
                kfg.CLASS_ATT_FEATS: class_att_feats,
                kfg.ATT_FEATS: att_feats,
                kfg.GLOBAL_FEATS: global_feat,
                kfg.CLASS_RESULTS: results
            })
        return ret
