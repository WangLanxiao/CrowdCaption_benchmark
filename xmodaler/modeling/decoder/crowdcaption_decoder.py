
import torch
from torch import nn
import torch.nn.functional as F
from xmodaler.config import configurable
from xmodaler.config import CfgNode as CN
from xmodaler.config import kfg
from .build import DECODER_REGISTRY
from xmodaler.modeling.layers import get_act_layer
from .decoder import Decoder


__all__ = ["Crowdcaption_decoder"]


class crowd_guided(nn.Module):
    def __init__(self, embed_dim,num_heads,act_type,elu_alpha):
        super(crowd_guided, self).__init__()
        self.num_heads=num_heads
        self.drop = nn.Dropout(0.5)
        self.head_dim=embed_dim//num_heads
        output_dim = 2 * embed_dim if act_type == 'GLU' else embed_dim

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.query2 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.query1 = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.key = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(embed_dim, output_dim))
        act = get_act_layer(act_type)(elu_alpha)
        if act is not None:
            sequential.append(act)
        sequential.append(torch.nn.GroupNorm(self.num_heads, embed_dim))
        self.value = nn.Sequential(*sequential)

        sequential = []
        sequential.append(nn.Linear(128, 64))
        sequential.append(nn.ReLU())
        sequential.append(nn.Dropout(0.1))
        self.map = nn.Sequential(*sequential)
        self.spactial = nn.Linear(64, 1)
        self.channel = nn.Linear(64, 128)

        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, hidden, visual):
        # 96*1024     96*37*1024
        batch_size = visual.size()[0]
        query1 = visual.view(-1, visual.size()[-1])
        value = visual.view(-1, visual.size()[-1])

        # 96*8*37**128    96*8*37**128
        query1 = self.drop(self.query1(query1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2))
        value = self.drop(self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2))
        # 96*8*128    96*8*128
        query2 = self.drop(self.query2(hidden).view(batch_size, self.num_heads, self.head_dim))
        key = self.drop(self.key(hidden).view(batch_size, self.num_heads, self.head_dim))

        attn_scoremap = query2.unsqueeze(-2) * query1
        attn_scoreweights = self.map(attn_scoremap)
        attn_scoreweights_pool = attn_scoreweights.mean(-2)

        alpha_spatial = self.spactial(attn_scoreweights).squeeze(-1)
        alpha_spatial = F.softmax(alpha_spatial, dim=-1)  #96*8*37
        alpha_channel = self.channel(attn_scoreweights_pool)
        alpha_channel = torch.sigmoid(alpha_channel)  #96*8*128
        value = torch.matmul(alpha_spatial.unsqueeze(-2), value).squeeze(-2)   # 96*8*1*37    96*8*37*128  =====> 96*8*128
        att = (key * value * alpha_channel).view(batch_size, self.num_heads * self.head_dim)
        att = self.drop(att)
        att = self.proj(torch.cat([att, hidden], dim=-1))
        att = self.layer_norm(att)
        return att

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
        v = self.wv(feats)
        inputs = self.wh(key).expand_as(v) + v
        alpha = F.softmax(self.wa(torch.tanh(inputs)).squeeze(-1), dim=1)
        att_feats = torch.bmm(alpha.unsqueeze(1), feats)
        return att_feats


@DECODER_REGISTRY.register()
class Crowdcaption_decoder(Decoder):
    @configurable
    def __init__(
            self,
            *,
            hidden_size: int,
            ctx_drop: float,
            bilinear_dim: int,
            att_heads: int,
            act_type: str,
            elu_alpha: float
    ):
        super(Crowdcaption_decoder, self).__init__()

        self.num_layers = 2
        self.hidden_size = hidden_size

        # First LSTM layer
        rnn_input_size = hidden_size + bilinear_dim
        self.att_lstm = nn.LSTMCell(rnn_input_size, hidden_size)
        self.ctx_drop = nn.Dropout(ctx_drop)

        self.attention = crowd_guided(
            embed_dim=bilinear_dim,
            num_heads=att_heads,
            act_type=act_type,
            elu_alpha=elu_alpha
        )
        self.att2ctx = nn.Sequential(
            nn.Linear(bilinear_dim + hidden_size + hidden_size, 2 * hidden_size),
            nn.GLU()
        )

        self.combine_att = SpacialAttention(feat_size=hidden_size, hidden_size=hidden_size, att_size=hidden_size)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.DECODER_DIM,
            "ctx_drop": cfg.MODEL.PRED_DROPOUT,
            "bilinear_dim": cfg.MODEL.DECODER_DIM,
            "att_heads": 8,
            "act_type": "celu",
            "elu_alpha": 1.3
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def preprocess(self, batched_inputs):
        att_feats = batched_inputs[kfg.ATT_FEATS]
        init_states = self.init_states(att_feats.shape[0])
        batched_inputs.update(init_states)
        return batched_inputs

    def forward(self, batched_inputs):
        wt = batched_inputs[kfg.G_TOKEN_EMBED]   # 96 * 1024
        att_feats = batched_inputs[kfg.ATT_FEATS]    # 96 * 37 * 1024
        # att_masks = batched_inputs[kfg.ATT_MASKS]     # 96 * 37
        gv_feat = batched_inputs[kfg.GLOBAL_FEATS]  # 96 * 1024
        hidden_states = batched_inputs[kfg.G_HIDDEN_STATES]  # 96 * 1024   # list of tensors
        cell_states = batched_inputs[kfg.G_CELL_STATES]  # 96 * 1024

        class_att_feats = batched_inputs[kfg.CLASS_ATT_FEATS]    # 96 * 37 * 2048
        dim = class_att_feats.size()[-1]
        class_object_feat = class_att_feats.narrow(-1, 0, dim // 3)
        class_behavior_feat = class_att_feats.narrow(-1, dim // 3, dim // 3)
        class_attribution_feat = class_att_feats.narrow(-1, 2 * dim // 3, dim // 3)

        combine_feat=torch.cat([gv_feat,class_object_feat,class_behavior_feat,class_attribution_feat],1)
        final_feat=self.combine_att(hidden_states[0].unsqueeze(1),combine_feat)*0.5+gv_feat
        h_att, c_att = self.att_lstm(torch.cat([wt, final_feat.squeeze(1) + self.ctx_drop(hidden_states[1])], 1),
                                     (hidden_states[0], cell_states[0]))
        att = self.attention(h_att, att_feats)
        ctx_input = torch.cat([att, h_att,final_feat.squeeze(1)], 1)
        output = self.att2ctx(ctx_input)
        hidden_states = [h_att, output]
        cell_states = [c_att]

        return {
            kfg.G_HIDDEN_STATES: hidden_states,
            kfg.G_CELL_STATES: cell_states
        }