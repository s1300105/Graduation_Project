import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv, encode_input
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import numpy as np
class BertGGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super().__init__()
        self.device = device

        # DataBatch(x=[..., 101]) のように second dim が現在の特徴次元
        feat_dim   = emb_size          # 例: 101（実データに合わせる）
        hidden_dim = gated_graph_conv_args["out_channels"]  # 例: 256

        # ① 入力をGGNNの隠れ次元へ射影
        self.input_proj = nn.Linear(feat_dim, hidden_dim)

        # ② GatedGraphConv（in_channelsは指定不可 → 入力=出力次元が前提）
        self.ggnn = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=gated_graph_conv_args["num_layers"],
            aggr=gated_graph_conv_args.get("aggr", "add"),
            bias=gated_graph_conv_args.get("bias", True),
        )

        # ③ Conv 側は hidden と元の x を結合する前提
        conv_args = dict(conv_args)
        conv_args["fc_1_size"] = hidden_dim + feat_dim
        conv_args["fc_2_size"] = hidden_dim
        self.conv = Conv(**conv_args)

        

    def forward(self, data):
        x, edge_index, text = data.x, data.edge_index, data.func
        h0 = self.input_proj(x)                 # [N, hidden_dim]
        h  = self.ggnn(h0, edge_index)          # [N, hidden_dim]
        p  = self.conv(h, x)                    # [B] in (0,1)  ← Conv は確率を返す

        # CrossEntropyLoss 用 2クラスlogitsに変換
        eps = 1e-6
        p = p.clamp(eps, 1 - eps)
        z1 = th.log(p / (1 - p))             # logit(p)
        z0 = th.zeros_like(z1)               # 便宜上クラス0のlogit=0
        logits = th.stack([z0, z1], dim=1)   # [B, 2]

        return logits

    
    def save(self, path):
        print(path)
        th.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(th.load(path))

