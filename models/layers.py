import os, sys, torch
PROJECT_ROOT = "/home/yudai/Project/research/Graduation_Project"
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv



torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def encode_input(text, tokenizer):
    max_length = 512
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)
        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)
        self.drop = nn.Dropout(p=0.2)

        # --- ここから: 実ラン形状でFC in_featuresを推定 ---
        with torch.no_grad():
            C = self.conv1d_1_args["in_channels"]
            # Z 経路（hidden と x を結合した長さ）
            dummy_Z = torch.zeros(1, C, int(fc_1_size))
            Z = self.mp_1(F.relu(self.conv1d_1(dummy_Z)))
            Z = self.mp_2(self.conv1d_2(Z))
            Z_flatten_size = int(Z.shape[1] * Z.shape[-1])

            # Y 経路（hidden のみ）
            dummy_Y = torch.zeros(1, C, int(fc_2_size))
            Y = self.mp_1(F.relu(self.conv1d_1(dummy_Y)))
            Y = self.mp_2(self.conv1d_2(Y))
            Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        # Dense layers（in_features を実計測で設定）
        self.fc1 = nn.Linear(Z_flatten_size, 1)
        self.fc2 = nn.Linear(Y_flatten_size, 1)


    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))

        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Net(nn.Module):

    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, Conv=None):
        super().__init__()
        self.device = device

        rgcn_args = dict(gated_graph_conv_args)
        rgcn_args.setdefault("num_layers", 2)
        rgcn_args.setdefault("num_relations", 3)
        rgcn_args.setdefault("num_bases", 3)
        self.rgcn_args = rgcn_args

        hidden = rgcn_args["out_channels"]

        self.input_proj = nn.Linear(emb_size, hidden)

        self.rgcn_layers = nn.ModuleList([
            RGCNConv(
                in_channels = hidden,
                out_channels= hidden,
                num_relations= rgcn_args["num_relations"],
                num_bases   = rgcn_args["num_bases"]
            )
            for _ in range(rgcn_args["num_layers"])
        ])

        assert Conv is not None, "Conv クラス（あなたの実装）を渡してください"
        self.conv = Conv(
            **conv_args,
            fc_1_size = hidden + emb_size,
            fc_2_size = hidden
        ).to(device)

    def forward(self, data):
        from models.LMGNN import sanitize_for_rgcn
        d = sanitize_for_rgcn(data, trim_x=True, coalesce=True)
        x, edge_index, edge_type = d.x, d.edge_index, d.edge_type

        h = self.input_proj(x)
        for conv in self.rgcn_layers:
            h = F.relu(conv(h, edge_index, edge_type))

        out = self.conv(h, x)  # 既存の Conv 出力（sigmoid確率など）
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))



class PositionwiseFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    純粋な Cross-Attention だけを行うブロック。
    - FFN / LayerNorm / 残差接続は一切入れない。
    - それらは LMGNN.py 側で組み合わせて使う想定。
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # [B, L, D] 形式で扱う
        )

    def forward(
        self,
        query: torch.Tensor,      # [B, L_q, D]
        key: torch.Tensor,        # [B, L_k, D]
        value: torch.Tensor,      # [B, L_k, D]
        key_padding_mask=None,    # [B, L_k] (任意)
        need_weights: bool = False,
    ):
        """
        戻り値:
            out: [B, L_q, D]
            attn_weights: [B, num_heads, L_q, L_k]（need_weights=True のとき）
        """
        out, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return out



class GraphCodeFusion(nn.Module):
    """
    CodeBERT埋め込みとR-GCN埋め込みをCross-Attentionで融合するブロック。

    手順:
      1. code, graph それぞれ projection
      2. 双方向 cross-attention
      3. Add & LayerNorm (各側)
      4. pooling → concat
      5. FFN → LayerNorm → Linear
    """
    def __init__(
        self,
        code_dim: int,
        graph_dim: int,
        proj_dim: int = 200,
        num_heads: int = 4,
        ffn_hidden_dim: int = None,
        fusion_out_dim: int = 200,   # ここは最終的に出したい次元
        dropout: float = 0.1,
        attn_layers: int = 3,
    ):
        super().__init__()

        self.proj_code = nn.Linear(code_dim, proj_dim)
        self.proj_graph = nn.Linear(graph_dim, proj_dim)

        # 2) Cross-Attention を「attn_layers 層」分持つ
        self.attn_code_layers = nn.ModuleList([
            CrossAttentionBlock(proj_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(attn_layers)
        ])
        self.attn_graph_layers = nn.ModuleList([
            CrossAttentionBlock(proj_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(attn_layers)
        ])

        # 各層ごとの LayerNorm（Transformer 風）
        self.norm_code_layers = nn.ModuleList([
            nn.LayerNorm(proj_dim) for _ in range(attn_layers)
        ])
        self.norm_graph_layers = nn.ModuleList([
            nn.LayerNorm(proj_dim) for _ in range(attn_layers)
        ])

        # 4〜6) FFN → LayerNorm → Linear (concat後)
        fused_dim = proj_dim * 2  # code + graph
        self.ffn = PositionwiseFFN(fused_dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.norm_fused = nn.LayerNorm(fused_dim)
        self.linear_out = nn.Linear(fused_dim, fusion_out_dim)

    def forward(
        self,
        code_emb: torch.Tensor,            # [B, L_code, code_dim]
        graph_emb: torch.Tensor,           # [B, L_graph, graph_dim]
        code_mask: torch.Tensor | None = None,   # [B, L_code]
        graph_mask: torch.Tensor | None = None,  # [B, L_graph]
    ):

        """
        code_emb:  [B, L_code, code_dim]
        graph_emb:[B, L_graph, graph_dim]
        mask    : [B, L] (1=有効ノード, 0=パディング) or None
        return  : fused_repr [B, fusion_out_dim]  # グラフごとのベクトル
        """

        code_h = self.proj_code(code_emb)
        graph_h = self.proj_graph(graph_emb)

            # 2) Cross-Attention を attn_layers 回繰り返す
        for attn_code, attn_graph, norm_c, norm_g in zip(
            self.attn_code_layers,
            self.attn_graph_layers,
            self.norm_code_layers,
            self.norm_graph_layers,
        ):
            # code が query → graph を見る Cross-Attn
            code_ca = attn_code(
                query=code_h,
                key=graph_h,
                value=graph_h,
            )  # [B, Lc, D]

            # graph が query → code を見る Cross-Attn
            graph_ca = attn_graph(
                query=graph_h,
                key=code_h,
                value=code_h,
            )  # [B, Lg, D]

            # 残差接続 + LayerNorm（層ごとに別の Norm）
            code_h  = norm_c(code_h  + code_ca)    # [B, Lc, D]
            graph_h = norm_g(graph_h + graph_ca)   # [B, Lg, D]

        # 最終層の出力を out として扱う
        code_out, graph_out = code_h, graph_h

        # 4) プーリング（mask があれば “有効ノードだけ”）
        if code_mask is not None:
            code_mask_f = code_mask.float().unsqueeze(-1)        # [B, Lc, 1]
            code_sum  = (code_out * code_mask_f).sum(dim=1)      # [B, D]
            denom_c   = code_mask_f.sum(dim=1).clamp(min=1e-6)   # [B, 1]
            code_vec  = code_sum / denom_c                       # [B, D]
        else:
            code_vec  = code_out.mean(dim=1)                     # [B, D]

        if graph_mask is not None:
            graph_mask_f = graph_mask.float().unsqueeze(-1)      # [B, Lg, 1]
            graph_sum  = (graph_out * graph_mask_f).sum(dim=1)   # [B, D]
            denom_g   = graph_mask_f.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            graph_vec = graph_sum / denom_g                      # [B, D]
        else:
            graph_vec = graph_out.mean(dim=1)

        # 5) concat → FFN → LayerNorm → Linear
        fused = torch.cat([code_vec, graph_vec], dim=-1)  # [B, 2D]
        fused = self.ffn(fused)                           # [B, 2D]
        fused = self.norm_fused(fused)                    # [B, 2D]
        fused = self.linear_out(fused)                    # [B, fusion_out_dim]

        return fused