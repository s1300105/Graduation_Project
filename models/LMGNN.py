import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from .layers import GraphCodeFusion
from torch_geometric.data import Batch as GeoBatch


# ★ 置換: GGNN → R-GCN
# from torch_geometric.nn.conv import GatedGraphConv   # ← 不要
from torch_geometric.nn import RGCNConv
try:
    # PyG>=2.3 ならこちら
    from torch_geometric.utils import coalesce as pyg_coalesce
    _HAS_COALESCE = True
except Exception:
    _HAS_COALESCE = False

# === あなたの既存 Conv（そのまま利用） ==========================
# from .layers import Conv, encode_input  # encode_input は未使用でもOK
# ここでは質問に含まれている Conv 実装を前提とします
# ---------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce as pyg_coalesce, add_self_loops

def _infer_num_nodes(edge_index: torch.Tensor, fallback: int | None = None) -> int:
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1
    return int(fallback) if fallback is not None else 0

@torch.no_grad()
def _coalesce_triplets(edge_index: torch.Tensor,
                       edge_type: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    (src, dst, edge_type) の三つ組で重複除去。
    edge_type が None の場合は (src, dst) で重複除去。
    """
    if edge_index.numel() == 0:
        return edge_index, edge_type

    # 型と連続性を保証
    edge_index = edge_index.long().contiguous()
    if edge_type is not None:
        edge_type = edge_type.long().contiguous()

    if edge_type is None or edge_type.numel() != edge_index.size(1):
        # edge_type が無い or 長さ不一致 → ペアで coalesce（type は捨てるか先頭を残す）
        ei_pairs = edge_index.t()                              # (E, 2)
        uniq, first_idx = torch.unique(ei_pairs, dim=0, return_inverse=False, return_counts=False, return_index=True)
        edge_index = uniq.t().contiguous()
        if edge_type is not None and edge_type.numel() > 0:
            edge_type = edge_type.index_select(0, first_idx.sort()[0]).contiguous()
        else:
            edge_type = None
        return edge_index, edge_type

    # 三つ組で coalesce
    trip = torch.stack([edge_index[0], edge_index[1], edge_type], dim=1)  # (E, 3)
    uniq = torch.unique(trip, dim=0)                                      # (E',3)
    edge_index = uniq[:, :2].t().contiguous()
    edge_type  = uniq[:, 2].contiguous().long()
    return edge_index, edge_type

def sanitize_for_rgcn(d, *, trim_x: bool = True, do_coalesce: bool = True):
    """
    RGCNConv に安全に渡せるよう Data/Batch を整形:
      - long 型へ正規化
      - (src,dst,etype) で重複除去
      - エッジが無い場合は自己ループ付与（etype=0）
      - num_nodes と x の行数を整合
    """
    ei = d.edge_index
    et = getattr(d, "edge_type", None)

    if ei is None:
        # エッジ自体が無いケース
        n = int(getattr(d, "num_nodes", d.x.size(0) if getattr(d, "x", None) is not None else 0))
        if n <= 0 and getattr(d, "x", None) is not None:
            n = int(d.x.size(0))
        ei, _ = add_self_loops(torch.empty(2, 0, dtype=torch.long, device=d.x.device if getattr(d,"x",None) is not None else None),
                               num_nodes=n)
        et = torch.zeros(ei.size(1), dtype=torch.long, device=ei.device)
    else:
        ei = ei.long().contiguous()
        if et is not None:
            et = et.long().contiguous()

        if do_coalesce:
            ei, et = _coalesce_triplets(ei, et)

        # 0 本になったらフォールバック
        num_nodes = int(getattr(d, "num_nodes", _infer_num_nodes(ei, getattr(d, "x", None).size(0) if getattr(d,"x",None) is not None else None)))
        if ei.numel() == 0:
            ei, _ = add_self_loops(ei, num_nodes=num_nodes)
            et = torch.zeros(ei.size(1), dtype=torch.long, device=ei.device)

    # num_nodes の整合
    num_nodes = int(getattr(d, "num_nodes", _infer_num_nodes(ei, getattr(d, "x", None).size(0) if getattr(d,"x",None) is not None else None)))
    d.num_nodes = num_nodes

    # x の行数トリム（過剰パディング対策）
    if trim_x and getattr(d, "x", None) is not None and d.x.size(0) > num_nodes:
        d.x = d.x[:num_nodes].contiguous()

    d.edge_index = ei
    if et is not None:
        d.edge_type = et
    return d


class BertRGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, Conv=None):
        super().__init__()
        self.device = device

        # ------------------------------------------------------------
        # 1) R-GCN 用の設定
        # ------------------------------------------------------------
        hidden_dim   = gated_graph_conv_args.get("out_channels", 200)
        num_rel      = gated_graph_conv_args.get("num_relations", 3)
        num_layers   = gated_graph_conv_args.get("num_layers", 6)
        self.hidden_dim = hidden_dim
        self.code_dim   = emb_size

        # CodeBERT 埋め込み → R-GCN 入力次元に射影
        self.input_proj = nn.Linear(emb_size, hidden_dim)

        # R-GCN スタック
        self.rgcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.rgcn_layers.append(
                RGCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_relations=num_rel,
                )
            )

        # 既存 conv ヘッド（もう使わないなら None のままでOK）
        #self.conv = None
        #if Conv is not None:
        #    self.conv = Conv(**conv_args)

        # ------------------------------------------------------------
        # 2) ノードレベル Cross-Attention 融合ブロック
        #    入力: [B, L, code_dim], [B, L, hidden_dim], mask [B, L]
        #    出力: [B, L, fusion_out_dim]
        # ------------------------------------------------------------
        self.fusion = GraphCodeFusion(
            code_dim=emb_size,
            graph_dim=hidden_dim,
            proj_dim=hidden_dim,
            num_heads=4,
            ffn_hidden_dim=hidden_dim * 4,
            fusion_out_dim=hidden_dim,  # ノードごとに hidden_dim 次元を返す
            dropout=0.1,
        )

        # ------------------------------------------------------------
        # 3) グラフレベルの分類層
        #    （マスク付き平均プーリング後に 2 クラス logits）
        # ------------------------------------------------------------
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        """
        data.x         : [N, emb_size]   CodeBERT ノード埋め込み
        data.edge_index: [2, E]
        data.edge_type : [E]
        data.batch     : [N]            ノード→グラフID
        """
        x          = data.x          # [N, code_dim]
        edge_index = data.edge_index
        edge_type  = data.edge_type

        # ★ 1) 安全な batch ベクトルの再構成
        if isinstance(data, GeoBatch) and hasattr(data, "ptr") and data.ptr is not None:
            # ptr: [num_graphs+1], 差分が各グラフのノード数
            num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]        # [B]
            batch = torch.arange(num_nodes_per_graph.size(0), device=x.device).repeat_interleave(num_nodes_per_graph)
        elif hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            # 単一グラフの場合など：全部 0 番グラフとして扱う
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # 念のため整合チェック
        if batch.size(0) != x.size(0):
            # デバッグしやすいように assert ではなく調整するならこう
            # ここでは「多いほうに合わせず、とりあえず小さいほうに合わせる」などもあり
            min_n = min(batch.size(0), x.size(0))
            batch = batch[:min_n]
            x     = x[:min_n]
            edge_index = edge_index[:, edge_index.max(dim=0).values < min_n]  # ざっくり調整（必要なら）


        # ============================
        # 1) R-GCN でノード埋め込み h を作る
        # ============================
        # 1) CodeBERT x → R-GCN h
        h = self.input_proj(x)
        for conv in self.rgcn_layers:
            h = F.relu(conv(h, edge_index, edge_type))

        # 2) ノード列を [B, L, D] に（dense化）
        code_dense, mask = to_dense_batch(x, batch)    # [B, L, code_dim], [B, L]
        graph_dense, _   = to_dense_batch(h, batch)    # [B, L, hidden_dim]

        # 3) GraphCodeFusion で “グラフ表現” を直接もらう
        graph_repr = self.fusion(code_dense, graph_dense, mask)  # [B, hidden_dim]

        # 4) 分類
        logits = self.classifier(graph_repr)   # [B, 2]
        return logits


# models/codebert_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel

class CodeBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base", tune_last_n_layers: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768 or 769

        # ① 全層いったん凍結
        for p in self.bert.parameters():
            p.requires_grad = False

        # ② encoder の最後の n 層だけ解凍
        encoder_layers = self.bert.encoder.layer  # nn.ModuleList
        for layer in encoder_layers[-tune_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        # ③ （必要なら）LayerNorm や pooler も解凍してよい
        if hasattr(self.bert, "pooler"):
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids:      [N, L]
        attention_mask: [N, L]
        戻り値:         [N, hidden_size]  （CLS or 平均など）
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 通常は CLS を使う
        cls = out.last_hidden_state[:, 0, :]  # [N, H]
        return cls
