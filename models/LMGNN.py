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


from transformers import AutoTokenizer


class BertRGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device, Conv=None):
        super().__init__()
        self.device = device

        hidden_dim   = gated_graph_conv_args.get("out_channels", 200)
        num_rel      = gated_graph_conv_args.get("num_relations", 3)
        num_layers   = gated_graph_conv_args.get("num_layers", 6)
        self.hidden_dim = hidden_dim
        self.code_dim   = emb_size

        # R-GCN 用
        self.input_proj = nn.Linear(emb_size, hidden_dim)
        self.rgcn_layers = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_rel)
            for _ in range(num_layers)
        ])

        # ★ 用途② CodeBERT（トークン列用）
        self.func_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.func_encoder   = CodeBERTEncoder(
            model_name="microsoft/codebert-base",
            tune_last_n_layers=2,   # ここだけ finetune される
        )

        # GraphCodeFusion: code_dim を func_encoder.hidden_size に合わせる
        self.fusion = GraphCodeFusion(
            code_dim=self.func_encoder.hidden_size,
            graph_dim=hidden_dim,
            proj_dim=hidden_dim,
            num_heads=4,
            ffn_hidden_dim=hidden_dim * 4,
            fusion_out_dim=hidden_dim,
            dropout=0.1,
            attn_layers=1,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )


    def forward(self, data):
        x          = data.x
        edge_index = data.edge_index
        edge_type  = data.edge_type

        # ★ node → graph インデックスを安全に作る
        if isinstance(data, GeoBatch) and hasattr(data, "ptr") and data.ptr is not None:
            # ptr: [0, n0, n0+n1, n0+n1+n2, ...]
            num_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]   # [B]
            assert int(num_nodes_per_graph.sum().item()) == x.size(0), \
                f"sum(ptr diff)={int(num_nodes_per_graph.sum().item())}, x.size(0)={x.size(0)} がズレてる"

            batch_idx = torch.arange(num_nodes_per_graph.size(0), device=x.device) \
                            .repeat_interleave(num_nodes_per_graph)   # [N_nodes]
        else:
            # 単一グラフ or last resort
            batch_idx = x.new_zeros(x.size(0), dtype=torch.long)

        # 1) R-GCN でノード埋め込み
        h = self.input_proj(x)
        for conv in self.rgcn_layers:
            h = F.relu(conv(h, edge_index, edge_type))

        

        graph_dense, graph_mask = to_dense_batch(h, batch_idx)

        # 2) 関数コードをトークナイズして CodeBERT に通す
        funcs = getattr(data, "func", None)
        if funcs is None:
            print("func is None. That is why code mask is None. Tokenized code is composed of 0")
            # 念のため保険：func がない場合は 0 ベクトル
            B = graph_dense.size(0)
            code_emb = graph_dense.new_zeros(B, 1, self.func_encoder.hidden_size)
            code_mask = None
        else:
            # PyG の Batch では data.func は list[str] になっている想定
            if isinstance(funcs, str):
                funcs = [funcs]
            enc = self.func_tokenizer(
                list(funcs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,  # 必要に応じて調整
            )
            input_ids     = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)  # [B, L_code]

            # ★ トークン列を取得（CLS ではない）
            code_emb = self.func_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls=False,        # [B, L_code, H]
            )
            code_mask = attention_mask  # そのまま mask に使える

        # 3) トークン列 vs ノード列で Cross-Attention
        graph_repr = self.fusion(
            code_emb=code_emb,              # [B, L_code, H]
            graph_emb=graph_dense,          # [B, L_graph, D]
            code_mask=code_mask,            # [B, L_code]
            graph_mask=graph_mask,          # [B, L_graph]
        )                                   # → [B, hidden_dim]

        # 4) 分類
        logits = self.classifier(graph_repr)
        return logits



# models/codebert_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel

class CodeBERTEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base", tune_last_n_layers: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # 全層いったん凍結
        for p in self.bert.parameters():
            p.requires_grad = False

        # encoder の最後の n 層だけ解凍（ここが finetune 対象）
        encoder_layers = self.bert.encoder.layer
        for layer in encoder_layers[-tune_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        if hasattr(self.bert, "pooler"):
            for p in self.bert.pooler.parameters():
                p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_cls: bool = True,
    ):
        """
        return_cls=True なら [B, H] (CLS)、
        False なら [B, L, H] (トークン列) を返す。
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B, L, H]

        if return_cls:
            return last[:, 0, :]      # [B, H]
        else:
            return last               # [B, L, H]

