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
    def __init__(
        self,
        gated_graph_conv_args,
        conv_args,
        emb_size,
        device,
        Conv=None,
        debug_shapes: bool = False,
        log_wandb: bool = False,
        wandb_prefix: str = "debug",
    ):
        super().__init__()
        self.device = device

        self.num_relations = gated_graph_conv_args.get("num_relations", 3)
        self.num_layers = gated_graph_conv_args.get("num_layers", 6)

        # --- 1) CodeBERT ---
        self.func_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.func_encoder = CodeBERTEncoder(
            model_name="microsoft/codebert-base",
            tune_last_n_layers=2,
        )

        # === 方針②: モデル内部次元を CodeBERT に統一 ===
        self.bert_dim = self.func_encoder.hidden_size  # usually 768
        self.model_dim = self.bert_dim                 # unified model dimension
        self.node_feat_dim = emb_size                  # original node feature dimension

        # --- 2) Pre-GCN Contextualization (Nodes attend to Code tokens) ---
        # model_dim == bert_dim, so no projection needed
        self.code_to_model = nn.Identity()

        self.node_code_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.node_code_norm = nn.LayerNorm(self.model_dim)

        # --- 3) R-GCN ---
        # ② CodeBERT部分(768次元)のみ正規化（type_idは整数なので除外）
        self.node_feat_norm = nn.LayerNorm(self.node_feat_dim - 1)
        # ① type_id を Embedding で処理、CodeBERT 部分を別途投影して結合
        self.type_embed     = nn.Embedding(64, 64)
        self.node_feat_proj = nn.Sequential(          # ③ 2層MLPで表現空間のギャップを吸収
            nn.Linear(self.node_feat_dim - 1, self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.node_combine   = nn.Linear(self.model_dim + 64, self.model_dim)
        self.node_repr_norm = nn.LayerNorm(self.model_dim)  # ① node_combine 後の正規化

        # ③ コード側逆方向 Cross-Attention（コードがノードを参照）
        self.code_node_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.code_node_norm = nn.LayerNorm(self.model_dim)

        self.rgcn_layers = nn.ModuleList([
            RGCNConv(self.model_dim, self.model_dim, self.num_relations)
            for _ in range(self.num_layers)
        ])
        self.rgcn_norm = nn.LayerNorm(self.model_dim)

        # --- 4) Fusion (Code ↔ Graph) ---
        self.fusion = GraphCodeFusion(
            code_dim=self.bert_dim,          # 768
            graph_dim=self.model_dim,        # 768
            proj_dim=256,                    # ボトルネック (768→256)
            num_heads=4,
            ffn_hidden_dim=256 * 4,          # 1024
            fusion_out_dim=self.model_dim,   # 出力は768に戻す
            dropout=0.1,
            attn_layers=3,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.model_dim),           # ② Fusion出力を正規化してから分類
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),                              # ReLU → GELU
            nn.Dropout(0.2),
            nn.Linear(self.model_dim, 2),
        )

        self.debug_shapes = debug_shapes
        self.log_wandb = log_wandb
        self.wandb_prefix = wandb_prefix
        self._wandb = None
        if self.log_wandb:
            try:
                import wandb
                self._wandb = wandb
            except Exception:
                self._wandb = None

    def forward(self, data):
        node_feat = data.x
        edge_index = data.edge_index
        edge_type = data.edge_type

        # ------------------------------------------------------------
        # Step 1) CodeBERT token representations
        # ------------------------------------------------------------
        func_texts = getattr(data, "func", None)

        # Determine batch assignment for nodes
        if isinstance(data, GeoBatch) and hasattr(data, "ptr") and data.ptr is not None:
            nodes_per_graph = data.ptr[1:] - data.ptr[:-1]
            batch_index = torch.arange(nodes_per_graph.size(0), device=node_feat.device).repeat_interleave(nodes_per_graph)
            batch_size = int(nodes_per_graph.size(0))
        else:
            batch_index = node_feat.new_zeros(node_feat.size(0), dtype=torch.long)
            batch_size = 1

        if func_texts is None:
            code_tok_repr = node_feat.new_zeros(batch_size, 1, self.bert_dim)  # [B, 1, 768]
            code_tok_mask = None
        else:
            if isinstance(func_texts, str):
                func_texts = [func_texts]

            enc = self.func_tokenizer(
                list(func_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            code_tok_repr = self.func_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_cls=False,  # [B, L_code, 768]
            )
            code_tok_mask = attention_mask  # [B, L_code] (1=valid, 0=pad)

        # ------------------------------------------------------------
        # Step 2) Contextualize nodes by attending to code tokens
        # ------------------------------------------------------------
        # ① type_id を分離して Embedding、② CodeBERT 部分を LayerNorm してから投影
        type_ids      = node_feat[:, 0].long().clamp(0, 63)
        codebert_feat = node_feat[:, 1:]                            # [N, 768]
        codebert_norm = self.node_feat_norm(codebert_feat)          # ② 正規化
        type_emb      = self.type_embed(type_ids)                   # ① [N, 64]
        feat_emb      = self.node_feat_proj(codebert_norm)          # ① [N, 768]
        node_repr0    = self.node_repr_norm(F.gelu(self.node_combine(
            torch.cat([feat_emb, type_emb], dim=-1)
        )))  # [N, 768]

        code_tok_repr_model = self.code_to_model(code_tok_repr)  # Identity: [B, L_code, 768]

        node_repr_dense, node_valid_mask = to_dense_batch(node_repr0, batch_index)  # [B, L_node, 768]

        code_key_padding_mask = (code_tok_mask == 0) if code_tok_mask is not None else None  # True=ignore

        if self.debug_shapes:
            kp_shape = None if code_key_padding_mask is None else tuple(code_key_padding_mask.shape)
            print(
                f"[debug] node_repr_dense={tuple(node_repr_dense.shape)}, "
                f"code_tok_repr_model={tuple(code_tok_repr_model.shape)}, "
                f"code_key_padding_mask={kp_shape}"
            )

        attn_out, _ = self.node_code_attn(
            query=node_repr_dense,
            key=code_tok_repr_model,
            value=code_tok_repr_model,
            key_padding_mask=code_key_padding_mask,
        )
        node_repr_ctx_dense = self.node_code_norm(node_repr_dense + attn_out)

        # Back to flat: [TotalNodes, 768]
        node_repr_ctx = node_repr_ctx_dense[node_valid_mask]

        # ③ コードがノードを参照（逆方向 Cross-Attention）
        node_key_padding_mask = ~node_valid_mask  # True=padding 位置を無視
        # 全ノードがパディングのサンプルがあると全マスク→NaN になるため先頭を強制有効化
        all_node_masked = node_key_padding_mask.all(dim=1, keepdim=True)
        safe_node_key_padding_mask = node_key_padding_mask & ~all_node_masked
        code_attn_out, _ = self.code_node_attn(
            query=code_tok_repr_model,
            key=node_repr_dense,
            value=node_repr_dense,
            key_padding_mask=safe_node_key_padding_mask,
        )
        code_tok_repr = self.code_node_norm(code_tok_repr_model + code_attn_out)

        # ------------------------------------------------------------
        # Step 3) R-GCN over contextualized node representations
        # ------------------------------------------------------------
        node_repr_gcn = node_repr_ctx
        for rgcn in self.rgcn_layers:
            node_repr_gcn = F.relu(rgcn(node_repr_gcn, edge_index, edge_type)) + node_repr_gcn
        node_repr_gcn = self.rgcn_norm(node_repr_gcn)

        # ------------------------------------------------------------
        # Step 4) Fusion + Classification
        # ------------------------------------------------------------
        graph_node_repr_dense, graph_node_mask = to_dense_batch(node_repr_gcn, batch_index)  # [B, L_graph, 768]

        graph_repr = self.fusion(
            code_emb=code_tok_repr,              # [B, L_code, 768]
            graph_emb=graph_node_repr_dense,     # [B, L_graph, 768]
            code_mask=code_tok_mask,             # [B, L_code]
            graph_mask=graph_node_mask,          # [B, L_graph]
        )  # [B, model_dim(=768)]

        logits = self.classifier(graph_repr)
        return logits

    

# models/LMGNN.py に追加
class CodeBERTOnly(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.func_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.func_encoder = CodeBERTEncoder(
            model_name="microsoft/codebert-base",
            tune_last_n_layers=2,
        )

        self.bert_dim = self.func_encoder.hidden_size  # 768

        # 768統一（BertRGCNのclassifier容量と揃える）
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.bert_dim, 2),
        )

    def forward(self, data):
        func_texts = getattr(data, "func", None)
        if func_texts is None:
            # 通常ここには来ない前提なら 1 でOK（安全）
            return torch.zeros(1, 2, device=self.device)

        if isinstance(func_texts, str):
            func_texts = [func_texts]

        enc = self.func_tokenizer(
            list(func_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        cls_emb = self.func_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_cls=True,   # [B, 768]
        )

        logits = self.classifier(cls_emb)
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

