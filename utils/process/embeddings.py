import os
import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import tokenizer  # 使わないなら消してOK
from utils.functions import log as logger
from torch_geometric.utils import add_self_loops

# transformers の import は遅延ロード（fastモード時に無駄に読み込まない）
try:
    from transformers import RobertaTokenizer, RobertaModel
except Exception:
    RobertaTokenizer = None
    RobertaModel = None


class NodesEmbedding:
    """
    Node特徴を作るクラス。
    - mode="codebert": CodeBERT (CLS) を使う本番モード
    - mode="fast"    : 形状を維持したダミー特徴（超高速・検証用）
    """
    def __init__(
        self,
        nodes_dim: int,
        *,
        max_len: int = 128,
        batch_size: int = 32,
        dtype=torch.float16,
        mode: str = "codebert",  # ★ 追加
    ):
        self.nodes_dim = int(nodes_dim)
        assert self.nodes_dim >= 0

        # 環境変数で fast を強制（例：USE_FAST_NODES_EMBEDDING=1）
        env_fast = os.environ.get("USE_FAST_NODES_EMBEDDING", "").strip() in {"1", "true", "True"}
        if env_fast:
            mode = "fast"
        self.mode = mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # 共通: 出力形状は [type_id(1) + 768] = 769
        self.bert_dim = 768
        self.feat_dim = 1 + self.bert_dim
        # パディング用バッファ（CPU, fp32）
        self.target = torch.zeros(self.nodes_dim, self.feat_dim, dtype=torch.float32)

        # 推論パラメータ
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        # CodeBERT モードのみ重い初期化
        self._cache = {}
        self._is_codebert = (self.mode == "codebert")
        if self._is_codebert:
            if RobertaTokenizer is None or RobertaModel is None:
                raise RuntimeError("transformers が読み込めません。fastモードを使用するか、transformers をインストールしてください。")
            # HF の新APIは dtype 推奨（古い torch_dtype は非推奨）
            self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
            self.bert_model = RobertaModel.from_pretrained(
                "microsoft/codebert-base",
                dtype=dtype if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.bert_model.eval()
        else:
            # fastモードはモデルを持たない
            self.tokenizer_bert = None
            self.bert_model = None

    def __call__(self, nodes):
        if self._is_codebert:
            emb_np = self._embed_nodes_codebert(nodes)     # np.array [N, feat_dim]
        else:
            emb_np = self._embed_nodes_fast(nodes)         # np.array [N, feat_dim]
        nodes_tensor = torch.from_numpy(emb_np).float()    # CPU, fp32

        # 前回より短いグラフが来た時に残りが汚れないようゼロ埋め
        self.target.zero_()
        n = min(nodes_tensor.size(0), self.nodes_dim)
        if n > 0:
            self.target[:n, :] = nodes_tensor[:n, :]

        return self.target

    # ========= fastモード =========
    def _embed_nodes_fast(self, nodes):
        """
        CodeBERTを使わず、先頭数次元のみ簡単な決定的特徴で埋めた [N,769] を返す。
        [0]=type_id, [1]=len(code), [2]=line, [3]=column, [4]=hash(code[:4])
        残りは0。
        """
        node_items = list(nodes.items())
        N = len(node_items)
        out = np.zeros((N, self.feat_dim), dtype=np.float32)
        for i, (_, node) in enumerate(node_items):
            code = node.get_code() or ""
            type_id = getattr(node, "type", 0)
            # line/column の取り方は実装に合わせて安全に
            try:
                line = node.get_line_number() or 0
            except Exception:
                line = getattr(node, "line_number", 0) or 0
            try:
                col = node.get_column_number() or 0
            except Exception:
                col = getattr(node, "column_number", 0) or 0

            out[i, 0] = float(type_id)
            out[i, 1] = float(len(code))
            out[i, 2] = float(line)
            out[i, 3] = float(col)
            h = int.from_bytes(code.encode("utf-8")[:4].ljust(4, b"\0"), "little")
            out[i, 4] = float(h % 1_000_003)
        return out

    # ========= CodeBERTモード =========
    def _batch_encode(self, texts):
        return self.tokenizer_bert(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

    def _embed_nodes_codebert(self, nodes):
        node_items = list(nodes.items())
        codes, type_ids = [], []
        for _, node in node_items:
            code = node.get_code()
            codes.append(code if isinstance(code, str) else str(code))
            type_ids.append(getattr(node, "type", 0))

        feats = []
        with torch.inference_mode():
            for i in range(0, len(codes), self.batch_size):
                batch_codes = codes[i: i + self.batch_size]

                # キャッシュミスだけ推論
                miss_texts = [c for c in batch_codes if c not in self._cache]
                if miss_texts:
                    enc = self._batch_encode(miss_texts)
                    enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
                    if self.device.type == "cuda":
                        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                            hidden = self.bert_model(**enc).last_hidden_state  # [B,L,H]
                    else:
                        hidden = self.bert_model(**enc).last_hidden_state
                    cls = hidden[:, 0]                         # [B,H]
                    cls_np = cls.detach().cpu().numpy()
                    # キャッシュへ保存
                    for j, c in enumerate(miss_texts):
                        self._cache[c] = cls_np[j]
                    # 一時テンソル開放
                    del enc, hidden, cls
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # バッチ全要素をキャッシュから取り出し
                for c in batch_codes:
                    feats.append(self._cache[c])

        feats = np.stack(feats, axis=0).astype(np.float32)          # [N, 768]
        type_arr = np.asarray(type_ids, dtype=np.float32).reshape(-1, 1)
        emb = np.concatenate([type_arr, feats], axis=1)             # [N, 1+768]
        return emb


class GraphsEmbedding:
    """
    ※ 旧・単関係用。多関係は下の build_edges_all を使用推奨。
    """
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)
        return torch.tensor(connections).long()

    def nodes_connectivity(self, nodes):
        coo = [[], []]
        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")
            for _, edge in node.edges.items():
                if edge.type != self.edge_type:
                    continue
                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)
                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)
        return coo


EDGE_MAP = {
    "AST":0, "AST_EDGE":0,
    "CFG":1, "CFG_EDGE":1,
    "PDG":2, "REACHING_DEF":2, "DATA_FLOW":2, "DDG":2,
    # 必要ならここに追記
}

def build_edges_all(nodes):
    src, dst, et = [], [], []
    for _, node in nodes.items():
        u = node.order
        for _, e in (node.edges or {}).items():
            t = e.type
            if t not in EDGE_MAP:
                continue
            if e.node_in in nodes and nodes[e.node_in].order != u:
                src.append(nodes[e.node_in].order); dst.append(u);  et.append(EDGE_MAP[t])
            if e.node_out in nodes and nodes[e.node_out].order != u:
                src.append(u); dst.append(nodes[e.node_out].order); et.append(EDGE_MAP[t])
    if not src:
        return torch.empty((2,0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(et, dtype=torch.long)


def nodes_to_input(nodes, target, nodes_dim, *, mode: str = "codebert"):
    """
    nodes: OrderedDict[node_id -> node]
    target: int/float
    nodes_dim: 最大ノード数（パディング幅）
    mode: "codebert" | "fast"  ← ★ ここで切替（検証時は "fast" 推奨）
    """
    nodes_embedding = NodesEmbedding(
        nodes_dim,
        max_len=128,
        batch_size=32,
        dtype=torch.float16,
        mode=mode,  # ★ 追加
    )
    x = nodes_embedding(nodes)  # [nodes_dim, 1+768] 先頭n行が実ノード、残りは0埋め
    n_real = min(len(nodes), nodes_dim)

    edge_index, edge_type = build_edges_all(nodes)

    # エッジ0本のときは自己ループでフォールバック
    if edge_index.numel() == 0:
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_real)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

    node_list = sorted(list(nodes.values()), key=lambda z: z.order)

    def _i(x):  # int or -1
        try:
            v = int(x)
            return v if v > 0 else -1
        except Exception:
            return -1

    start_lines, end_lines, start_cols, end_cols = [], [], [], []
    for nd in node_list:
        sline = _i(getattr(nd, "line_number", None))
        scol  = _i(getattr(nd, "column_number", None))

        # END_LINE が無いので CODE の改行数から簡易推定（最低でも start と同じ）
        code_str = nd.get_code() if hasattr(nd, "get_code") else ""
        if sline > 0 and isinstance(code_str, str):
            e_line = sline + code_str.count("\n")
        else:
            e_line = sline

        # end_column は持っていないので同値に（必要なら後で改善）
        ecol = scol

        start_lines.append(sline)
        end_lines.append(e_line if e_line > 0 else sline)
        start_cols.append(scol)
        end_cols.append(ecol)

    line_t     = torch.tensor(start_lines, dtype=torch.long)
    end_line_t = torch.tensor(end_lines,  dtype=torch.long)
    col_t      = torch.tensor(start_cols, dtype=torch.long)
    end_col_t  = torch.tensor(end_cols,   dtype=torch.long)

    # === ここまで追加 ===

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([float(target)], dtype=torch.float32),
    )
    data = coalesce_by_triplet_(data)
    data.num_nodes = n_real

    # === ここから追加：属性として積む ===
    data.line = line_t
    data.end_line = end_line_t
    data.column = col_t
    data.end_column = end_col_t
    # === ここまで追加 ===

    return data




@torch.no_grad()
def coalesce_by_triplet_(d: Data) -> Data:
    """
    (src, dst, edge_type) が完全一致のエッジ重複を除去（in-place）。
    edge_attr 等があれば同様に並び替えてください。
    """
    ei = d.edge_index
    assert ei is not None and torch.is_tensor(ei) and ei.dim() == 2 and ei.size(0) == 2

    # edge_type が無ければ (src,dst) で統合
    if not (hasattr(d, "edge_type") and torch.is_tensor(d.edge_type) and d.edge_type.numel() == ei.size(1)):
        pairs = torch.stack([ei[0], ei[1]], dim=1)
        uniq, idx = torch.unique(pairs, dim=0, return_inverse=False, return_counts=False, return_indices=True)
        d.edge_index = uniq.t().contiguous()
        # もし edge_type が None でなく長さ不一致なら、合わせてインデックスで引き直す
        return d

    et = d.edge_type
    trip = torch.stack([ei[0], ei[1], et], dim=1)              # (E, 3)
    uniq, inv = torch.unique(trip, dim=0, return_inverse=True) # uniq: (E',3)

    # 一意な行だけ残す
    keep = torch.unique(inv, sorted=True)
    uniq_trip = uniq.index_select(0, torch.arange(uniq.size(0), device=uniq.device))  # not strictly needed
    d.edge_index = uniq[:, :2].t().contiguous()
    d.edge_type  = uniq[:, 2].contiguous()

    return d
