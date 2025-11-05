import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import tokenizer  # 使わないなら消してOK
from utils.functions import log as logger
from transformers import RobertaTokenizer, RobertaModel

class NodesEmbedding:
    def __init__(
        self,
        nodes_dim: int,
        *,
        max_len: int = 128,      # ← ここを短くするとさらに省メモリ（64/96/128）
        batch_size: int = 32,    # ← OOMなら16/8へ
        dtype=torch.float16      # ← GPUならfp16、CPUなら自動でfp32に落ちる
    ):
        self.nodes_dim = int(nodes_dim)
        assert self.nodes_dim >= 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # HuggingFace
        self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained(
            "microsoft/codebert-base",
            dtype=dtype if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.bert_model.eval()  # 推論モード

        self.bert_dim = self.bert_model.config.hidden_size  # 768
        self.feat_dim = 1 + self.bert_dim                   # [type_id | CLS(768)]

        # パディング用バッファ（CPU, fp32）
        self.target = torch.zeros(self.nodes_dim, self.feat_dim, dtype=torch.float32)

        # 推論パラメータ
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        # 同一コード再計算回避のキャッシュ
        self._cache = {}

    def __call__(self, nodes):
        emb_np = self.embed_nodes(nodes)                 # np.array [N, feat_dim]
        nodes_tensor = torch.from_numpy(emb_np).float()  # CPU, fp32

        # 前回より短いグラフが来た時に残りが汚れないようゼロ埋め
        self.target.zero_()
        n = min(nodes_tensor.size(0), self.nodes_dim)
        if n > 0:
            self.target[:n, :] = nodes_tensor[:n, :]

        return self.target

    def _batch_encode(self, texts):
        # 直接HuggingFaceのtokenizerに通す：余計な中間テンソルを作らない
        return self.tokenizer_bert(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )

    def embed_nodes(self, nodes):
        # 入力整形（リスト化）
        node_items = list(nodes.items())
        codes = []
        type_ids = []
        for _, node in node_items:
            code = node.get_code()
            codes.append(code if isinstance(code, str) else str(code))
            type_ids.append(getattr(node, "type", 0))

        feats = []
        # 勾配も中間も完全OFF
        with torch.inference_mode():
            for i in range(0, len(codes), self.batch_size):
                batch_codes = codes[i: i + self.batch_size]

                # キャッシュミスだけ推論
                miss_positions = []
                miss_texts = []
                for j, c in enumerate(batch_codes):
                    if c not in self._cache:
                        miss_positions.append(j)
                        miss_texts.append(c)

                if miss_texts:
                    enc = self._batch_encode(miss_texts)
                    enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

                    if self.device.type == "cuda":
                        # 半精度で前向き
                        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                            hidden = self.bert_model(**enc).last_hidden_state  # [B, L, H]
                    else:
                        hidden = self.bert_model(**enc).last_hidden_state

                    cls = hidden[:, 0]                         # [B, H]
                    cls_np = cls.detach().cpu().numpy()        # 即CPUへ

                    # キャッシュへ保存
                    for j, c in zip(range(len(miss_positions)), miss_texts):
                        self._cache[c] = cls_np[j]

                    # 一時テンソルを開放
                    del enc, hidden, cls
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                # バッチ全要素をキャッシュから取り出し
                for c in batch_codes:
                    feats.append(self._cache[c])

        feats = np.stack(feats, axis=0).astype(np.float32)          # [N, 768]
        type_arr = np.asarray(type_ids, dtype=np.float32).reshape(-1, 1)  # [N, 1]
        emb = np.concatenate([type_arr, feats], axis=1)             # [N, 1+768]
        return emb


class GraphsEmbedding:
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


def nodes_to_input(nodes, target, nodes_dim, edge_type):
    # 省メモリ設定はここで調整可能（必要なら configs へ）
    nodes_embedding = NodesEmbedding(
        nodes_dim,
        max_len=128,     # 64/96/128 などで調整
        batch_size=32,   # OOMなら16→8
        dtype=torch.float16
    )
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    return Data(
        x=nodes_embedding(nodes),                  # [nodes_dim, 1+768] (CPU, fp32)
        edge_index=graphs_embedding(nodes),        # COO
        y=label
    )
