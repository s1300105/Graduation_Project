import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .layers import Conv, encode_input
from torch_geometric.nn.conv import GatedGraphConv
from transformers import RobertaTokenizer, RobertaModel

class BertGGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(BertGGCN, self).__init__()
        self.k = 0.1
        self.device = device

        # GatedGraphConv と Conv をデバイスへ
        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(
            **conv_args,
            fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
            fc_2_size=gated_graph_conv_args["out_channels"],
        ).to(device)

        # CodeBERT（安全読み込み）
        self.nb_class = 2
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained(
            "microsoft/codebert-base",
            use_safetensors=True,
            low_cpu_mem_usage=True,
        ).to(device)

        # 隠れ次元は config から直接
        self.feat_dim = self.bert_model.config.hidden_size
        self.classifier = nn.Linear(self.feat_dim, self.nb_class).to(device)
        self.pre_lin = None


    def forward(self, data):
        # DataBatch(x=[N, D], edge_index=[2, E], y=[B], func=[B], batch=[N], ...)
        # NOTE: 前処理で data.x を完成させる。forward中に update_nodes はしない。
        # if self.training:
        #     self.update_nodes(data)  # ← 使わない（PyGのdata.xはTensor）

        # ---- Graph branch ----

        print("[sanity] x shape:", batch.x.shape)  # 例: torch.Size([1640, 101])



        x, edge_index = data.x, data.edge_index            # x: [N, D]
        in_dim = x.size(-1)
        out_dim = self.ggnn.out_channels
        if in_dim > out_dim:
            if self.pre_lin is None:
                self.pre_lin = nn.Linear(in_dim, out_dim).to(self.device)
            x = self.pre_lin(x)
        x = self.ggnn(x, edge_index)                       # [N, H]
        x = self.conv(x, data.x)                           # 期待: [B, C] の「確率」 (layers.Conv 実装に依存)

        # ---- Text branch (CodeBERT) ----
        input_ids, attention_mask = encode_input(data.func, self.tokenizer)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # [B, L, H] -> [B, H]（[CLS]相当=先頭トークン）
        cls_feats = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        cls_logit = self.classifier(cls_feats)             # [B, C] logits

        # ---- 融合 ----
        # conv側が確率、bert側がlogitだったので空間を揃える：
        # 1) BERT logits -> softmax で確率へ
        cls_prob = F.softmax(cls_logit, dim=1)             # [B, C]
        # 2) Conv出力は既に softmax 済み想定（layers.Convがそうなっている）
        #    もし logits なら F.softmax(x, dim=1) に変えてください。
        prob = self.k * x + (1 - self.k) * cls_prob        # [B, C] convex mixture
        log_prob = th.log(prob + 1e-10)                    # NLLLoss 用に log-prob に

        return log_prob

    def update_nodes(self, data):
        # PyG では forward 中に .x を作り直さない（重すぎ & 破壊的）。
        # ノード文字列→埋め込み（CodeBERT等）は学習前に前処理で作って data.x に入れてください。
        raise NotImplementedError(
            "Precompute node embeddings offline and store them in data.x (Tensor)."
        )

    def save(self, path):
        th.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(th.load(path))

