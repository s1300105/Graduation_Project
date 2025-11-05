import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

th.manual_seed(2020)


def encode_input(text, tokenizer):
    max_length = 512
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask
