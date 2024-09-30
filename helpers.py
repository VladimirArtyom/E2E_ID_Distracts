import torch.nn as nn
import torch
from torch import Tensor
def initEmbedding(num_embeddings: int, embedding_dim: int, padding_idx: int):
    embed = nn.Embedding(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         padding_idx=padding_idx)
    nn.init.uniform_(embed.weight, a=-0.1, b=0.1)
    nn.init.constant_(embed.weight[padding_idx], val=0)
    return embed

def initLinear(in_features: int, out_features: int,
               bias: bool = True) -> nn.Module:
    lin = nn.Linear(in_features, out_features, bias)
    nn.init.uniform_(lin.weight, a=-0.1, b=0.1)
    if bias:
        nn.init.uniform_(lin.bias, a=-0.1, b=0.1)
    return lin

def initLSTM(input_size: int, hidden_size: int, **kwargs) -> nn.Module:
    lstm = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, params in lstm.named_parameters():
        if "weight" in name or "bias" in name:
            nn.init.uniform_(params, a=-0.1, b=0.1)
    return lstm


def initLSTMCell(input_size: int, hidden_size: int, **kwargs) -> nn.Module:
    cell = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, params in cell.named_parameters():
        if "weight" in name or "bias" in name:
            nn.init.uniform_(params, a=-0.1, b=0.1)
    return cell

def len_to_mask(source: Tensor, max_len: int=None, dtype=None):
    """
    If x1 = [3, 2] and max_len_x1 = 4, then x1_mask would look like:
    [[1, 1, 1, 0],
    [1, 1, 0, 0]] 
    1 is the valid mask <- Go on try in test
    """
    assert len(source.shape) == 1 # 1D tensor
    max_len = max_len if max_len is not None else source.max().item()
    out: Tensor = torch.arange(max_len, device=source.device,
                                dtype=source.dtype) # [maxlen]
    out = out.expand(len(source), max_len) < source.unsqueeze(1)
    #out_expand --> [batch_size, max_len]
    #source_unsqueeze --> [batch_size, 1] --> max_len < that 1 value, weird asf 
    if dtype is not None:
        out = torch.as_tensor(out, dtype=dtype, device=source.device)
    return out

def create_mask(x1: Tensor, x2: Tensor,
                 max_len_x1: int, max_len_x2: int):
    """
     The aim is to create 
     a mask that combines the valid positions of both x1 and x2 
     because these prepared the sequences for the attention calculation later which valid and not valid
    """
    batch_size: Tensor = x1.size(0)
    x1_mask = len_to_mask(x1, max_len_x1)
    x2_mask = len_to_mask(x2, max_len_x2)
    
    x1_mask = x1_mask.view(batch_size, 1, -1).transpose(-2, -1) #[batch_size, x1_maxlen, 1]
    x2_mask = x2_mask.view(batch_size, 1, -1) #[batch_size, 1, x2_maxlen]
    #print(x1_mask.shape, x2_mask.shape)

    return x1_mask * x2_mask # [batch_size, x1_maxlen, x2_maxlen] <-- This is only element-wise multiplication


    