import torch
from torch import nn

def naive_torch_attention_with_bias(q, k, v, position_bias=None,
                                    dropout = None, training = False, 
                                    layer_head_mask = None):
    scores = torch.einsum( "...qd,...kd->...qk", q, k)
    if position_bias is not None:
        scores += position_bias
    
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)

    if dropout is not None:
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=training
        ) 
    
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_outputs = torch.einsum("...hqk,...hkd->...hqd", attn_weights, v)

    return attn_outputs, attn_weights

