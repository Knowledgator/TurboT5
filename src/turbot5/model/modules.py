# coding=utf-8
# Copyright 2020 Mesh TensorFlow authors, Knowledgator, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from transformers.activations import ACT2FN

from .config import T5Config
from ..ops.rms_norm import RMSLayerNorm
from ..ops.gated_mlp import GatedMLP

class T5LayerNorm(nn.Module):
    """
    RMS Normalization layer along the last dimension.

    This is similar to torch.nn.functional.normalize but with eps being added
    instead of max.

    Expects contiguous input of shape (..., dim), and returns normalized data
    of the same shape. For each dim-length vector x, the result has

        x / sqrt( x*x.sum() + eps)

    If weights are included, they are a parameter of length dim which multiplies
    the result.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """

    def __init__(self, dim: int, eps: float = 1e-6, use_triton = False):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.use_triton = use_triton

    def triton_forward(self, x: torch.Tensor):
        return RMSLayerNorm.apply(x, self.weight, self.eps)  # type: ignore
    
    def _forward(self, x: torch.Tensor):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)

        return self.weight * x
    
    def forward(self, x: torch.Tensor):
        if x.device.type == 'cuda' and self.use_triton:
            res = self.triton_forward(x)
        else:
            res = self._forward(x)
        return res
    

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.use_triton = config.use_triton
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]
        self.use_gelu_act = True if config.dense_act_fn == 'gelu' else False
    
    def _forward(self, hidden_states):
        hidden_act = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
    def triton_forward(self, hidden_states):
        hidden_states = GatedMLP.apply(hidden_states, self.wi_0.weight, self.wi_1.weight, self.use_gelu_act)
        return hidden_states
    
    def forward(self, hidden_states):

        if hidden_states.device.type == 'cuda' and self.use_triton:
            hidden_states = self.triton_forward(hidden_states)
        else:
            hidden_states = self._forward(hidden_states)

        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)

        return hidden_states
