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

import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer

from .config import T5Config
from .modules import T5LayerNorm
from .utils import (_split_into_blocks, _concatenate_3_blocks, _make_global_fixed_block_ids,
                    _create_global_aggregates, _get_local_attention_mask, _make_side_relative_position_ids)

from .padding import _upad_input, pad_input

from ..ops.flash_attention import flash_attention_with_bias
from ..ops.fused_bias_attention import flash_attention_with_fusing_bias
from ..ops.naive_attention import naive_torch_attention_with_bias
from ..ops.attention_bias import triton_compute_bias
from ..ops.varlen_fused_bias_attn import flash_attention_with_fusing_bias_varlen

class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device).unsqueeze(1)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device).unsqueeze(0)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values


    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=query_states.device, dtype=query_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=query_states.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias


        attn_output, attn_weights = naive_torch_attention_with_bias(query_states, key_states, value_states, position_bias_masked,
                                                          dropout = self.dropout, training = self.training,
                                                          layer_head_mask=layer_head_mask)
        #(batch_size, seq_len, hidden_dim)
        attn_output = self.o(unshape(attn_output)) 

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class T5TritonBasicAttention(T5Attention):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=query_states.device, dtype=query_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                if self.config.use_triton and query_states.device.type=='cuda':
                    bidirectional = not self.is_decoder
                    position_bias = triton_compute_bias(self.relative_attention_bias.weight, 
                                                        real_seq_length, key_length, self.n_heads,
                                                        bidirectional, self.relative_attention_num_buckets,
                                                        self.relative_attention_max_distance, dtype=query_states.dtype)
                    position_bias = position_bias.permute([2, 0, 1]).unsqueeze(0)
                else:
                    position_bias = self.compute_bias(real_seq_length, key_length, device=query_states.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        attn_weights = None
        attn_output = flash_attention_with_bias(query_states, key_states, value_states, position_bias_masked,
                                                causal=self.config.is_decoder, sm_scale = 1.0)

        #(batch_size, seq_len, hidden_dim)
        attn_output = self.o(unshape(attn_output)) 

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    

class T5FlashAttention(T5Attention):
    def __init__(self, config, cross_attention = False, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self.causal = config.is_decoder and not cross_attention

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.relative_attention_bias.weight

        attn_weights = None

        if mask is None:
            attn_output = flash_attention_with_fusing_bias(query_states, key_states, value_states, 
                                                    position_bias,
                                                    causal=self.causal, sm_scale = 1.0,
                                                    NUM_BUCKETS=self.relative_attention_num_buckets,
                                                    MAX_DISTANCE=self.relative_attention_max_distance)
        else:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            q, k, v, indices_q, cu_seq_lens, max_seq_lens = _upad_input(q, k, v, mask, seq_length, self.n_heads)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attention_with_fusing_bias_varlen(query_states, key_states, value_states, 
                                                    position_bias,
                                                    cu_seqlens_q = cu_seqlens_q, 
                                                    cu_seqlens_k = cu_seqlens_k,
                                                    max_seqlen_in_batch_q = max_seqlen_in_batch_q,
                                                    max_seqlen_in_batch_k = max_seqlen_in_batch_k,
                                                    causal=self.causal, sm_scale = 1.0,
                                                    NUM_BUCKETS=self.relative_attention_num_buckets,
                                                    MAX_DISTANCE=self.relative_attention_max_distance)
            

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

        attn_output = self.o(unshape(attn_output)) 

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    
class T5LocalAttention(T5Attention):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.block_len = config.sliding_window

    def compute_bias(self, block_len: int):
        """Compute binned relative position bias"""
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        memory_position = torch.arange(3*block_len, dtype=torch.long, device=target_device)
        context_position = memory_position[block_len:-block_len]

        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=None,
        output_attentions=False,
    ):
        
        if past_key_value is not None:
            warnings.warn('Currently, passing past key value states is not supported.')
        if use_cache is not None:
            warnings.warn('Currently, using cache is not supported for chosen attention type.')
                                      
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def shape_blocks(states):
            return states.view(-1, states.size(2), states.size(3), states.size(4))

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        # get query/key/value states -> (batch_size, n_heads, seq_length, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        
        # Split into blocks -> (batch_size, num_blocks, n_heads, block_len, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=2)
        key_states = _split_into_blocks(key_states, self.block_len, dim=2)
        value_states = _split_into_blocks(value_states, self.block_len, dim=2)

        # Concatenate 3 blocks for keys and values -> (batch_size*num_blocks, n_heads, 3 * block_len, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=3)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=3)

        query_states = shape_blocks(query_states)
        key_states = shape_blocks(key_states)
        value_states = shape_blocks(value_states)

        if mask is not None:
            # Replace masked positions with -1e10 (according to the original implementation)
            mask = torch.where(mask > 0, 0.0, -1e10)
            mask = shape_blocks(mask)
            
        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, query_states.shape[2], 3 * key_states.shape[2]), 
                    device=query_states.device, dtype=query_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(query_states.shape[2])
                
                position_bias = shape_blocks(position_bias)

            if mask is not None:
                # We need to adjust position bias shape to be sum with mask
                position_bias = position_bias + mask

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias
        

        attn_weights = None
        if self.config.use_triton and query_states.device.type == 'cuda':
            attn_output = flash_attention_with_bias(query_states, key_states, value_states, position_bias_masked,
                                                causal=self.config.is_decoder, sm_scale = 1.0)
        else:
            attn_output, attn_weights = naive_torch_attention_with_bias(query_states, key_states, value_states, position_bias_masked,
                                                          dropout = self.dropout, training = self.training,
                                                          layer_head_mask=layer_head_mask)
        
        # (batch_size*n_blocks, n_heads, block_len, dim_per_head) -> # (batch_size, seq_length, dim)
        attn_output = unshape(attn_output)[:,:seq_length,:]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5BlockAttention(T5Attention):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.block_len = config.sliding_window
        self.n_global_tokens = config.global_block_size

    def compute_bias(self, query_length, key_length, side_length=0, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        side_position = torch.arange(side_length, dtype=torch.long, device=device)

        context_position = torch.arange(side_length, query_length+side_length, dtype=torch.long, device=device).unsqueeze(1)
        memory_position = torch.arange(side_length, key_length, dtype=torch.long, device=device)
        memory_position = torch.cat([memory_position, side_position]).unsqueeze(0)

        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def get_side_position_bias(self, position_bias, init_key_length):
        # position_bias shape (1, num_heads, query_length, key_length)
        side_position_bias = position_bias[:, :, : self.n_global_tokens, :init_key_length]
        return side_position_bias

    def get_block_position_bias(self, position_bias, side_length, init_key_length):
        # position_bias shape (1, num_heads, block_len, key_length)
        if side_length:
            block_position_bias = position_bias[:, :, self.n_global_tokens :, :self.block_len+self.n_global_tokens]
        else:
            block_position_bias = position_bias[:, :, :init_key_length, :init_key_length]
        return block_position_bias        

    def get_extended_attention_mask(self, attention_mask, input_shape = None, device=None):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )
        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e10
        return extended_attention_mask

    def prepare_side_attention(self, attention_mask):
        #(batch_size, 1, 1, num_blocks, block_len)
        local_attention_mask = _split_into_blocks(attention_mask, self.block_len, 3)
        num_blocks = local_attention_mask.shape[3]
        if attention_mask.shape[-1]//self.block_len>=1>=1:
            # local_attention_mask[:, :, :, 0, :self.n_global_tokens] = -1e10
            side_attention_mask = torch.zeros(attention_mask.shape[0], 1, 1, num_blocks,
                                                        self.n_global_tokens, device=attention_mask.device)
            side_attention_mask[:, :, :, 0, :] = -1e10
            local_attention_mask = torch.cat([local_attention_mask, side_attention_mask], dim=-1)
        return local_attention_mask.squeeze(1).squeeze(1)[:, :, None, None, :]

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=None,
        output_attentions=False,
    ):
        
        if past_key_value is not None:
            warnings.warn('Currently, passing past key value states is not supported.')
        if use_cache is not None:
            warnings.warn('Currently, using cache is not supported for chosen attention type.')
                                      
        batch_size, seq_length = hidden_states.shape[:2]

        full_blocks = seq_length//self.block_len>=1

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def shape_blocks(states):
            return states.view(-1, states.size(2), states.size(3), states.size(4))

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        if mask is not None:
            #(batch_size, 1, 1, seq_length)
            mask = self.get_extended_attention_mask(mask)

        # get query/key/value states -> (batch_size, n_heads, seq_length, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        init_key_length = key_states.shape[-2]
        key_target_length = init_key_length+(-init_key_length%self.block_len)+self.n_global_tokens
        if position_bias is None:
            block_len = self.block_len+self.n_global_tokens if full_blocks else query_states.shape[2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, block_len, key_target_length), 
                    device=key_states.device, dtype=key_states.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(block_len, key_target_length)

        if full_blocks:
            side_key = key_states[:, :, :self.n_global_tokens, :]
            side_query = query_states[:, :, :self.n_global_tokens, :]
            side_value = value_states[:, :, :self.n_global_tokens, :]

            side_position_bias = self.get_side_position_bias(position_bias, init_key_length)
            if self.config.use_triton and query_states.device.type == 'cuda':
                side_global_value = flash_attention_with_bias(side_query, key_states, value_states, side_position_bias,
                                                    causal=self.config.is_decoder, sm_scale = 1.0)
            else:
                side_global_value, _ = naive_torch_attention_with_bias(side_query, key_states, value_states, side_position_bias,
                                                            dropout = self.dropout, training = self.training,
                                                            layer_head_mask=layer_head_mask)
            
        key_states = _split_into_blocks(key_states, self.block_len, 2)
        query_states = _split_into_blocks(query_states, self.block_len, 2)
        value_states = _split_into_blocks(value_states, self.block_len, 2)

        if full_blocks:
            reps = [1] * (side_key.ndim + 1)
            reps[1] = key_states.shape[1]
            side_key = side_key.unsqueeze(1).repeat(reps)
            side_value = side_value.unsqueeze(1).repeat(reps)
            side_length = side_key.shape[-2]
            #(batch_size, n_blocks, n_heads, block_len+side_length, hid_dim)
            key_states = torch.cat([side_key, key_states], dim=3)
            value_states = torch.cat([side_value, value_states], dim=3)
        else:
            side_length = 0
        
        query_states = shape_blocks(query_states)
        key_states =  shape_blocks(key_states)
        value_states = shape_blocks(value_states)

        # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
        block_bias = self.get_block_position_bias(position_bias, side_length, init_key_length)

        if mask is not None:
            #(batch_size, 1, 1, num_blocks, block_len+side_length)
            local_attention_mask = self.prepare_side_attention(mask)

            block_bias=local_attention_mask+block_bias

        if self.pruned_heads:
            mask = torch.ones(block_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = block_bias[:, mask.bool()]
        else:
            position_bias_masked = block_bias

        position_bias_masked = shape_blocks(position_bias_masked)

        attn_weights = None
        if self.config.use_triton and query_states.device == 'cuda':
            attn_output = flash_attention_with_bias(query_states, key_states, value_states, position_bias_masked,
                                                causal=self.config.is_decoder, sm_scale = 1.0)
        else:
            attn_output, attn_weights = naive_torch_attention_with_bias(query_states, key_states, value_states, position_bias_masked,
                                                          dropout = self.dropout, training = self.training,
                                                          layer_head_mask=layer_head_mask)

        #(batch_size, seq_len, hidden_dim)
        attn_output = unshape(attn_output)[:,:seq_length,:]

        if full_blocks:
            attn_output[:, :self.n_global_tokens, :] = unshape(side_global_value)

        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5TransientGlobalAttention(T5Attention):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.block_len = config.sliding_window
        self.global_block_size = config.global_block_size

        # Relativen attention bias
        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)

        self.global_input_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon, 
                                                        use_triton=self.config.use_triton)
        self.gradient_checkpointing = False

    def compute_bias(self, block_len: int):
        """Compute binned relative position bias"""
        target_device = (
            self.relative_attention_bias.weight.device
            if self.relative_attention_bias.weight.device.type != "meta"
            else None
        )
        memory_position = torch.arange(3 * block_len, dtype=torch.long, device=target_device)
        context_position = memory_position[block_len:-block_len]

        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def compute_side_bias(self, mask: torch.Tensor, global_segment_ids: torch.Tensor) -> torch.Tensor:
        # (batch_size, 1, seq_len, global_seq_len)
        side_attention_mask = torch.eq(mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
        attention_side_bias = torch.where(side_attention_mask > 0, 0.0, -1e10)
        # (batch_size, seq_len, global_seq_len)
        side_relative_position = _make_side_relative_position_ids(mask, self.global_block_size)
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (batch_size, seq_len, global_seq_len, num_heads)
        # side_bias = self.global_relative_attention_bias(side_relative_position_bucket)
        side_bias = self.relative_attention_bias(side_relative_position_bucket)

        # (batch_size, num_heads, seq_len, global_seq_len)
        side_bias = side_bias.permute([0, 3, 1, 2])
        # (batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=None,
        output_attentions=False,
    ):
        if past_key_value is not None:
            warnings.warn('Currently, passing past key value states is not supported.')
        if use_cache is not None:
            warnings.warn('Currently, using cache is not supported for chosen attention type.')
                      
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
        
        def shape_blocks(states):
            return states.view(-1, states.size(2), states.size(3), states.size(4))
        
        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)

        # Prepare components for transient-global attention
        # Obtain block_ids and global_segment_ids
        # global_seq_len := seq_len // self.global_block_size
        # shapes: (batch_size, seq_len) & (batch_size, global_seq_len)
        block_ids, global_segment_ids = _make_global_fixed_block_ids(
            mask if mask is not None else torch.ones(hidden_states.shape[:-1]),
            self.global_block_size,
        )
        # Create global inputs
        _global_seq_len = global_segment_ids.shape[-1]
        global_inputs = _create_global_aggregates(hidden_states, block_ids, _global_seq_len)
        global_inputs = self.global_input_layer_norm(global_inputs)

        # get query states -> (batch_size, n_heads, seq_length, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        # Get global/side key/value states  shape: (batch_size, n_heads, global_seq_len, dim_per_head)
        side_key_states = shape(self.k(global_inputs))
        side_value_states = shape(self.v(global_inputs))

        # Split into blocks -> (batch_size, num_blocks, n_heads, block_len, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=2)
        key_states = _split_into_blocks(key_states, self.block_len, dim=2)
        value_states = _split_into_blocks(value_states, self.block_len, dim=2)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, n_heads, 3 * block_len, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=3)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=3)

        # Tile side inputs across local key/value blocks
        # New shape: (batch_size, num_blocks, n_heads, global_seq_len, dim_per_head)
        reps = [1] * (side_key_states.ndim + 1)
        reps[1] = key_states.shape[1]
        side_key_states = side_key_states.unsqueeze(1).repeat(reps)
        side_value_states = side_value_states.unsqueeze(1).repeat(reps)

        # Concatenate "local" and "side"/"global" key/value states to allow each token to attend global aggregated ones
        # New shape: (batch_size, num_blocks, n_heads, 3 * block_len + global_seq_len, dim_per_head)
        key_states = torch.cat([key_states, side_key_states], dim=3)
        value_states = torch.cat([value_states, side_value_states], dim=3)

        #(batch_size*num_blocks, n_heads, block_len, dim_per_head)
        query_states = shape_blocks(query_states)
        # New shape: (batch_size*num_blocks, n_heads, 3 * block_len + global_seq_len, dim_per_head)
        key_states = shape_blocks(key_states)
        value_states = shape_blocks(value_states)

        if mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = _get_local_attention_mask(mask, self.block_len, hidden_states.device)
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = torch.where(local_attention_mask > 0, 0.0, -1e10)
        else:
            local_attention_mask = None
        
        if position_bias is None:
            # position_bias shape: # (1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.n_heads, query_states.shape[2], 3 * key_states.shape[2]),
                    device=key_states.device,
                    dtype=key_states.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(query_states.shape[2])

            if local_attention_mask is not None:
                # (batch_size, n_heads, block_len, 3 * block_len)
                position_bias = position_bias + local_attention_mask
            position_bias = position_bias.type(key_states.dtype)

            # Calculate global/side bias - shape: # (batch_size, num_heads, seq_len, global_seq_len)
            if mask is None:
                mask = torch.ones(batch_size, seq_length)
                reps = [1] * position_bias.ndim
                reps[0] = key_states.shape[0]
                position_bias = position_bias.repeat(reps)
            # (batch_size, num_heads, seq_len, global_seq_len)
            side_position_bias = self.compute_side_bias(mask, global_segment_ids)
            # (batch_size, num_blocks, num_heads, block_len, global_seq_len)
            side_position_bias = _split_into_blocks(side_position_bias, query_states.shape[2], dim=-2)#.transpose(1, 2)

            side_position_bias = side_position_bias.type(key_states.dtype).to(key_states.device)

            # (batch_size * num_blocks, num_heads, block_len, 3 * block_len + global_seq_len)
            side_position_bias = side_position_bias

            position_bias = shape_blocks(torch.cat([position_bias, side_position_bias], dim=-1))

        attn_weights = None
        if self.config.use_triton and query_states.device.type == 'cuda':
            attn_output = flash_attention_with_bias(query_states, key_states, value_states, position_bias,
                                                causal=self.config.is_decoder, sm_scale = 1.0)
        else:
            attn_output, attn_weights = naive_torch_attention_with_bias(query_states, key_states, value_states, position_bias,
                                                          dropout = self.dropout, training = self.training,
                                                          layer_head_mask=layer_head_mask)
     
        #(batch_size, seq_len, hidden_dim)
        attn_output = unshape(attn_output)[:,:seq_length,:]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs