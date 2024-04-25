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

from typing import Any, Tuple, Union, List
import importlib.util
import importlib.metadata
import torch
from torch import nn
import torch.nn.functional as F

# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_flash_attn_available = _is_package_available("flash_attn")

def is_flash_attn_available():
    # Let's add an extra check to see if cuda is available
    import torch

    return _flash_attn_available and torch.cuda.is_available()

def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def mean_pooling(embeddings, attention_mask):
    if attention_mask is None:
        attention_mask = torch.ones(embeddings.shape[1]).to(embeddings.device)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_sqrt_pooling(embeddings, attention_mask):
    if attention_mask is None:
        attention_mask = torch.ones(embeddings.shape[1]).to(embeddings.device)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.sqrt(torch.clamp(input_mask_expanded.sum(1), min=1e-9))

def sum_pooling(embeddings, attention_mask):
    if attention_mask is None:
        attention_mask = torch.ones(embeddings.shape[1]).to(embeddings.device)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
    return torch.sum(embeddings * input_mask_expanded, 1)

# Max Pooling - Take the max value over time for every dimension. 
def max_pooling(embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
    return torch.max(embeddings, 1)[0]


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x

def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0 and x.shape[dim] // block_len>=1:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    elif x.shape[dim] // block_len<1:
        block_len = x.shape[dim]
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device).transpose(1, 2)
    return x.reshape(output_shape).transpose(1, 2)

def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def _make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)

def _split_attention_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0 and x.shape[dim] // block_len>=1:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    elif x.shape[dim] // block_len<1:
        block_len = x.shape[dim]
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)

def _get_local_attention_mask(attention_mask: torch.Tensor, block_len: int, device: torch.device) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_attention_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    curr_block_len = local_attention_mask.shape[2]
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, curr_block_len)
    # [batch_size, num_block, 1 , block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(2).to(device)


def _make_global_fixed_block_ids(
    attention_mask: torch.Tensor, global_block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: torch.Tensor) -> torch.Tensor:
        block_ends = (torch.arange(seq_len) % global_block_size) == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = torch.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = torch.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    fixed_block_mask = torch.ones_like(attention_mask, device=attention_mask.device) / global_block_size
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(attention_mask.dtype)
    _global_block_ids_lower_bound = torch.tensor(-1, dtype=global_block_ids.dtype, device=global_block_ids.device)
    global_block_ids = torch.where(
        global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound
    )
    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = torch.max(global_block_ids, dim=-1).values.repeat(num_globals, 1).transpose(0, 1)
    else:
        _sequence_block_ids_max = torch.zeros(
            batch_size, 0, dtype=global_block_ids.dtype, device=global_block_ids.device
        )
    global_segment_ids = torch.cumsum(torch.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = torch.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    return global_block_ids.type(torch.int), global_segment_ids.type(torch.int)


def _make_side_relative_position_ids(attention_mask: torch.Tensor, global_block_size: int) -> torch.Tensor:
    """Create the relative position tensor for local -> global attention."""
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    global_seq_len = global_segment_ids.shape[-1]
    global_positions = torch.arange(global_seq_len, device=block_ids.device)
    side_relative_position = global_positions - block_ids[..., None]
    return side_relative_position.type(torch.int64)


def _create_global_aggregates(
    hidden_states: torch.Tensor, block_ids: torch.Tensor, global_seq_len: int
) -> torch.Tensor:
    """Compute individual block aggregates by summing over individual blocks."""
    # (batch..., seq_len, global_seq_len))
    block_ids = block_ids.where(
        block_ids >= 0, torch.tensor(global_seq_len, dtype=block_ids.dtype, device=block_ids.device)
    )
    one_hot_block_ids = nn.functional.one_hot(block_ids.type(torch.int64), global_seq_len + 1)[:, :, :-1]
    return torch.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids.type(hidden_states.dtype))