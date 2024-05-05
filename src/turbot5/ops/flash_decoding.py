import math
import torch
import triton
import triton.language as tl

def flash_attn_v2_fwd(q, k, v, bias, causal, sm_scale, NUM_SPLITS, BLOCK_M, BLOCK_N, num_warps, num_stages):

    B, H, M, D = q.shape
    N = k.shape[2]
    P_SEQ = N - M

    # Trick to support shape such as (1, 1, seqlen_q, seqlen_k)
    bias_batch_stride = bias.stride(0) if bias is not None else 0
    bias_heads_stride = bias.stride(1) if bias is not None else 0
    if bias is not None:
        if (bias.shape[0] != q.shape[0]) and (bias.shape[0] == 1):
            bias_batch_stride = 0
        if (bias.shape[1] != q.shape[1]) and (bias.shape[1] == 1):
            bias_heads_stride = 0

    divisible_n = N % BLOCK_N == 0
    # consider using 3d grid to avoid div & rem
    grid = (NUM_SPLITS, H, B)
    o = torch.empty_like(q)
    L = torch.zeros((B, H, NUM_SPLITS, M), device=q.device, dtype=torch.float32)
    ml = torch.zeros((B, H, NUM_SPLITS, M), device=q.device, dtype=torch.float32)
    so = torch.empty((B, H, NUM_SPLITS, M, D), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device.index):
        _fwd_kernel[grid](
            q, k, v, bias, sm_scale,
            L, ml, so,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            so.stride(0), so.stride(1), so.stride(2), so.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3),
            bias_batch_stride, bias_heads_stride,
            bias.stride(2) if bias is not None else 0,
            bias.stride(3) if bias is not None else 0,
            B, H, M, N, P_SEQ,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
            DIVISIBLE_N=divisible_n,
            HAS_BIAS=(bias is not None),
            NUM_SPLITS = NUM_SPLITS,
            num_warps=num_warps, num_stages=num_stages,
        )
    ml = ml.squeeze(-1)
    L = L.squeeze(-1)
    so = so.squeeze(-2)
    a_max = torch.max(ml, dim=-1, keepdim=True).values
    alpha = torch.exp(ml-a_max)
    max_log_scores_ = torch.log(alpha*L)
    weights = torch.softmax(max_log_scores_, dim=-1)
    res = torch.sum(weights.unsqueeze(-1) * so, dim=-2, keepdim=True)
    return res, L, ml

@triton.jit
def _fwd_kernel(
    Q, K, V, B, sm_scale,
    L, ml, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_os, stride_om,
    stride_lz, stride_lh, stride_ls, stride_lm,
    stride_bz, stride_bh, stride_bm, stride_bn,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    HAS_BIAS: tl.constexpr, NUM_SPLITS:tl.constexpr
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    off_s = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    n_per_split = N//NUM_SPLITS
    split_n_start = off_s*n_per_split
    split_n_end = N if off_s+1 == NUM_SPLITS else split_n_start+n_per_split

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634

    # offset pointers for (batch, head)
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    O += off_z * stride_oz + off_h * stride_oh + off_s*stride_os
    if HAS_BIAS:
        B += off_z * stride_bz + off_h * stride_bh
    L += off_z * stride_lz + off_h * stride_lh + off_s*stride_ls # l's shape is (B, H, NUM_SPLITS, M)
    ml += off_z * stride_lz + off_h * stride_lh + off_s*stride_ls # l's shape is (B, H, NUM_SPLITS, M)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :]) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m
    ml_ptrs = ml + offs_m
    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    mask_m = offs_m < M
    q = tl.load(q_ptrs, cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)
    # loop over k, v and update accumulators
    offs_n_init = offs_n_base+split_n_start
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    if HAS_BIAS:
        bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n_init[None, :] * stride_bn) # (BLOCK_M, BLOCK_N)

    for start_n in range(split_n_start, split_n_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        mask_n = offs_n < N
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- load bias --
        if HAS_BIAS:
            b = tl.load(bias_ptrs, mask_m[:, None] & mask_n[None, :])

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k) * sm_scale
        if HAS_BIAS:
            s += b

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new)*log2e)
        p = tl.math.exp2((s - m_i_new[:, None])*log2e)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        if HAS_BIAS:
            bias_ptrs += BLOCK_N * stride_bn

    acc = acc * (1.0 / l_i[:, None])
    # l = tl.log(l_i)#m_i + tl.log(l_i) # log(normalizer)
    l = l_i
    tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
    tl.store(ml_ptrs, m_i, mask=mask_m, cache_modifier=".cg")

    tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")