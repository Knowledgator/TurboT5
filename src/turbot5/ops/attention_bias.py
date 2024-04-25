import torch
import triton
import triton.language as tl

#out shape (N, M, NH)
@triton.jit
def bias_kernel(out, weights, stride_om, stride_on, stride_wn,
                N: tl.constexpr, M: tl.constexpr, NH: tl.constexpr, 
                BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_NH: tl.constexpr, 
                BIDIRECTIONAL: tl.constexpr, NUM_BUCKETS: tl.constexpr, 
                MAX_DISTANCE: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # Compute relative positions
    relative_positions = offs_n[None, :]-offs_m[:, None]

    # Compute bucket indices based on relative positions
    relative_buckets = tl.zeros_like(relative_positions)
    num_buckets = NUM_BUCKETS
    if BIDIRECTIONAL:
        num_buckets //= 2
        relative_buckets += (relative_positions > 0).to(tl.int32) * num_buckets
        relative_positions = tl.abs(relative_positions)
    else:
        relative_positions = tl.maximum(-relative_positions, tl.zeros_like(relative_positions))

    # Half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_positions < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        tl.log(relative_positions.to(tl.float32) / max_exact)
        / tl.log(MAX_DISTANCE / max_exact)
        * (num_buckets - max_exact)
    ).to(tl.int32)
    relative_position_if_large = tl.minimum(relative_position_if_large, num_buckets - 1)

    relative_buckets += tl.where(is_small, relative_positions, relative_position_if_large)

    for i in range(0, NH, BLOCK_NH):
        offs_nh = i + tl.arange(0, BLOCK_NH)
        bucket_offs = relative_buckets[:, :, None] * stride_wn + offs_nh[None, None, :]

        # Retrieve bias values from weights tensor
        bias_ptrs = weights + bucket_offs  # (BLOCK_M, BLOCK_N, BLOCK_NH)
        bias_values = tl.load(bias_ptrs)

        out_offs = (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)[:, :, None] + offs_nh[None, None, :]
        out_ptrs = out + out_offs

        o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)[:, :, None] & (offs_nh[None, None, :] < NH)

        # Store bias values in the output tensor
        tl.store(out_ptrs, bias_values, mask=o_mask)

@triton.jit
def bias_kernel_backward(
    d_weights, d_out, weights, stride_om, stride_on, stride_wn,
    N: tl.constexpr, M: tl.constexpr, NH: tl.constexpr, 
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_NH: tl.constexpr, 
    BIDIRECTIONAL: tl.constexpr, NUM_BUCKETS: tl.constexpr, 
    MAX_DISTANCE: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    relative_positions = offs_m[:, None] - offs_n[None, :]

    relative_buckets = tl.zeros_like(relative_positions)
    num_buckets = NUM_BUCKETS
    if BIDIRECTIONAL:
        num_buckets //= 2
        relative_buckets += (relative_positions > 0).to(tl.int32) * num_buckets
        relative_positions = tl.abs(relative_positions)
    else:
        relative_positions = tl.maximum(-relative_positions, tl.zeros_like(relative_positions))

    # Half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_positions < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        tl.log(relative_positions.to(tl.float32) / max_exact)
        / tl.log(MAX_DISTANCE / max_exact)
        * (num_buckets - max_exact)
    ).to(tl.int32)
    relative_position_if_large = tl.minimum(relative_position_if_large, num_buckets - 1)

    relative_buckets += tl.where(is_small, relative_positions, relative_position_if_large)

    for i in range(0, NH, BLOCK_NH):
        offs_nh = i + tl.arange(0, BLOCK_NH)
        bucket_offs = relative_buckets[:, :, None] * stride_wn + offs_nh[None, None, :]

        d_out_ptrs = d_out + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)[:, :, None] + offs_nh[None, None, :]
        o_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)[:, :, None] & (offs_nh[None, None, :] < NH)
        d_out_values = tl.load(d_out_ptrs, mask=o_mask, other=0.0)

        d_weights_ptrs = d_weights + bucket_offs
        tl.atomic_add(d_weights_ptrs, d_out_values, mask=relative_buckets[:, :, None] < NUM_BUCKETS)


class BiasOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, M, N, NH, BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE, dtype=torch.float16):
        ctx.save_for_backward(weights)
        ctx.M, ctx.N, ctx.NH = M, N, NH
        ctx.BIDIRECTIONAL, ctx.NUM_BUCKETS, ctx.MAX_DISTANCE = BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE
        ctx.dtype = dtype

        out = torch.empty((M, N, NH), device=weights.device, dtype=dtype)
        # Config
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_H = 16

        # Launch forward kernel
        grid = (triton.cdiv(N, BLOCK_SIZE_N) * triton.cdiv(M, BLOCK_SIZE_M),)
        bias_kernel[grid](
            out,
            weights,
            out.stride(0), out.stride(1), weights.stride(0),
            N, M, NH,
            BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_H,
            BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE, out.stride(1)
        )

        return out

    @staticmethod
    def backward(ctx, d_out):
        weights, = ctx.saved_tensors
        M, N, NH = ctx.M, ctx.N, ctx.NH
        BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE = ctx.BIDIRECTIONAL, ctx.NUM_BUCKETS, ctx.MAX_DISTANCE
        dtype = ctx.dtype

        d_weights = torch.zeros_like(weights)

        # Config
        BLOCK_SIZE_N = 32
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_H = 16

        # Launch backward kernel
        grid = (triton.cdiv(N, BLOCK_SIZE_N) * triton.cdiv(M, BLOCK_SIZE_M),)
        bias_kernel_backward[grid](
            d_weights,
            d_out,
            weights,
            d_out.stride(0), d_out.stride(1), weights.stride(0),
            N, M, NH,
            BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_H,
            BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE, d_out.stride(1)
        )

        return d_weights, None, None, None, None, None, None, None

def triton_compute_bias(weights, M, N, NH, BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE, dtype=torch.float16):
    # Check constraints
    assert weights.shape == (NUM_BUCKETS, NH), "Incorrect shape of weights tensor"
    assert weights.is_contiguous(), "Weights tensor must be contiguous"
    assert N > 0 and M > 0 and NH > 0, "Invalid dimensions"
    assert BIDIRECTIONAL in [True, False], "BIDIRECTIONAL must be a boolean"
    assert NUM_BUCKETS > 0, "NUM_BUCKETS must be positive"
    assert MAX_DISTANCE > 0, "MAX_DISTANCE must be positive"
    return BiasOp.apply(weights, M, N, NH, BIDIRECTIONAL, NUM_BUCKETS, MAX_DISTANCE, dtype)