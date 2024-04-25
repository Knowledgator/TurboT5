import torch
import triton
import triton.language as tl

from src.ops.rms_norm import T5LayerNorm


def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    def torch_layer_norm(x, w, eps):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + eps)
        x_normalized = x * rms

        # Convert into half-precision if necessary
        if w.dtype in [torch.float16, torch.bfloat16]:
            x_normalized = x_normalized.to(w.dtype)

        res = x_normalized*w
        return res
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = T5LayerNorm.apply(x, weight, eps)
    y_ref = torch_layer_norm(x, weight, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = [_.grad.clone() for _ in [x, weight]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)