import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from triton.language.extra import libdevice


@triton.jit
def layer_norm_kernel(
    input_ptr, # pointer to input tensor
    output_ptr, # pointer to output tensor
    gamma_ptr, # pointer to weight (gamma)
    beta_ptr, # pointer to bias (beta)
    stride_in , # stride of input tensor
    stride_out, # stride of output tensor
    is_bf16: tl.constexpr, # output data type
    H: tl.constexpr, # hidden size
    eps: tl.constexpr, # epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr, # block size for parallelism
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x_row_ptr = input_ptr + pid * stride_in + cols
    x = tl.load(x_row_ptr, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / H
    x_centered = x - mean
    x_centered = tl.where(mask, x_centered, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / H
    inv_std = tl.rsqrt(var + eps)
    x_norm = x_centered * inv_std

    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * gamma + beta
    # y = y.to(tl.bfloat16) if is_bf16 else y.to(tl.float16)
    y = y.to(tl.load(x_row_ptr, mask=mask, other=0.0).dtype) 
    y_row_ptr = output_ptr + pid * stride_out + cols
    tl.store(y_row_ptr, y, mask=mask)

def layer_norm_forward(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Applies Layer Normalization using a custom Triton kernel."""
    assert x.is_cuda, "Input tensor must be on CUDA device."
    assert gamma.is_cuda and beta.is_cuda, "Gamma and Beta tensors must be on CUDA device."
    batch_size, hidden_size = x.size()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (batch_size ,)

    is_bf16 = x.dtype == torch.bfloat16

    layer_norm_kernel[grid](
        x,
        output,
        gamma,
        beta,
        stride_in=x.stride(0),
        stride_out=output.stride(0),
        is_bf16=is_bf16,
        H=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

if __name__ == "__main__":
    B,S,H = 4,16,1024
    x = torch.randn((B*S,H), device="cuda", dtype=torch.bfloat16)
    gamma = torch.ones((H,), device="cuda", dtype=torch.bfloat16)
    beta = torch.zeros((H,), device="cuda", dtype=torch.bfloat16)

    ref = F.layer_norm(x, normalized_shape=(H,), weight=gamma, bias=beta, eps=1e-5)
    y = layer_norm_forward(x, gamma, beta, eps=1e-5)

    torch.testing.assert_close(ref, y, atol=1e-2, rtol=1e-2)
    print("LayerNorm implementation is correct.")
    print("Max absolute difference:", (ref - y).abs().max().item())