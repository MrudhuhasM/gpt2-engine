
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def gelu_kernel(
    input_ptr, # pointer to input tensor
    output_ptr, # pointer to output tensor
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,  # block size for parallelism
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x_f = x.to(tl.float32)

    k0 = x_f + 0.044715 * x_f * x_f * x_f
    k1 = libdevice.tanh(0.7978845608028654 * k0)
    y = 0.5 * x_f * (1.0 + k1)
    y = y.to(x.dtype)
    tl.store(output_ptr + offsets, y, mask=mask)


def gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """Applies the GELU activation function using a custom Triton kernel."""
    assert x.is_cuda, "Input tensor must be on CUDA device."
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    # Simple test
    x = torch.randn((1,1024), device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.gelu(x, approximate='tanh')
    y = gelu_forward(x)
    
    print("Max absolute difference:", (ref - y).abs().max().item())
    assert torch.allclose(ref, y, atol=1e-2, rtol=1e-2), "GELU implementation is incorrect."
    print("GELU implementation is correct.")
