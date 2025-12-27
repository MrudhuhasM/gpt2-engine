import torch

from gpt2_engine.kernels.gelu import gelu_forward as triton_gelu
from gpt2_engine.kernels.layer_norm import layer_norm_forward as triton_ln

def gelu(x: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    # Use Triton only for prefill (large numel), fall back to eager for decode (small numel)
    if use_triton and x.is_cuda and x.numel() >= 1_000_000:
        return triton_gelu(x)
    return torch.nn.functional.gelu(x, approximate="tanh")


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    use_triton: bool = True,
) -> torch.Tensor:
    # Use Triton only for prefill (large M), fall back to eager for decode (M=1 or small)
    # Expected x shape: (M, H)
    M = x.shape[0] if x.dim() > 1 else 1
    if use_triton and x.is_cuda and M >= 256:
        return triton_ln(x, weight, bias, eps=eps)
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)