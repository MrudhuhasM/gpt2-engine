import torch

from gpt2_engine.kernels.gelu import gelu_forward as triton_gelu
from gpt2_engine.kernels.layer_norm import layer_norm_forward as triton_ln
from gpt2_engine.kernels.softmax import softmax_forward as triton_softmax

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


def softmax(x: torch.Tensor, is_causal: bool = False, use_triton: bool = True) -> torch.Tensor:
    # x shape: (Batch, Heads, S, T)
    S, T = x.shape[-2], x.shape[-1]
    if use_triton and x.is_cuda and T >= 128:
        return triton_softmax(x, is_causal=is_causal)
    
    if is_causal:
        # Correct causal mask even for T > S
        # row i corresponds to global T-S+i
        # Mask if col j > global_row_idx
        row_idx = torch.arange(S, device=x.device).view(-1, 1) + (T - S)
        col_idx = torch.arange(T, device=x.device).view(1, -1)
        mask = col_idx > row_idx
        x = x.masked_fill(mask, float('-inf'))
        
    return torch.nn.functional.softmax(x, dim=-1)
