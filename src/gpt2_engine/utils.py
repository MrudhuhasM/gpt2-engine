import torch
from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_ctx: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_triton: bool = True


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def top_k_sample(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:

    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    if temperature <= 0.0:
        raise ValueError("temperature must be a positive float.")
    
    if temperature != 1.0:
        logits = logits / temperature

    if k is not None and k > 0:
        values, index = torch.topk(logits, k)
        probs = torch.softmax(values, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        next_token = index.gather(-1, next_idx)
        return next_token
    
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token