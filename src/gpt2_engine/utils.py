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
    quantized: bool = False



def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def top_k_sample(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """
    Sample from top-k logits.
    logits: (batch_size, vocab_size)
    """
    values, indices = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sample_indices = torch.multinomial(probs, num_samples=1)
    # Map back to original indices
    # sample_indices: (batch_size, 1) -> (batch_size, 1) indices into 'indices'
    return torch.gather(indices, -1, sample_indices)

