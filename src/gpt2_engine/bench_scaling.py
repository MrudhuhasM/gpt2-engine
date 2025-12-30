import torch
import torch.nn.functional as F
from gpt2_engine.kernels.attention import fused_attention_forward

def bench(fn, warmup=25, iters=100):
    # simple CUDA timing using events
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starter.record()
    for _ in range(iters):
        fn()
    ender.record()
    torch.cuda.synchronize()

    ms = starter.elapsed_time(ender) / iters
    return ms

def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = "cuda"
    seq_lens = [1024, 2048, 4096, 8192]
    batch_size = 1
    n_heads = 12
    head_dim = 64
    
    print(f"{'Seq Len':<10} | {'PyTorch (ms)':<15} | {'Triton (ms)':<15} | {'Speedup':<10}")
    print("-" * 60)

    for seq_len in seq_lens:
        try:
            q = torch.randn((batch_size, n_heads, seq_len, head_dim), device=device, dtype=torch.bfloat16)
            k = torch.randn((batch_size, n_heads, seq_len, head_dim), device=device, dtype=torch.bfloat16)
            v = torch.randn((batch_size, n_heads, seq_len, head_dim), device=device, dtype=torch.bfloat16)
            sm_scale = 1.0 / (head_dim ** 0.5)

            def torch_attention():
                return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

            def triton_attention():
                return fused_attention_forward(q, k, v, sm_scale)

            ms_torch = bench(torch_attention)
            ms_triton = bench(triton_attention)
            speedup = ms_torch / ms_triton

            print(f"{seq_len:<10} | {ms_torch:<15.4f} | {ms_triton:<15.4f} | {speedup:.2f}x")
        except RuntimeError as e:
             print(f"{seq_len:<10} | {'OOM/Error':<15} | {'-':<15} | -")

if __name__ == "__main__":
    main()
