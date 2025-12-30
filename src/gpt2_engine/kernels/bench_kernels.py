import time
import torch
import torch.nn.functional as F

from gpt2_engine.kernels.gelu import gelu_forward
from gpt2_engine.kernels.layer_norm import layer_norm_forward
from gpt2_engine.kernels.softmax import softmax_forward
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
    device = "cuda"
    results = []

    # ---------------- GELU ----------------
    N = 4 * 1024 * 1024
    x = torch.randn(N, device=device, dtype=torch.bfloat16)

    def torch_gelu():
        return F.gelu(x, approximate="tanh")

    def triton_gelu():
        return gelu_forward(x)

    ms_torch = bench(torch_gelu)
    ms_triton = bench(triton_gelu)
    results.append(("GELU", ms_torch, ms_triton, f"N={N:,}"))

    print(f"GELU Benchmark (N={N:,} elements)")
    print(f"PyTorch: {ms_torch:.4f} ms")
    print(f"Triton:  {ms_triton:.4f} ms")
    print()

    # ---------------- LayerNorm ----------------
    B, S, H_ln = 16, 1024, 768
    x2 = torch.randn((B * S, H_ln), device=device, dtype=torch.bfloat16)
    gamma = torch.ones((H_ln,), device=device, dtype=torch.bfloat16)
    beta = torch.zeros((H_ln,), device=device, dtype=torch.bfloat16)
    eps = 1e-5

    def torch_ln():
        return F.layer_norm(x2, (H_ln,), gamma, beta, eps)

    def triton_ln():
        return layer_norm_forward(x2, gamma, beta, eps)

    ms_torch = bench(torch_ln)
    ms_triton = bench(triton_ln)
    results.append(("LayerNorm", ms_torch, ms_triton, f"B={B}, S={S}, H={H_ln}"))

    print(f"LayerNorm Benchmark (B={B}, S={S}, H={H_ln})")
    print(f"PyTorch: {ms_torch:.4f} ms")
    print(f"Triton:  {ms_triton:.4f} ms")
    print()

    # ---------------- Softmax ----------------
    B, H, S, T = 16, 12, 1024, 1024
    x3 = torch.randn((B, H, S, T), device=device, dtype=torch.bfloat16)
    mask = torch.triu(torch.ones((S, T), device=device), diagonal=1).bool()

    def torch_softmax():
        scores = x3.masked_fill(mask, float('-inf'))
        return F.softmax(scores, dim=-1)

    def triton_softmax():
        return softmax_forward(x3, is_causal=True)

    ms_torch = bench(torch_softmax)
    ms_triton = bench(triton_softmax)
    results.append(("Softmax", ms_torch, ms_triton, f"B={B}, H={H}, S={S}, T={T}"))

    print(f"Softmax Benchmark (B={B}, H={H}, S={S}, T={T})")
    print(f"PyTorch: {ms_torch:.4f} ms")
    print(f"Triton:  {ms_triton:.4f} ms")
    print()

    # ---------------- Attention ----------------
    B, H, S, D = 2, 12, 1024, 64
    q = torch.randn((B, H, S, D), device=device, dtype=torch.bfloat16)
    k = torch.randn((B, H, S, D), device=device, dtype=torch.bfloat16)
    v = torch.randn((B, H, S, D), device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / (D ** 0.5)

    def torch_attention():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)

    def triton_attention():
        return fused_attention_forward(q, k, v, sm_scale)

    ms_torch = bench(torch_attention)
    ms_triton = bench(triton_attention)
    results.append(("Attention", ms_torch, ms_triton, f"B={B}, H={H}, S={S}, D={D}"))

    print(f"Attention Benchmark (B={B}, H={H}, S={S}, D={D})")
    print(f"PyTorch (SDPA): {ms_torch:.4f} ms")
    print(f"Triton (Fused): {ms_triton:.4f} ms")
    print()

    # Write all results to benchmark_results.txt
    with open('benchmark_results.txt', 'w') as f:
        f.write("Triton Kernel Benchmarks\n")
        f.write("========================\n")
        f.write(f"{'Kernel':<15} | {'PyTorch (ms)':<15} | {'Triton (ms)':<15} | {'Configuration'}\n")
        f.write("-" * 80 + "\n")
        for name, ms_p, ms_t, config in results:
            f.write(f"{name:<15} | {ms_p:<15.4f} | {ms_t:<15.4f} | {config}\n")
        f.write("\n")
    
if __name__ == "__main__":
    main()
