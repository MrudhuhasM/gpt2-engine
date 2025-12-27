import time
import torch
import torch.nn.functional as F

from gpt2_engine.kernels.gelu import gelu_forward
from gpt2_engine.kernels.layer_norm import layer_norm_forward


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

    # ---------------- GELU ----------------
    N = 4 * 1024 * 1024
    x = torch.randn(N, device=device, dtype=torch.bfloat16)

    def torch_gelu():
        return F.gelu(x, approximate="tanh")

    def triton_gelu():
        return gelu_forward(x)

    ms_torch = bench(torch_gelu)
    ms_triton = bench(triton_gelu)

    bytes_total = x.numel() * x.element_size()
    gb = bytes_total / (1024**3)

    # 2x for read + write
    torch_gbps = (2 * gb) / (ms_torch / 1e3)
    triton_gbps = (2 * gb) / (ms_triton / 1e3)

    print(f"GELU Benchmark (N={N:,} elements)")
    print(f"PyTorch: {ms_torch:.4f} ms  ({torch_gbps:.2f} GB/s)")
    print(f"Triton:  {ms_triton:.4f} ms  ({triton_gbps:.2f} GB/s)")
    print()

    # Write to benchmark_results.txt
    with open('/home/mrudhuhas/Documents/Projects/gpt2-engine/benchmark_results.txt', 'a') as f:
        f.write(f"GELU Kernel Benchmark (N={N:,} elements)\n")
        f.write(f"PyTorch: {ms_torch:.4f} ms  ({torch_gbps:.2f} GB/s)\n")
        f.write(f"Triton:  {ms_triton:.4f} ms  ({triton_gbps:.2f} GB/s)\n")
        f.write("\n")

    # ---------------- LayerNorm ----------------
    B, S, H = 16, 1024, 768
    x2 = torch.randn((B * S, H), device=device, dtype=torch.bfloat16)
    gamma = torch.ones((H,), device=device, dtype=torch.bfloat16)
    beta = torch.zeros((H,), device=device, dtype=torch.bfloat16)
    eps = 1e-5

    def torch_ln():
        return F.layer_norm(x2, (H,), gamma, beta, eps)

    def triton_ln():
        return layer_norm_forward(x2, gamma, beta, eps)

    ms_torch = bench(torch_ln)
    ms_triton = bench(triton_ln)

    print(f"LayerNorm Benchmark (B={B}, S={S}, H={H})")
    print(f"PyTorch: {ms_torch:.4f} ms")
    print(f"Triton:  {ms_triton:.4f} ms")

    # Write to benchmark_results.txt
    with open('/home/mrudhuhas/Documents/Projects/gpt2-engine/benchmark_results.txt', 'a') as f:
        f.write(f"LayerNorm Kernel Benchmark (B={B}, S={S}, H={H})\n")
        f.write(f"PyTorch: {ms_torch:.4f} ms\n")
        f.write(f"Triton:  {ms_triton:.4f} ms\n")


if __name__ == "__main__":
    main()
