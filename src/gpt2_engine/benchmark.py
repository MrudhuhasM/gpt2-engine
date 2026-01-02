import time
import torch
import os
from transformers import GPT2TokenizerFast

from gpt2_engine.weights import build_and_load
from gpt2_engine.utils import top_k_sample, set_seed
from gpt2_engine.quantize import replace_linear_with_quant

@torch.no_grad()
def generate_tokens(model, input_ids, new_tokens: int, use_cache: bool = True, k: int = 50):
    model.eval()
    pkv = None

    # warm start: process prompt
    logits, pkv = model(input_ids, past_key_values=None)
    if use_cache and pkv is not None:
        pkv = [[k.clone(), v.clone()] for k, v in pkv]

    next_logits = logits[:, -1, :]
    next_token = top_k_sample(next_logits, k=k)
    out = [input_ids, next_token]
    cur = next_token
    
    for _ in range(new_tokens - 1):
        logits, pkv = model(cur, past_key_values=pkv)
        if use_cache and pkv is not None:
            pkv = [[k.clone(), v.clone()] for k, v in pkv]

        next_logits = logits[:, -1, :]
        cur = top_k_sample(next_logits, k=k)
        out.append(cur)
    return torch.cat(out, dim=1)

def measure_tps(fn, warmup=5, iters=20):
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters

@torch.no_grad()
def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA required for benchmark")
        return

    print("Running End-to-End Benchmark (Generate)...")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt = "Alan Turing was a"
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    gen_tokens = 50 
    
    results = []
    
    # 1. PyTorch Eager FP16 (No Triton Ops)
    print("\nLoading Baseline (PyTorch Eager FP16)...")
    model_torch, _ = build_and_load("gpt2", device=device, dtype=torch.float16)
    # Disable Triton ops explicitly
    model_torch.cfg.use_triton = False
    
    def run_torch():
        generate_tokens(model_torch, input_ids, new_tokens=gen_tokens)
    
    print("Benchmarking PyTorch Eager...")
    dt_torch = measure_tps(run_torch)
    tps_torch = gen_tokens / dt_torch
    mem_torch = torch.cuda.max_memory_allocated() / 1024**3
    results.append(("PyTorch FP16 (Baseline)", tps_torch, mem_torch))
    
    del model_torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. Triton FP16 (Triton Ops)
    print("\nLoading Triton FP16...")
    model_triton, _ = build_and_load("gpt2", device=device, dtype=torch.float16)
    model_triton.cfg.use_triton = True
    
    def run_triton():
        generate_tokens(model_triton, input_ids, new_tokens=gen_tokens)
    
    print("Benchmarking Triton FP16...")
    dt_triton = measure_tps(run_triton)
    tps_triton = gen_tokens / dt_triton
    mem_triton = torch.cuda.max_memory_allocated() / 1024**3
    results.append(("Triton FP16 Ops", tps_triton, mem_triton))
    
    # Clean up
    del model_triton
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 3. W8A16 Quantized Model
    print("\nLoading W8A16 Model...")
    model_int8, _ = build_and_load("gpt2", device=device, dtype=torch.float16)
    model_int8.cfg.use_triton = True
    replace_linear_with_quant(model_int8)
    
    def run_int8():
        generate_tokens(model_int8, input_ids, new_tokens=gen_tokens)
        
    print("Benchmarking W8A16...")
    dt_int8 = measure_tps(run_int8)
    tps_int8 = gen_tokens / dt_int8
    mem_int8 = torch.cuda.max_memory_allocated() / 1024**3
    results.append(("Triton W8A16 (Quant)", tps_int8, mem_int8))
    
    # Display Results
    print("\n" + "="*80)
    print(f"{'Configuration':<30} | {'TPS':<10} | {'Peak VRAM (GB)':<15} | {'Speedup':<10}")
    print("-" * 80)
    
    baseline_tps = results[0][1]
    
    for name, tps, mem in results:
        speedup = tps / baseline_tps
        print(f"{name:<30} | {tps:<10.2f} | {mem:<15.2f} | {speedup:<10.2f}x")
    print("="*80)
    
    with open("benchmark_results.txt", "w") as f:
        f.write("End-to-End Generation Benchmark (TPS)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Configuration':<30} | {'TPS':<10} | {'Peak VRAM (GB)':<15} | {'Speedup':<10}\n")
        f.write("-" * 80 + "\n")
        for name, tps, mem in results:
            speedup = tps / baseline_tps
            f.write(f"{name:<30} | {tps:<10.2f} | {mem:<15.2f} | {speedup:<10.2f}x\n")

if __name__ == "__main__":
    main()
