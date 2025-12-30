import time
import torch
import os
from transformers import GPT2TokenizerFast

from gpt2_engine.weights import build_and_load
from gpt2_engine.utils import top_k_sample, set_seed


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

@torch.no_grad()
def hf_generate_tokens(model, input_ids, new_tokens: int, k: int = 50):
    # HF model has different pkv structure (tuple of tuples)
    model.eval()
    pkv = None
    
    res = model(input_ids, past_key_values=None, use_cache=True)
    logits, pkv = res.logits, res.past_key_values
    
    next_logits = logits[:, -1, :]
    next_token = top_k_sample(next_logits, k=k)
    out = [input_ids, next_token]
    cur = next_token
    
    for _ in range(new_tokens - 1):
        res = model(cur, past_key_values=pkv, use_cache=True)
        logits, pkv = res.logits, res.past_key_values
        next_logits = logits[:, -1, :]
        cur = top_k_sample(next_logits, k=k)
        out.append(cur)
    return torch.cat(out, dim=1)


def measure_tps(fn, warmup=10, iters=50):
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
    assert torch.cuda.is_available(), "Benchmark expects CUDA."

    device = "cuda"
    dtype = torch.bfloat16

    my_model, hf_model = build_and_load("gpt2", device=device, dtype=dtype)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "Alan Turing was a"
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

    gen_tokens = 20
    iters = 20
    warmup = 5

    results = []

    # 1) My model eager (No Triton)
    my_model.cfg.use_triton = False
    def run_my_no_triton():
        generate_tokens(my_model, input_ids, new_tokens=gen_tokens, use_cache=True, k=50)

    dt = measure_tps(run_my_no_triton, warmup=warmup, iters=iters)
    tps = gen_tokens / dt
    results.append(("MyModel (Eager)", dt * 1000, tps))
    print(f"[MyModel (Eager)] avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}")

    # 2) My model eager (Triton)
    my_model.cfg.use_triton = True
    def run_my_triton():
        generate_tokens(my_model, input_ids, new_tokens=gen_tokens, use_cache=True, k=50)

    dt = measure_tps(run_my_triton, warmup=warmup, iters=iters)
    tps = gen_tokens / dt
    results.append(("MyModel (Triton)", dt * 1000, tps))
    print(f"[MyModel (Triton)] avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}")

    # 3) My model compile (No Triton)
    my_model.cfg.use_triton = False
    compiled_no_tri = torch.compile(my_model)
    def run_my_compiled_no_tri():
        generate_tokens(compiled_no_tri, input_ids, new_tokens=gen_tokens, use_cache=True, k=50)

    dt = measure_tps(run_my_compiled_no_tri, warmup=warmup, iters=iters)
    tps = gen_tokens / dt
    results.append(("MyModel (Compile + No Triton)", dt * 1000, tps))
    print(f"[MyModel (Compile + No Triton)] avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}")

    # 4) My model compile (Triton)
    my_model.cfg.use_triton = True
    compiled_tri = torch.compile(my_model)
    def run_my_compiled_tri():
        generate_tokens(compiled_tri, input_ids, new_tokens=gen_tokens, use_cache=True, k=50)

    dt = measure_tps(run_my_compiled_tri, warmup=warmup, iters=iters)
    tps = gen_tokens / dt
    results.append(("MyModel (Compile + Triton)", dt * 1000, tps))
    print(f"[MyModel (Compile + Triton)] avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}")

    # 5) HF Model
    def run_hf():
        hf_generate_tokens(hf_model, input_ids, new_tokens=gen_tokens, k=50)
        
    dt = measure_tps(run_hf, warmup=warmup, iters=iters)
    tps = gen_tokens / dt
    results.append(("HF GPT2", dt * 1000, tps))
    print(f"[HF GPT2] avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}")

    # Write to benchmark_results.txt
    # We overwrite the end-to-end section or just append and we'll manually clean if needed, 
    # but let's try to overwrite the file properly this time by reading kernel section first.
    
    kernel_results = ""
    if os.path.exists("benchmark_results.txt"):
        with open("benchmark_results.txt", "r") as f:
            content = f.read()
            if "End-to-End Generation Benchmarks" in content:
                kernel_results = content.split("End-to-End Generation Benchmarks")[0]
            else:
                kernel_results = content

    with open("benchmark_results.txt", "w") as f:
        f.write(kernel_results.strip() + "\n\n")
        f.write("End-to-End Generation Benchmarks (GPT-2 Small)\n")
        f.write("==============================================\n")
        f.write(f"{'Mode':<30} | {'Avg Step (ms)':<15} | {'Tokens/Sec'}\n")
        f.write("-" * 65 + "\n")
        for mode, ms, tps in results:
            f.write(f"{mode:<30} | {ms:<15.2f} | {tps:.2f}\n")
        f.write("\n")

if __name__ == "__main__":
    main()
