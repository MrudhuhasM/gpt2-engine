import time
import torch
from transformers import GPT2TokenizerFast

from gpt2_engine.weights import build_and_load
from gpt2_engine.utils import top_k_sample, set_seed


@torch.no_grad()
def generate_tokens(model, input_ids, new_tokens: int, use_cache: bool = True, k: int = 50):
    model.eval()
    pkv = None

    # warm start: process prompt
    if use_cache:
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

    # no cache: re-run whole prefix each step (slow baseline)
    seq = input_ids
    for _ in range(new_tokens):
        logits, _ = model(seq, past_key_values=None)
        next_token = top_k_sample(logits[:, -1, :], k=k)
        seq = torch.cat([seq, next_token], dim=1)
    return seq


def measure_tps(fn, warmup=10, iters=100):
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

    my_model, hf_model = build_and_load(device=device, dtype=dtype)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt = "Alan Turing was a"
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

    gen_tokens = 20
    iters = 100
    warmup = 10

    with open("benchmark_results.txt", "w") as f:
        # 1) My model eager
        def run_my_eager():
            generate_tokens(my_model, input_ids, new_tokens=gen_tokens, use_cache=True, k=50)

        dt = measure_tps(run_my_eager, warmup=warmup, iters=iters)
        tps = gen_tokens / dt
        result = f"[MyModel Eager]  avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}"
        print(result)
        f.write(result + "\n")

        # 2) My model torch.compile
        compiled = torch.compile(my_model, mode="reduce-overhead")

        def run_my_compiled():
            generate_tokens(compiled, input_ids, new_tokens=gen_tokens, use_cache=True, k=50)

        dt = measure_tps(run_my_compiled, warmup=warmup, iters=iters)
        tps = gen_tokens / dt
        result = f"[MyModel Compile] avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}"
        print(result)
        f.write(result + "\n")

        # 3) HF model baseline (also uses cache internally if you call use_cache=True,
        # but simplest is just use forward in a similar way via our generate wrapper)
        def hf_forward(input_ids, past_key_values=None):
            out = hf_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
            return out.logits, out.past_key_values

        def run_hf():
            # small wrapper matching our (logits, pkv) format
            pkv = None
            logits, pkv = hf_forward(input_ids, None)
            next_logits = logits[:, -1, :]
            cur = top_k_sample(next_logits, k=50)
            for _ in range(gen_tokens - 1):
                logits, pkv = hf_forward(cur, pkv)
                cur = top_k_sample(logits[:, -1, :], k=50)

        dt = measure_tps(run_hf, warmup=warmup, iters=iters)
        tps = gen_tokens / dt
        result = f"[HF GPT2]        avg step: {dt*1000:.2f} ms | TPS: {tps:.2f}"
        print(result)
        f.write(result + "\n")


if __name__ == "__main__":
    main()
