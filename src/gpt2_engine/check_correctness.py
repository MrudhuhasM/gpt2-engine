import torch
from transformers import GPT2TokenizerFast

from gpt2_engine.weights import build_and_load
from gpt2_engine.kernels.attention import fused_attention_forward

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    my_model, hf_model = build_and_load("gpt2", device=device, dtype=dtype)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    text = "Nikola tesla was a"
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device=device)

    hf_out = hf_model(input_ids=input_ids).logits
    my_out, _ = my_model(input_ids=input_ids)

    diff = (hf_out - my_out).abs().max().item()
    print(f"Max absolute difference between HF and local model outputs: {diff}")

    full_logits,_ = my_model(input_ids=input_ids)

    prefix = input_ids[:, :-1]
    last_token = input_ids[:, -1:]

    _, past_key_values = my_model(prefix)
    step_logits, _ = my_model(last_token, past_key_values=past_key_values)

    diff_cache = (full_logits[:, -1, :] - step_logits[:, -1, :]).abs().max().item()
    print(f"Max absolute difference between full sequence and cached step outputs: {diff_cache}")

    if dtype == torch.float32:
        print("Target diff less than 1e-4")
    else:
        print("Target diff less than 1e-2")

    # Decoding verification (KV Cache)
    print("\nRunning Decoding Verification (KV Cache)...")
    # Simulate decoding step: 1 Query token, 100 Past keys
    q = torch.randn(1, 12, 1, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 12, 100, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 12, 100, 64, device=device, dtype=torch.bfloat16)

    # Run yours
    # Note: user instruction mentioned 0.1, we use it directly.
    # In real usage it would be 1.0 / math.sqrt(64) = 0.125
    out_triton = fused_attention_forward(q, k, v, 0.1)

    # Run Reference (SDPA)
    out_torch = torch.nn.functional.scaled_dot_product_attention(
       q, k, v, is_causal=False, scale=0.1 # False because we attend to all 100 past tokens
    )

    # Compare
    assert torch.allclose(out_triton, out_torch, atol=1e-2)
    print("Decoding Verification PASSED!")

if __name__ == "__main__":
    main()

