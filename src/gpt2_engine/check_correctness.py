import torch
from transformers import GPT2TokenizerFast

from gpt2_engine.weights import build_and_load

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


if __name__ == "__main__":
    main()

