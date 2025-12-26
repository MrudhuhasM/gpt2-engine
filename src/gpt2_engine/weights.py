import torch
from transformers import GPT2LMHeadModel as HFGPT2LMHeadModel

from gpt2_engine.utils import GPT2Config
from gpt2_engine.model import GPT2LMHEadModel

def load_hf_weights_to_model(local_model: GPT2LMHEadModel, hf_model: HFGPT2LMHeadModel):
    my_sd = local_model.state_dict()
    hf_sd = hf_model.state_dict()

    my_sd['transformer.wte.weight'] = hf_sd['transformer.wte.weight']
    my_sd['transformer.wpe.weight'] = hf_sd['transformer.wpe.weight']
    my_sd['transformer.ln_f.weight'] = hf_sd['transformer.ln_f.weight']
    my_sd['transformer.ln_f.bias'] = hf_sd['transformer.ln_f.bias']

    my_sd['lm_head.weight'] = hf_sd['lm_head.weight']

    for i in range(local_model.cfg.n_layer):
        my_sd[f'transformer.h.{i}.ln_1.weight'] = hf_sd[f'transformer.h.{i}.ln_1.weight']
        my_sd[f'transformer.h.{i}.ln_1.bias'] = hf_sd[f'transformer.h.{i}.ln_1.bias']
        my_sd[f'transformer.h.{i}.ln_2.weight'] = hf_sd[f'transformer.h.{i}.ln_2.weight']
        my_sd[f'transformer.h.{i}.ln_2.bias'] = hf_sd[f'transformer.h.{i}.ln_2.bias']

        my_sd[f'transformer.h.{i}.attn.c_attn.weight'] = hf_sd[f'transformer.h.{i}.attn.c_attn.weight'].t()
        my_sd[f'transformer.h.{i}.attn.c_attn.bias'] = hf_sd[f'transformer.h.{i}.attn.c_attn.bias']
        my_sd[f'transformer.h.{i}.attn.c_proj.weight'] = hf_sd[f'transformer.h.{i}.attn.c_proj.weight'].t()
        my_sd[f'transformer.h.{i}.attn.c_proj.bias'] = hf_sd[f'transformer.h.{i}.attn.c_proj.bias']

        my_sd[f'transformer.h.{i}.mlp.c_fc.weight'] = hf_sd[f'transformer.h.{i}.mlp.c_fc.weight'].t()
        my_sd[f'transformer.h.{i}.mlp.c_fc.bias'] = hf_sd[f'transformer.h.{i}.mlp.c_fc.bias']
        my_sd[f'transformer.h.{i}.mlp.c_proj.weight'] = hf_sd[f'transformer.h.{i}.mlp.c_proj.weight'].t()
        my_sd[f'transformer.h.{i}.mlp.c_proj.bias'] = hf_sd[f'transformer.h.{i}.mlp.c_proj.bias']

    local_model.load_state_dict(my_sd, strict=True)

def build_and_load(model_name: str = "gpt2", device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    hf = HFGPT2LMHeadModel.from_pretrained(model_name)
    hf.eval()

    cfg = GPT2Config(
        vocab_size=hf.config.vocab_size,
        n_positions=hf.config.n_positions,
        n_embd=hf.config.n_embd,
        n_layer=hf.config.n_layer,
        n_head=hf.config.n_head,
        layer_norm_epsilon=hf.config.layer_norm_epsilon,
        dropout=0.0,
    )

    my_model = GPT2LMHEadModel(cfg)
    my_model.eval()

    load_hf_weights_to_model(my_model, hf)

    my_model = my_model.to(device=device, dtype=dtype)
    hf = hf.to(device=device, dtype=dtype)
    return my_model, hf


if __name__ == "__main__":
    my, hf = build_and_load("gpt2", device="cuda", dtype=torch.bfloat16)
    print("Models built and weights loaded successfully.")

