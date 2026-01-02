import torch
from transformers import GPT2LMHeadModel as HFGPT2LMHeadModel

from gpt2_engine.utils import GPT2Config
from gpt2_engine.model import GPT2LMHEadModel
from gpt2_engine.quantize import quantize_weights, QuantLinear

def load_hf_weights_to_model(local_model: GPT2LMHEadModel, hf_model: HFGPT2LMHeadModel):
    # If quantize is True, we need to handle loading differently for QuantLinear
    # Because state_dict expects different keys (weight/scales/bias) vs (weight/bias)
    
    # We'll load directly component by component
    
    hf_sd = hf_model.state_dict()
    
    # Helper to load standard parameters
    def copy_param(local_name, hf_name, transpose=False):
        # Handle quantization if target is QuantLinear
        pass 
        # Actually simplest to let load_state_dict handle common parts and manually handle QuantLinear
    
    # Common non-linear parts
    with torch.no_grad():
        local_model.transformer.wte.weight.copy_(hf_sd['transformer.wte.weight'])
        local_model.transformer.wpe.weight.copy_(hf_sd['transformer.wpe.weight'])
        local_model.transformer.ln_f.weight.copy_(hf_sd['transformer.ln_f.weight'])
        local_model.transformer.ln_f.bias.copy_(hf_sd['transformer.ln_f.bias'])
        
        # Tie weights
        local_model.lm_head.weight = local_model.transformer.wte.weight

        for i in range(local_model.cfg.n_layer):
            block = local_model.transformer.h[i]
            
            # Layer Norms
            block.ln_1.weight.copy_(hf_sd[f'transformer.h.{i}.ln_1.weight'])
            block.ln_1.bias.copy_(hf_sd[f'transformer.h.{i}.ln_1.bias'])
            block.ln_2.weight.copy_(hf_sd[f'transformer.h.{i}.ln_2.weight'])
            block.ln_2.bias.copy_(hf_sd[f'transformer.h.{i}.ln_2.bias'])
            
            # Attention
            copy_linear(block.attn.c_attn, hf_sd[f'transformer.h.{i}.attn.c_attn.weight'], hf_sd[f'transformer.h.{i}.attn.c_attn.bias'])
            copy_linear(block.attn.c_proj, hf_sd[f'transformer.h.{i}.attn.c_proj.weight'], hf_sd[f'transformer.h.{i}.attn.c_proj.bias'])
            
            # MLP
            copy_linear(block.mlp.c_fc, hf_sd[f'transformer.h.{i}.mlp.c_fc.weight'], hf_sd[f'transformer.h.{i}.mlp.c_fc.bias'])
            copy_linear(block.mlp.c_proj, hf_sd[f'transformer.h.{i}.mlp.c_proj.weight'], hf_sd[f'transformer.h.{i}.mlp.c_proj.bias'])

def copy_linear(module, hf_weight, hf_bias):
    # HF weights for GPT-2 are Conv1D (in_features, out_features) which is Transposed relative to Linear (out, in)
    # BUT wait: HF GPT2 Conv1D weight is (d_model, 3*d_model). Linear weight is (3*d_model, d_model).
    # So hf_weight needs transpose typically.
    
    # Check if module is QuantLinear
    if isinstance(module, QuantLinear):
        # Quantize on the fly
        # HF weight is (in, out). We need (out, in) for quantize_weights input
        w_fp = hf_weight.t() 
        w_int8, scales = quantize_weights(w_fp)
        
        module.weight.copy_(w_int8)
        module.scales.copy_(scales.view(-1))
        
        if module.bias is not None:
             module.bias.copy_(hf_bias)
    else:
        # Standard Linear
        module.weight.copy_(hf_weight.t())
        if module.bias is not None:
            module.bias.copy_(hf_bias)


def build_and_load(model_name: str = "gpt2", device: str = "cuda", dtype: torch.dtype = torch.bfloat16, quantize: bool = False):
    hf = HFGPT2LMHeadModel.from_pretrained(model_name)
    hf.eval()

    # Move HF model to CPU to save GPU memory during conversion if possible, 
    # but quantize_weights might need GPU if implemented with torch ops on GPU.
    # We'll keep it on CPU or move as needed. Hf default is CPU.
    
    cfg = GPT2Config(
        vocab_size=hf.config.vocab_size,
        n_positions=hf.config.n_positions,
        n_embd=hf.config.n_embd,
        n_layer=hf.config.n_layer,
        n_head=hf.config.n_head,
        layer_norm_epsilon=hf.config.layer_norm_epsilon,
        dropout=0.0,
        quantized=quantize
    )

    my_model = GPT2LMHEadModel(cfg)
    my_model.eval()

    load_hf_weights_to_model(my_model, hf)

    my_model = my_model.to(device=device)
    # Cast non-quantized parts to dtype
    # QuantLinear parts: weight is int8 (stays), scales is fp16 (stays), bias fp16 (stays)
    # We should ensure everything matches 'dtype' except int8 weights.
    
    # Helper to cast float params
    for name, param in my_model.named_parameters():
        if param.dtype != torch.int8: # Don't touch quantized weights (though they are buffers)
             param.data = param.data.to(dtype=dtype)
             
    for name, buf in my_model.named_buffers():
         if buf.dtype != torch.int8:
             buf.data = buf.data.to(dtype=dtype)
             
    # Cleanup HF
    del hf
    torch.cuda.empty_cache()
    
    return my_model, None


if __name__ == "__main__":
    my, _ = build_and_load("gpt2", device="cuda", dtype=torch.float16, quantize=True)
    print("Models built and weights loaded successfully.")
