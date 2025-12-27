import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2_engine.utils import GPT2Config

from gpt2_engine.ops import gelu, layer_norm

PastKeyValue = Tuple[torch.Tensor, torch.Tensor]  # (key, value)


class Gpt2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embed = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.triu(torch.ones((config.n_positions, config.n_positions)), diagonal=1).bool()
        self.register_buffer("mask", mask, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        return x.view(batch_size, seq_len, num_heads * head_dim)  # (batch_size, seq_len, n_embd)
    def forward(
        self,
        x: torch.Tensor,  # shape: (batch_size, seq_len, n_embd)
        layer_past: Optional[PastKeyValue] = None,  # (key, value), each key/value: (batch_size, num_heads, seq_len_past, head_dim)
    ) -> Tuple[torch.Tensor, PastKeyValue]:
        batch_size, seq_len, _ = x.size()

        qkv = self.c_attn(x)  # (batch_size, seq_len, 3 * n_embd)
        q, k, v = qkv.split(self.n_embed, dim=2)  # each: (batch_size, seq_len, n_embd)

        q = self._split_heads(q)  # (batch_size, num_heads, seq_len, head_dim)
        k = self._split_heads(k)  # (batch_size, num_heads, seq_len, head_dim)
        v = self._split_heads(v)  # (batch_size, num_heads, seq_len, head_dim)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=2)  # (batch_size, num_heads, seq_len_past + seq_len, head_dim)
            v = torch.cat((past_v, v), dim=2)  # (batch_size, num_heads, seq_len_past + seq_len, head_dim)

        present = (k, v)

        Sk = k.size(2)  # seq_len_total

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len_total)
        
        causal = self.mask[Sk - seq_len:Sk, :Sk]  # (seq_len, seq_len_total)
        attn_scores = attn_scores.masked_fill(causal.view(1, 1, seq_len, Sk), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len_total)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = self._merge_heads(attn_output)  # (batch_size, seq_len, n_embd)
        attn_output = self.c_proj(attn_output)  # (batch_size, seq_len, n_embd)
        attn_output = self.dropout(attn_output)

        return attn_output, present

class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.cfg = config
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=True)
        self.act = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        # x = self.act(x)
        x = gelu(x, use_triton=self.cfg.use_triton)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.cfg = config
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = Gpt2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    def forward(
        self,
        x: torch.Tensor,  # shape: (batch_size, seq_len, n_embd)
        layer_past: Optional[PastKeyValue] = None,  # (key, value), each key/value: (batch_size, num_heads, seq_len_past, head_dim)
    ) -> Tuple[torch.Tensor, PastKeyValue]:
        # Attention block
        # a = self.ln_1(x)
        B,S,H = x.size()
        a2d = x.view(B*S, H)
        a2d = layer_norm(a2d, self.ln_1.weight, self.ln_1.bias, eps=self.ln_1.eps, use_triton=self.cfg.use_triton)
        a = a2d.view(B, S, H)
        attn_output, present = self.attn(a, layer_past=layer_past)
        x = x + attn_output

        # MLP block
        m2d = x.view(B*S, H)
        m2d = layer_norm(m2d, self.ln_2.weight, self.ln_2.bias, eps=self.ln_2.eps, use_triton=self.cfg.use_triton)
        m = m2d.view(B, S, H)
        # m = self.ln_2(x)
        mlp_output = self.mlp(m)
        x = x + mlp_output

        return x, present  


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.cfg = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])

    def forward(
        self,
        input_ids: torch.Tensor,  # shape: (batch_size, seq_len)
        past_key_values: Optional[List[PastKeyValue]] = None,  # list of (key, value), each key/value: (batch_size, num_heads, seq_len_past, head_dim)
    ) -> Tuple[torch.Tensor, List[PastKeyValue]]:
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        if past_key_values is None:
            past_key_values = [None] * self.cfg.n_layer
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(2)  # seq_len_past

        position_ids = torch.arange(
            past_length, past_length + seq_len, dtype=torch.long, device=device
        ).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(position_ids)
        x = self.drop(x)

        presents = []
        for block, layer_past in zip(self.h, past_key_values):
            x, present = block(x, layer_past)
            presents.append(present)

        B,S,H = x.size()
        x2d = x.view(B*S, H)
        x2d = layer_norm(x2d, self.ln_f.weight, self.ln_f.bias, eps=self.ln_f.eps, use_triton=self.cfg.use_triton)
        x = x2d.view(B, S, H)
        return x, presents  # x: (batch_size, seq_len, n_embd), presents: list of (key, value)



class GPT2LMHEadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.cfg = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # Weight tying

    def forward(
        self,
        input_ids: torch.Tensor,  # shape: (batch_size, seq_len)
        past_key_values: Optional[List[PastKeyValue]] = None,  # list of (key, value), each key/value: (batch_size, num_heads, seq_len_past, head_dim)
    ):
        hidden_states, past_key_values = self.transformer(  # hidden_states: (batch_size, seq_len, n_embd), past_key_values: updated list
            input_ids=input_ids, 
            past_key_values=past_key_values
        )
        logits = self.lm_head(hidden_states)  # logits: (batch_size, seq_len, vocab_size)
        return logits, past_key_values  

        