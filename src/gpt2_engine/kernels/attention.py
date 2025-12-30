import torch
import triton
import triton.language as tl

@triton.jit
def _fused_attention_kernel(
    Q, K, V, Out,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_on,
    sm_scale,
    seq_len_q,
    seq_len_k,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    
    q_ptrs = Q + (off_hz * stride_qm) + (offs_m[:, None] * stride_qk) + (offs_k[None, :] * 1)

    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + (off_hz * stride_kn) + (offs_n[None, :] * stride_kk) + (offs_k[:, None] * 1)
    v_ptrs = V + (off_hz * stride_vn) + (offs_n[:, None] * stride_vk) + (offs_k[None, :] * 1)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len_q, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    
    off_causal = seq_len_k - seq_len_q
    
    
    end_n = tl.minimum(seq_len_k, (start_m + 1) * BLOCK_M + off_causal)
    
    for start_n in range(0, end_n, BLOCK_N):
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < seq_len_k, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seq_len_k, other=0.0)
        
        qk = tl.dot(q, k)
        qk *= sm_scale
        

        col_idx = start_n + offs_n[None, :]
        row_seq_idx = offs_m[:, None] + off_causal
        
        mask = (col_idx < seq_len_k) & (row_seq_idx >= col_idx)
        qk = tl.where(mask, qk, float("-inf"))
        
        m_j = tl.max(qk, 1)
        p = tl.exp(qk - m_j[:, None])
        l_j = tl.sum(p, 1)
        
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_j - m_new)
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v) * beta[:, None]
        
        l_i = l_i * alpha + l_j * beta
        m_i = m_new
        
        k_ptrs += BLOCK_N * stride_kk 
        v_ptrs += BLOCK_N * stride_vk 
        
    acc = acc / l_i[:, None]
    
    out_ptrs = Out + (off_hz * stride_om) + (offs_m[:, None] * stride_on) + (offs_k[None, :] * 1)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len_q)

def fused_attention_forward(q, k, v, sm_scale):
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    
    q = q.view(-1, Sq, D)
    k = k.view(-1, Sk, D)
    v = v.view(-1, Sk, D)
    
    out = torch.empty_like(q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(Sq, BLOCK_M), q.shape[0])
    
    _fused_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        sm_scale,
        Sq,
        Sk,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=2,
    )
    
    return out.view(B, H, Sq, D)
