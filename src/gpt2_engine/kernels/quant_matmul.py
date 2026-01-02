import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8), 
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        # New configs for coalesced load optimization (larger K, smaller tiles)
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def quant_matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr, scales_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_scale, 
    # Meta
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Computes C = A x B
    A is (M, K) fp16
    B is (K, N) int8 (logically), stored as (N, K) column-major equivalent if stride_bk=1.
    scales is (N, ) fp16
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for A (M rows)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Offsets for B (N columns)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # Pointers
    # A: matches rows offs_am, cols offs_k starts at 0
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # B: Optimized Coalesced Load
    # We load B in shape (BLOCK_N, BLOCK_K) so that the inner dimension is K (stride 1).
    # Then we transpose it to (BLOCK_K, BLOCK_N) for calculation.
    # Note: stride_bn is the stride of N dimension (large stride K)
    #       stride_bk is the stride of K dimension (small stride 1)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    
    # Scales: matches cols offs_bn (per channel N)
    # Assumes scales stored contiguously [N]
    scales_ptrs = scales_ptr + offs_bn * stride_scale # stride usually 1
    
    # Load scales once (they don't depend on K)
    # scales shape: (BLOCK_N,)
    scales = tl.load(scales_ptrs, mask=offs_bn < N, other=0.0)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        
        # Load A (fp16)
        # We need mask logic if K is not multiple of BLOCK_SIZE_K?
        # Assuming K is multiple for now or minimal masking
        # For safety, mask K dims
        k_remaining = K - k * BLOCK_SIZE_K
        
        # Simplification: assume masked load if needed, but for blocks usually K is divisible
        # or we handle boundary.
        # To go fast, standard loop often assumes aligned, but we add masks
        load_mask = offs_k[None, :] < k_remaining
        
        a = tl.load(a_ptrs, mask=load_mask, other=0.0)
        
        # Load B (int8) - Coalesced
        # Shape loaded: (BLOCK_N, BLOCK_K)
        # Mask: offs_k in dim 1 < k_remaining. offs_bn in dim 0 < N (already implicitly handled by layout? No, careful)
        # Actually offs_bn is handled by B alloc size usually, but let's be safe
        b_mask = (offs_bn[:, None] < N) & (offs_k[None, :] < k_remaining)
        b_int8 = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Transpose to (BLOCK_K, BLOCK_N)
        b_int8 = tl.trans(b_int8)
        
        # Dequantize
        # b_int8 (K, N) -> float16
        # scales (N) -> broadcast to (K, N) implicitly or explicitly
        b = b_int8.to(tl.float16) * scales[None, :]

        # Accumulate
        accumulator += tl.dot(a, b)
        
        # Advance
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk # Advance in K dim (dim 1 of loaded block)
        
    c = accumulator.to(tl.float16)
    
    # Store C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)

def quant_linear(x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor = None):
    """
    x: (B, M, K) or (M, K)
    w_int8: (N, K) - The actual tensor data
    scales: (N, )
    bias: (N, )
    """
    # Flatten x to (M, K)
    x_dim = x.ndim
    if x_dim == 3:
        b, m, k = x.shape
        x_reshaped = x.view(-1, k)
    else:
        x_reshaped = x
        b = 1
        m = x.shape[0] # Actually M
        
    M, K = x_reshaped.shape
    N = w_int8.shape[0] # w_int8 is (N, K)
    
    assert w_int8.shape[1] == K
    
    # Output buffer
    c = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    # Kernel Launch
    # We treat w_int8 as B matrix of shape (K, N) but transposed storage.
    # stride_bn = K (moving N by 1 means moving row of W, which is K elements)
    # stride_bk = 1 (moving K by 1 means moving col of W, which is 1 element)
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    quant_matmul_kernel[grid](
        x_reshaped, w_int8, c, scales,
        M, N, K,
        x_reshaped.stride(0), x_reshaped.stride(1),
        1, K, # stride_bk, stride_bn
        c.stride(0), c.stride(1),
        scales.stride(0),
    )
    
    if bias is not None:
        c += bias
        
    if x_dim == 3:
        c = c.view(b, m, N)
        
    return c
