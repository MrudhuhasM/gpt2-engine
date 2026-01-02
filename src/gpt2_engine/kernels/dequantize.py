import torch
import triton
import triton.language as tl

@triton.jit
def dequantize_kernel(
    ptr_input,  # int8 pointer
    ptr_output, # fp16 pointer
    ptr_scale,  # fp16 pointer (scales)
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load int8 data 
    # Use standard load, Triton handles packed reads if contiguous
    data_int8 = tl.load(ptr_input + offsets, mask=mask)
    
    # Cast to fp16
    data_fp16 = data_int8.to(tl.float16)
    
    # Load scale (broadcast or per-element)
    # For this simple kernel, assume ptr_scale points to a single value or matches 
    # Let's implement assuming scale is per-element for simplicity of verification, 
    # or scalar. Let's do scalar broadcast for now as a warm up, or 
    # match the offsets if ptr_scale is an array. 
    # Let's assume ptr_scale has same shape as input for this "copy" kernel.
    scale = tl.load(ptr_scale + offsets, mask=mask)
    
    output = data_fp16 * scale
    
    tl.store(ptr_output + offsets, output, mask=mask)

def dequantize_triton(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize tensor using Triton kernel.
    Assumes x_int8 and scale are flattened or compatible.
    """
    assert x_int8.is_contiguous()
    assert scale.is_contiguous()
    
    output = torch.empty_like(x_int8, dtype=torch.float16)
    n_elements = x_int8.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    dequantize_kernel[grid](
        x_int8, output, scale, n_elements,
        BLOCK_SIZE=1024
    )
    
    return output
