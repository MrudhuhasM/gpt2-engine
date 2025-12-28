import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    stride_row,    # How much to jump to get to next row
    stride_col,    # Usually 1
    n_rows,        # Total number of rows
    n_cols,        # Seq_len_total (T)
    seq_len,       # Seq_len (S)
    BLOCK_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):

    

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    

    row_start_ptr = input_ptr + row_idx * stride_row
    input_ptrs = row_start_ptr + col_offsets * stride_col
    

    row = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    

    if is_causal:

        local_query_idx = row_idx % seq_len
        global_query_idx = local_query_idx + (n_cols - seq_len)
        causal_mask = col_offsets <= global_query_idx
        row = tl.where(causal_mask, row, -float('inf'))


    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    

    output_ptrs = output_ptr + row_idx * stride_row + col_offsets * stride_col
    tl.store(output_ptrs, output, mask=mask)

def softmax_forward(x: torch.Tensor, is_causal: bool = False):


    shape = x.shape
    seq_len = shape[-2]
    n_cols = shape[-1]
    

    x_2d = x.view(-1, n_cols)
    n_rows = x_2d.numel() // n_cols
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    assert n_cols <= 4096, "Seq len too large for simple kernel"
    
    out = torch.empty_like(x_2d)
    
    grid = (n_rows,)
    softmax_kernel[grid](
        x_2d, out,
        x_2d.stride(0), x_2d.stride(1),
        n_rows,
        n_cols,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        is_causal=is_causal
    )
    return out.view(*shape)

if __name__ == "__main__":

    S, T = 4, 4
    x = torch.randn((1, 1, S, T), device="cuda", dtype=torch.bfloat16) 
    

    mask = torch.triu(torch.ones((S, T), device="cuda"), diagonal=1).bool()
    ref_scores = x.masked_fill(mask, float('-inf'))
    ref_out = torch.nn.functional.softmax(ref_scores.to(torch.float32), dim=-1).to(torch.bfloat16)
    

    tri_out = softmax_forward(x, is_causal=True)
    
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)
    print("Causal Softmax implementation is correct.")


    S_dec, T_dec = 1, 4
    x_dec = torch.randn((1, 1, S_dec, T_dec), device="cuda", dtype=torch.bfloat16)

    tri_out_dec = softmax_forward(x_dec, is_causal=True)
    ref_out_dec = torch.nn.functional.softmax(x_dec.to(torch.float32), dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(ref_out_dec, tri_out_dec, atol=1e-2, rtol=1e-2)
    print("Decoding Case Softmax implementation is correct.")
