import torch
import torch.nn as nn
from gpt2_engine.quantize import FakeQuantLinear, quantize_weights, dequantize_weights

def test_quantization_accuracy():
    torch.manual_seed(0)
    in_features = 768
    out_features = 768 * 3  # Like c_attn
    
    linear = nn.Linear(in_features, out_features)
    fake_quant = FakeQuantLinear.from_linear(linear)
    
    x = torch.randn(1, 10, in_features)
    
    # Original output
    y_orig = linear(x)
    
    # Quantized output
    y_quant = fake_quant(x)
    
    diff = (y_orig - y_quant).abs()
    mse = diff.pow(2).mean()
    max_diff = diff.max()
    
    print(f"MSE: {mse.item():.6f}")
    print(f"Max Diff: {max_diff.item():.6f}")
    
    # Check if max diff is reasonable (int8 quantization roughly 1/127 precision relative to scale)
    # A heuristic check, since we don't have strict bounds for correctness without data distribution assumptions
    # But usually < 0.1 for standard normal inputs/weights is expected
    if max_diff < 0.2:
        print("PASS: Quantization error is within expected range.")
    else:
        print("FAIL: Quantization error is too high.")

if __name__ == "__main__":
    test_quantization_accuracy()

def test_triton_dequantization():
    if not torch.cuda.is_available():
        print("Skipping Triton test (CUDA not available)")
        return

    from gpt2_engine.kernels.dequantize import dequantize_triton
    
    torch.manual_seed(0)
    # create int8 data
    x_float = torch.randn(1024 * 10, device='cuda', dtype=torch.float16)
    scale = torch.ones_like(x_float) * 0.5
    
    # Quantize manually for test
    x_int8 = torch.round(x_float / scale).clamp(-127, 127).to(torch.int8)
    
    # Standard dequant
    y_ref = x_int8.to(torch.float16) * scale
    
    # Triton dequant
    y_tri = dequantize_triton(x_int8, scale)
    
    diff = (y_ref - y_tri).abs().max()
    print(f"Triton Dequant Max Diff: {diff.item()}")
    
    if diff < 1e-3:
        print("PASS: Triton dequantization matches.")
    else:
        print("FAIL: Triton dequantization mismatch.")

if __name__ == "__main__":
    test_quantization_accuracy()
    test_triton_dequantization()

def test_quant_linear():
    if not torch.cuda.is_available():
        print("Skipping QuantLinear test (CUDA not available)")
        return
    import torch.nn.functional as F
    from gpt2_engine.kernels.quant_matmul import quant_linear
    from gpt2_engine.quantize import quantize_weights
    
    torch.manual_seed(0)
    M, K, N = 32, 768, 768*3
    x = torch.randn(M, K, device='cuda', dtype=torch.float16)
    w = torch.randn(N, K, device='cuda', dtype=torch.float16)
    
    # Quantize W
    w_int8, scales = quantize_weights(w) # scales (N, 1)
    
    # Needs scales as (N,)
    scales_1d = scales.view(-1)
    
    # Triton Matmul
    y_tri = quant_linear(x, w_int8, scales_1d)
    
    # Reference
    # Dequantize first
    w_dq = w_int8.to(torch.float16) * scales
    y_ref = F.linear(x, w_dq)
    
    diff = (y_tri - y_ref).abs().max()
    print(f"QuantLinear Max Diff: {diff.item()}")

    # Compare with original float weight linear (just for reference)
    y_orig = F.linear(x, w)
    diff_orig = (y_tri - y_orig).abs().max()
    print(f"QuantLinear vs Float Linear Max Diff: {diff_orig.item()}")
    
    if diff < 0.2:
        print("PASS: QuantLinear matches dequantized reference.")
    else:
        print("FAIL: QuantLinear mismatch.")

if __name__ == "__main__":
    # Remove previous calls if re-running or append
    pass
    # I'll rely on the previous main block which called test_quantization_accuracy and test_triton_dequantization
    # I should have just appended the call to main, but I can't edit the MAIN block easily without replacing file.
    # So I will just invoke the new function in a new main block which overrides the previous one? 
    # Python doesn't work like that.
    # I will replace the Main block or just call the function.
    test_quant_linear()

def test_quant_linear_module():
    if not torch.cuda.is_available(): return
    from gpt2_engine.quantize import QuantLinear
    
    torch.manual_seed(0)
    M, K, N = 32, 768, 768
    x = torch.randn(M, K, device='cuda', dtype=torch.float16)
    
    # Standard Linear
    linear = nn.Linear(K, N).to('cuda').to(torch.float16)
    
    # Quant Linear
    q_linear = QuantLinear.from_linear(linear).to('cuda')

    
    # Forward
    y_ref = linear(x)
    y_q = q_linear(x)
    
    diff = (y_ref - y_q).abs().max()
    print(f"QuantLinear Module Max Diff: {diff.item()}")

    # Relaxed check for int8 quantization vs float16 linear
    if diff < 0.2:
        print("PASS: QuantLinear Module works.")
    else:
        print("FAIL: QuantLinear Module error too high.")

if __name__ == "__main__":
    test_quant_linear_module()
