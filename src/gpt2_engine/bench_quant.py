import torch
import time
from gpt2_engine.weights import build_and_load
from gpt2_engine.quantize import replace_linear_with_quant
from gpt2_engine.utils import set_seed
import copy

def get_model_size_mb(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

@torch.no_grad()
def benchmark_model(model, inputs, name="Model", runs=50):
    # Warmup
    for _ in range(5):
        model(inputs)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(runs):
        model(inputs)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = ((end - start) / runs) * 1000
    return avg_time_ms

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, skipping.")
        return

    print("Loading Baseline Model (FP16)...")
    # Load model
    model_fp16, _ = build_and_load("gpt2", device=device, dtype=torch.float16)
    model_fp16.eval()
    
    # Calculate FP16 Stats
    size_fp16 = get_model_size_mb(model_fp16)
    
    # Create input
    dummy_input = torch.randint(0, 50257, (1, 128)).to(device)
    
    print("Benchmarking FP16...")
    time_fp16 = benchmark_model(model_fp16, dummy_input, "FP16")
    
    # Quantize
    print("Quantizing to W8A16...")
    # We can quantize in place, but conceptually we are creating a new "version"
    replace_linear_with_quant(model_fp16)
    model_int8 = model_fp16 # It's modified in place
    
    # Calculate INT8 Stats
    size_int8 = get_model_size_mb(model_int8)
    
    print("Benchmarking W8A16...")
    time_int8 = benchmark_model(model_int8, dummy_input, "W8A16")
    
    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'FP16 (Baseline)':<15} | {'W8A16 (Quantized)':<15} | {'Diff':<10}")
    print("-" * 60)
    
    # Size
    diff_size = size_fp16 - size_int8
    diff_size_pct = (diff_size / size_fp16) * 100
    print(f"{'Model Size (MB)':<20} | {size_fp16:<15.2f} | {size_int8:<15.2f} | -{diff_size:.2f} ({diff_size_pct:.1f}%)")
    
    # Speed
    speedup = time_fp16 / time_int8
    print(f"{'Inference (ms)':<20} | {time_fp16:<15.2f} | {time_int8:<15.2f} | {speedup:.2f}x")
    
    print("="*60)
    print("\nConfiguration:")
    print(f"- Batch Size: 1")
    print(f"- Seq Length: 128")
    print(f"- Device: {torch.cuda.get_device_name(0)}")
    print(f"- Precision: W8A16 (Weights int8, Activations fp16)")

    # Save to file
    with open("bench_quant_results.txt", "w") as f:
        f.write("GPT-2 W8A16 Quantization Benchmark Results\n")
        f.write("="*60 + "\n")
        header = f"{'Metric':<20} | {'FP16 (Baseline)':<15} | {'W8A16 (Quantized)':<15} | {'Diff':<10}\n"
        f.write(header)
        f.write("-" * 60 + "\n")
        
        f.write(f"{'Model Size (MB)':<20} | {size_fp16:<15.2f} | {size_int8:<15.2f} | -{diff_size:.2f} ({diff_size_pct:.1f}%)\n")
        f.write(f"{'Inference (ms)':<20} | {time_fp16:<15.2f} | {time_int8:<15.2f} | {speedup:.2f}x\n")
        
        f.write("="*60 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"- Batch Size: 1\n")
        f.write(f"- Seq Length: 128\n")
        f.write(f"- Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"- Precision: W8A16 (Weights int8, Activations fp16)\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nResults saved to bench_quant_results.txt")


if __name__ == "__main__":
    main()
