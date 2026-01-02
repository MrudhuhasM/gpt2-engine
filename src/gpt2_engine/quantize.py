import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_weights(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to int8 using symmetric per-channel quantization.
    
    Args:
        w: Floating point weights of shape (out_features, in_features)
        
    Returns:
        w_int8: Quantized weights (int8) of shape (out_features, in_features)
        scales: Scaling factors of shape (out_features, 1)
    """
    # Find max absolute value per row (per output channel)
    # shape: (out_features, 1)
    max_val = w.abs().amax(dim=1, keepdim=True)
    
    # Calculate scale
    # We use 127 as the max value for int8
    # Avoid division by zero by clamping min value
    scale = max_val / 127.0
    scale = torch.clamp(scale, min=1e-8)
    
    # Quantize
    w_int8 = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    
    return w_int8, scale

def dequantize_weights(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize int8 weights to floating point.
    
    Args:
        w_int8: Quantized weights (int8)
        scale: Scaling factors
        
    Returns:
        w: Dequantized weights
    """
    return w_int8.to(scale.dtype) * scale

class FakeQuantLinear(nn.Module):
    """
    A Linear layer that simulates W8A16 quantization.
    It quantizes weights on the fly (or stores them quantized) and dequantizes for computation.
    This is for verification purposes.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # We start with standard weights for easy loading
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        idx = cls(linear.in_features, linear.out_features, linear.bias is not None)
        idx.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            idx.bias.data = linear.bias.data.clone()
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate quantization
        w_int8, scale = quantize_weights(self.weight)
        
        # Simulate dequantization
        w_fake_quant = dequantize_weights(w_int8, scale)
        
        # Compute using simulated weights
        return F.linear(x, w_fake_quant, self.bias)


from gpt2_engine.kernels.quant_matmul import quant_linear

class QuantLinear(nn.Module):
    """
    Triton-based W8A16 Linear Layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Register buffers for quantized weights so they are saved with model but not updated by optimizer
        self.register_buffer('weight', torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scales', torch.zeros((out_features,), dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        idx = cls(linear.in_features, linear.out_features, linear.bias is not None)
        idx.pack_weights(linear)
        return idx
        
    @torch.no_grad()
    def pack_weights(self, linear_layer: nn.Linear):
        w = linear_layer.weight.data
        if w.dtype != torch.float16:
             w = w.to(torch.float16)
             
        w_int8, scales = quantize_weights(w)
        self.weight.copy_(w_int8)
        self.scales.copy_(scales.view(-1))
        
        if linear_layer.bias is not None:
             b = linear_layer.bias.data
             if b.dtype != torch.float16:
                 b = b.to(torch.float16)
             self.bias.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quant_linear(x, self.weight, self.scales, self.bias)

def replace_linear_with_quant(model: nn.Module, exclude_names: list[str] = None):
    if exclude_names is None:
        exclude_names = ['lm_head']
        
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if name in exclude_names:
                continue
            
            # Check device of module to move new module there
            device = module.weight.device
            
            q_linear = QuantLinear.from_linear(module)
            q_linear = q_linear.to(device) # Move to same device
            
            setattr(model, name, q_linear)
        else:
            replace_linear_with_quant(module, exclude_names)
