import numpy as np

def qlora_forward(
    x: list[list[float]],
    quantized_W: list[list[int]],
    scale: float,
    zero_point: float,
    A: list[list[float]],
    B: list[list[float]],
    alpha: float = 1.0
) -> list[list[float]]:
    """
    QLoRA forward pass with 4-bit quantized frozen weights.
    
    Args:
        x: Input matrix (batch_size x in_features)
        quantized_W: 4-bit quantized weights (in_features x out_features)
            Values are integers that need to be dequantized
        scale: Quantization scale factor
        zero_point: Quantization zero point for dequantization
        A: LoRA matrix A (rank x out_features) - full precision
        B: LoRA matrix B (in_features x rank) - full precision
        alpha: LoRA scaling factor
        
    Returns:
        Output matrix (batch_size x out_features)
    """
    x = np.array(x, dtype=float)
    quantized_W = np.array(quantized_W, dtype=float)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # 1. dequantize frozen weight
    W = scale * (quantized_W - zero_point)

    # 2. base output + LoRA update
    base = x @ W
    lora = x @ B @ A

    out = base + alpha * lora
    return out.tolist()