# import numpy as np
import torch

def layernorm(x, eps=1e-5):
    mean = torch.mean(x, axis=-1, keepdims=True)
    var = torch.var(x, axis=-1, keepdims=True, correction=0)
    return (x - mean) / torch.sqrt(var + eps)


def rmsnorm(x: torch.Tensor, eps: float = 1e-5):
    rms = torch.sqrt(torch.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms
