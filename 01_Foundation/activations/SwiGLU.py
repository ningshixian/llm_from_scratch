import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SwiGLU(x) = SiLU(W1 * x) * (W2 * x)
其中，
SiLU(x) = x * sigmoid(x)
"""

class SwiGLU_MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
