import torch
import torch.nn.functional as F


def linear_attention(Q, K, V):
    phi_K = F.elu(K) + 1
    phi_Q = F.elu(Q) + 1
    QKV = phi_K.transpose(-1, -2) @ V
    QKV = phi_Q @ QKV

    sum_K = phi_K.sum(dim=1, keepdim=True)
    denominator = (phi_Q * sum_K).sum(dim=-1, keepdim=True)
    
    return QKV / denominator


# 🧪 Debug
Q = torch.randn(1, 8, 16)
K = torch.randn(1, 8, 16)
V = torch.randn(1, 8, 32)
out = linear_attention(Q, K, V)
print("Output shape:", out.shape)   # (1, 8, 32)
print("Has NaN?", torch.isnan(out).any().item())