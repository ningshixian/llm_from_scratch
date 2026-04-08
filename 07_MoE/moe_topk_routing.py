import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class Expert(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

"""
每个 token
-> gate 打分
-> 选 top-k expert
-> softmax 得到权重
-> 分别过这 k 个 expert
-> 加权求和
"""

class TopkRoutingMoE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # gate: 给每个 token 打到各个 expert 的分数
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # 多个 expert
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        out = torch.zeros_like(x)

        # 1) gate 打分
        scores = self.gate(x)   # (B, T, num_experts)

        # 2) 每个 token 选 top-k 个 expert
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)   # (B, T, K)

        # 3) 对 top-k 分数做 softmax，变成权重
        topk_probs = F.softmax(topk_scores, dim=-1)   # (B, T, K)

        # 4) 每个 token 分别送到它的 top-k expert
        for b in range(B):
            for t in range(T):
                token_x = x[b, t]   # (D,)

                for k in range(self.top_k):
                    expert_id = topk_indices[b, t, k].item()
                    prob = topk_probs[b, t, k]

                    expert_out = self.experts[expert_id](token_x)
                    out[b, t] += prob * expert_out

        return out


if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 128
    
    # 初始化 MoE 模块
    model = TopkRoutingMoE(
        dim=dim,
        hidden_dim=4 * dim,
        num_experts=4,
        top_k=2
    )

    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)

    # 执行前向传播
    output = model(x)

    # 验证输出形状
    print("--- MoE Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
