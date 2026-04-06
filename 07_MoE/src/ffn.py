import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class FeedForward(nn.Module):
    """
    LLaMA 中的 MLP 层，采用 SwiGLU 激活函数。
    结构特点：三个线性投影层，且不使用偏置项 (bias=False)。
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int=256):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)    # 常用于 LLaMA-SwiGLU 缩放 2/3 的隐藏维度
        # 向上对齐到 multiple_of 的倍数
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # W1: 用于生成门控信号
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)  # W3: 用于升维投影
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # W2: 用于降维回输出维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        经典形式：FFN(x) = ReLU(xW1) W2
        主流形式SwiGLU：FFN(x) = (xW1 ⊙ σ(xW3)) W2
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float], num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 门控网络：决定每个 Token 去往哪个专家
        self.gate = nn.Linear(dim, num_experts, bias=False)
        # 专家列表：创建 num_experts 个独立的 FeedForward 网络
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        # 1. 门控网络
        gate_logits = self.gate(x_flat) # (B*T, num_experts)
        # 2. Top-k 路由
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1) # 归一化权重
        
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # 3. 找出所有选中当前专家 i 的 token 索引
            batch_idx, k_idx = torch.where(indices == i)
            
            if len(batch_idx) == 0:
                continue
                
            # 4. 取出对应的输入进行计算
            expert_input = x_flat[batch_idx]
            expert_out = expert(expert_input)
            
            # 5. 获取对应的权重
            expert_weights = weights[batch_idx, k_idx].unsqueeze(-1) # (num_selected, 1)
            
            # 6. 将结果加权累加回输出张量
            output.index_add_(0, batch_idx, expert_out * expert_weights)
            
        return output.view(B, T, D)



if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 128
    
    # 初始化 MoE 模块
    model = MoE(
        dim=dim,
        hidden_dim=4 * dim,
        multiple_of=256,
        ffn_dim_multiplier=None
    )

    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)

    # 执行前向传播
    output = model(x)

    # 验证输出形状
    print("--- MoE Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
