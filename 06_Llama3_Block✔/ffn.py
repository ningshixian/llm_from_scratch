import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""LLaMA 前馈网络 (FFN) 完整实现"""

class LlamaMLP(nn.Module):
    """
    LLaMA 中的 MLP 层，采用 SwiGLU 激活函数。
    结构特点：三个线性投影层，且不使用偏置项 (bias=False)。
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # 固定依赖：SwiGLU 的三个无 bias 投影。
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        经典形式：FFN(x) = ReLU(xW1) W2
        主流形式：SwiGLU(x) = (xW1 ⊗ silu(xW3)) W2
            其中SiLU(x) = x * sigmoid(x)，⊗表示元素级乘法
        """
        gate = F.silu(self.gate_proj(x))  # SiLU激活
        up = self.up_proj(x)              # 上投影
        return self.down_proj(gate * up)      # 下投影


if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 128
    
    # 初始化 FFN 模块
    ffn = FeedForward(
        dim=dim,
        hidden_dim=4 * dim,
        multiple_of=256,
    )

    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)

    # 执行前向传播
    output = ffn(x)

    # 验证输出形状
    print("--- FeedForward (SwiGLU) Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
