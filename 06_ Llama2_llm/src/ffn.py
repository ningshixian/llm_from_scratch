import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""LLaMA 前馈网络 (FFN) 完整实现"""


class FeedForward(nn.Module):
    """
    LLaMA 中的 MLP 层，采用 SwiGLU 激活函数。
    结构特点：三个线性投影层，且不使用偏置项 (bias=False)。
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int=256):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数 (SwiGLU的特殊需求)
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        # 三个线性投影，没有偏置项
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # W1: 用于生成门控信号
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)  # W3: 用于升维投影
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # W2: 用于降维回输出维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        经典形式：FFN(x) = ReLU(xW1) W2
        主流形式：SwiGLU(x) = (xW1 ⊗ silu(xW3)) W2
            其中SiLU(x) = x * sigmoid(x)，⊗表示元素级乘法
        """
        gate_output = F.silu(self.gate_proj(x))  # SiLU激活
        up_output = self.up_proj(x)              # 上投影
        intermediate = gate_output * up_output   # 元素级乘法
        return self.down_proj(intermediate)      # 下投影


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
