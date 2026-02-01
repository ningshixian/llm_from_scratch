import torch
import torch.nn as nn
import torch.nn.functional as F

"""LLaMA 前馈网络 (FFN) 完整实现"""

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # layernorm作用在(-1) 最后一维进行归一化
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out_mean_var = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out_mean_var + self.beta # feature level
        return mean, var, out_mean_var, out

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    相比标准 LayerNorm，RMSNorm 省去了减去均值的步骤，计算效率更高。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 公式: x / sqrt(mean(x^2) + eps)
        # torch.rsqrt 是平方根倒数，效率更高
        rms = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * rms

class MLP(nn.Module):
    """
    LLaMA 中的 MLP 层，采用 SwiGLU 激活函数。
    结构特点：三个线性投影层，且不使用偏置项 (bias=False)。
    """
    def __init__(self, dim: int, hidden_dim: int = None, multiple_of: int = 256):
        super().__init__()
        
        # 1. 隐藏维度计算策略
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        # 2. 定义三个线性层
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # W1: 用于生成门控信号
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)  # W3: 用于升维投影
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)  # W2: 用于降维回输出维度

    def forward(self, x):
        """
        SwiGLU 实现公式: (SiLU(W1x) * W3x) * W2
        其中 SiLU(x) = x * sigmoid(x)
        """
        gate_output = F.silu(self.gate_proj(x))  # SiLU 激活 (即 Swish) 
        up_output = self.up_proj(x)              # 上投影
        intermediate = gate_output * up_output   # 门控
        return self.down_proj(intermediate)      # 下投影

class FFN(nn.Module):
    """
    整合模块：包含 Pre-Normalization 的完整前馈网络块
    """
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.mlp = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 典型的 Pre-Norm 结构
        # 注意：这里的输出通常会和输入 x 进行残差连接（在 Block 层处理）
        h = self.norm(x)
        h = self.mlp(h)
        return x + self.dropout(h)

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟输入: [Batch Size, Sequence Length, Embedding Dimension]
    dim = 4096
    x = torch.randn(2, 10, dim)
    
    ffn = FFN(dim=dim)
    output = ffn(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"隐藏层维度 (自动调整后): {ffn.mlp.gate_proj.out_features}")