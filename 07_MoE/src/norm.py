import torch
import torch.nn as nn


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

if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, dim = 4, 16, 64
    x = torch.randn(batch_size, seq_len, dim)

    # 初始化并应用 RMSNorm
    norm = RMSNorm(dim)
    output = norm(x)

    # 验证输出形状
    print("--- RMSNorm Test ---")
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)