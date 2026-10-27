import torch

def rotate_half(x: torch.Tensor):
    """将后半部分维度取负并与前半部分互换：[-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def precompute_cos_sin(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算旋转频率: (dim // 2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=inv_freq.device).float()
    
    # 外积得到每个位置的角度: (seq_len, dim // 2)
    freqs = torch.outer(t, inv_freq)
    
    # 拼成完整维度 (seq_len, dim)，形式为 [freqs, freqs]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: (batch_size, seq_len, n_heads, head_dim)
    # 扩展 cos, sin 维度以适配广播: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # 旋转公式：x * cos + rotate_half(x) * sin
    return (x * cos) + (rotate_half(x) * sin)

# --- 测试运行 ---
batch_size, seq_len, n_heads, head_dim = 2, 8, 4, 64
x = torch.randn(batch_size, seq_len, n_heads, head_dim)

cos, sin = precompute_cos_sin(head_dim, seq_len)
x_out = apply_rotary_emb(x, cos, sin)

print("输入尺寸:", x.shape)
print("输出尺寸:", x_out.shape)