import torch
import torch.nn.functional as F

"""
RoPE 的做法不是给 token 额外加一个位置向量，而是把位置信息写进查询（Query，Q）和键（Key，K）的旋转里，让注意力点积感知相对距离。

RoPE：对 Q/K 的偶数-奇数维做二维旋转
(x1, x2) → (x1*cos - x2*sin, x1*sin + x2*cos)
"""

def _precompute_freqs_cis(seq_len, dim, device=None):
    """ 生成位置与成对维对应的复数旋转因子
    """
    assert dim % 2 == 0, "dim 必须是偶数以便进行旋转"
    
    # 构建旋转频率（theta）
    channel_range = torch.arange(0, dim, 2, device=device)  # (D/2,)
    inv_freq = 1.0 / (10000 ** (channel_range / dim))

    # 计算不同位置的旋转角度
    pos = torch.arange(seq_len, device=device)  # (T,)
    angle = pos[:, None] * inv_freq[None, :]  # 外积 → 角度矩阵 (T, D/2)
    # angle = torch.cat((angle, angle), dim=-1)     # 扩展成 [seq_len, dim]
    
    cos = torch.cos(angle)  # (T, D/2)
    sin = torch.sin(angle)
    return cos, sin

def apply_rotary_emb(x, cos, sin):
    """将 Q/K 成对实数维转换为复数表示
    x: (B,T,H,D) 
    cos: (T, D/2)
    """
    # 拆分偶数/奇数维s
    x1 = x[..., 0::2]  # (B,T,H,D/2)
    x2 = x[..., 1::2]

    # 对齐维度
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 旋转（核心）
    x_rot_1 = x1 * cos - x2 * sin
    x_rot_2 = x1 * sin + x2 * cos

    # 使用 stack 和 flatten/reshape 来高效地交错合并
    # 1. 堆叠: [batch_size, num_heads, seq_len, head_dim / 2, 2]
    # 2. 展平: [batch_size, num_heads, seq_len, head_dim]        
    x_out = torch.stack([x_rot_1, x_rot_2], dim=-1)
    x_out = x_out.flatten(-2)
    

    return x_out


if __name__=='__main__':
    # 示例调用
    B,T,n_heads, head_dim = 2, 512, 6, 128
    dim = n_heads * head_dim

    # 创建张量 [B, T, n_heads, head_dim]
    q = torch.randn(B, T, n_heads, head_dim)
    k = torch.randn(B, T, n_heads, head_dim)

    # QK norm 通常在加 RoPE 之前做，确保数值稳定性
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))

    cos, sin = _precompute_freqs_cis(T, head_dim)

    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)
    print(q.shape, k.shape)
