import torch
import torch.nn.functional as F

# x shape: [batch, heads, seq_len, head_dim] (通常应用在 Attention 的 head 维度)
def _precompute_rotary_embeddings(seq_len, head_dim, base=10000, device=None):
    # 预计算频率 theta
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    # stride the time steps
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    # calculate the rotation frequencies at each (time, channel) pair
    freqs = torch.outer(t, inv_freq)  # [seq_len, dim/2]
    
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
    cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
    return cos, sin

# RoPE 核心公式: (x * cos) + (rotate_half(x) * sin)
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


if __name__=='__main__':
    # 示例调用
    seq_len = 2048
    n_embd = 768
    n_head = 6
    head_dim = n_embd // n_head
    B,T = 2, seq_len

    # 创建张量 [B, T, n_head, head_dim]
    q = torch.randn(B, T, n_head, head_dim)
    k = torch.randn(B, T, n_head, head_dim)

    cos, sin = _precompute_rotary_embeddings(seq_len, head_dim)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    print(q)
    q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm
    print(q)