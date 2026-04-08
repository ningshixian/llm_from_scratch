import torch
import torch.nn.functional as F

# 预计算缓存以加速推理
def _precompute_rotary_embeddings(seq_len, head_dim, base=10000, device=None):
    assert head_dim % 2 == 0, "head_dim 必须是偶数以便进行旋转"
    # 构建旋转频率 theta
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    
    # 计算不同位置的频率
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # 外积 [seq_len, dim/2]
    freqs = torch.cat((freqs, freqs), dim=-1)     # 扩展成 [seq_len, head_dim]
    
    # 计算旋转角度的正弦和余弦值 -> [1, seq_len, 1, head_dim]
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: [1, seq_len, 1, head_dim]
    """
    # multihead attention
    batch_size, seq_length, num_heads, head_dim = q.shape
    
    # RoPE 核心公式: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    # 按偶奇位成对旋转
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


if __name__=='__main__':
    # 示例调用
    seq_len = 512
    n_embd = 768
    n_head = 6
    head_dim = n_embd // n_head
    B,T = 2, seq_len

    # 创建张量 [B, T, n_head, head_dim]
    q = torch.randn(B, T, n_head, head_dim)
    k = torch.randn(B, T, n_head, head_dim)

    # QK norm 通常在加 RoPE 之前做，确保数值稳定性
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))

    cos, sin = _precompute_rotary_embeddings(seq_len, head_dim)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    print(q.shape, k.shape)
