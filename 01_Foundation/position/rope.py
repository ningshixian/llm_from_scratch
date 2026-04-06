import torch
import torch.nn.functional as F

# 预计算缓存以加速推理
def _precompute_rotary_embeddings(seq_len, head_dim, base=10000, device=None):
    # 构建旋转频率 theta
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    
    # 计算不同位置的频率
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # 外积 [seq_len, dim/2]
    
    # 计算旋转角度的正弦和余弦值
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
    cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    # q, k: [batch_size, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    # position_ids: [batch_size, seq_len]
    """
    # multihead attention
    batch_size, seq_length, num_heads, head_dim = q.shape
    
    # 根据position_ids获取对应位置的cos和sin
    cos = cos.index_select(0, position_ids.reshape(-1)).reshape(batch_size, seq_length, 1, head_dim)
    sin = sin.index_select(0, position_ids.reshape(-1)).reshape(batch_size, seq_length, 1, head_dim)
    
    # RoPE 核心公式: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def rotate_half(x):
    # 旋转向量的一半维度
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


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
    q, k = apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin)
    print(q)
    q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm
    print(q)