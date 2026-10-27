import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import apply_rotary_emb, _precompute_freqs_cis

"""GQA (Grouped Query Attention) 
目的：减少 KV Cache 缓存占用

假设配置：$H = 32$，$H_{kv} = 8$，$d_k = 128$。
- 分组大小 $G = 32 / 8 = 4$。
- 32 个 Query 头被分成 8 个组，每组 4 个头。
- 第 0 组：Query 头 $[0, 1, 2, 3]$ 一起使用 KV 头 0。
- 第 1 组：Query 头 $[4, 5, 6, 7]$ 一起使用 KV 头 1。
依此类推……

支持
- RoPE：作用在 Q 和 K 上（非常关键！）
- KV cache
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, max_seq_len=512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.dim_head = d_model // num_heads        # d_k
        self.repetitions = num_heads // num_kv_heads   # 每个 KV head 被多少个 Q head 共享

        # Q 全头
        self.wq = nn.Linear(d_model, num_heads * self.dim_head, bias=False)
        # K/V 少头
        self.wk = nn.Linear(d_model, num_kv_heads * self.dim_head, bias=False)
        self.wv = nn.Linear(d_model, num_kv_heads * self.dim_head, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(0.1)

        # causal mask（注册 buffer，一次构建）
        self.register_buffer('causal_mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角
    
    def forward(self, x, freqs_cos=None, freqs_sin=None, past_key_value=None):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.num_heads, self.dim_head)    # B, T, H, d_k
        k = self.wk(x).view(B, T, self.num_kv_heads, self.dim_head) # B, T, H_kv, d_k
        v = self.wv(x).view(B, T, self.num_kv_heads, self.dim_head) # B, T, H_kv, d_k
        
        # 应用RoPE（只作用 Q/K）
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)   # x: (B,T,H,D)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # KV Cache （针对自回归推理）
        # GQA 让 KV Cache 缩小到 H_kv/H，提升吞吐量
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)  # 在序列维度 T 上拼接: (B, H_kv, T_past + T_q, d_k)
            v = torch.cat([past_v, v], dim=2)
        
        # 转置：将头作为批次维度处理 -> (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # GQA：扩展 KV 到 Q 的头数
        # [B, n_kv_groups, T, head_dim] → [B, n_heads, T, head_dim]
        k = torch.repeat_interleave(k, repeats=self.repetitions, dim=1)
        v = torch.repeat_interleave(v, repeats=self.repetitions, dim=1)

        # 计算注意力（与 MHA 相同）
        scores = (q @ k.transpose(-2, -1)) / (self.dim_head ** 0.5)    # (B, H, T, d_k) * (B, H, d_k, T) -> (B, H, T, T)
        # causal mask（裁剪到当前 T）
        scores.masked_fill_(self.causal_mask.bool()[:T, :T], -torch.inf)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = attn @ v  # (B, H, T, d_k)

        # 合并 heads
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.wo(context)
        
        return output, (k, v)  # 返回 KV Cache 以便下次调用


if __name__ == "__main__":
    # 配置参数
    dim = 512
    n_heads = 8
    n_kv_groups = 2  # 8个Q头，2个KV头，每组4个Q共享1组KV
    seq_len = 64
    batch_size = 2

    # 初始化模型
    model = GroupedQueryAttention(
        d_model=dim,
        num_heads=n_heads,
        num_kv_heads=n_kv_groups
    )

    # 模拟输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 模拟 RoPE 频率
    # head_dim = dim // n_heads = 64
    f_cos, f_sin = _precompute_freqs_cis(seq_len, dim // n_heads)

    # 前向传播
    output, past_kv = model(x, f_cos, f_sin)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 验证输出维度
    assert output.shape == (batch_size, seq_len, dim), "维度校验失败！"
    print("GQA 实现验证成功！")