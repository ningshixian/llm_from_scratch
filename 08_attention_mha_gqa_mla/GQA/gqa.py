import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Q：多头（num_heads）
K/V：少头（num_kv_heads）
K/V 会被 repeat 到 Q 的头数
"""

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
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

        # # 注册为模型的缓冲区，用于存储因果掩码
        # self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))  # 下三角
        # self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.num_heads, self.dim_head)    # B, T, H, d_k
        k = self.wk(x).view(B, T, self.num_kv_heads, self.dim_head) # B, T, H_kv, d_k
        v = self.wv(x).view(B, T, self.num_kv_heads, self.dim_head) # B, T, H_kv, d_k

        # # 应用 RoPE 作用在 Q/K 上（与 MHA 相同）
        # # KV Cache

        # GQA：扩展 KV 到 Q 的头数
        k = k.repeat_interleave(self.repetitions, dim=2)  # B, T, H, d_k
        v = v.repeat_interleave(self.repetitions, dim=2)

        # 将头作为批次维度处理。
        q = q.transpose(1, 2)  # (B, H, T, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # # 计算注意力（与 MHA 相同）
        scores = (q @ k.transpose(-2, -1)) / (self.dim_head ** 0.5)    # (B, H, T, d_k) * (B, H, d_k, T) -> (B, H, T, T)
        mask = torch.tril(torch.ones(T, T))
        scores.masked_fill_(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, d_k)

        # 合并 heads
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


if __name__ == "__main__":
    B, T, D = 2, 4, 8
    num_heads = 4
    num_kv_heads = 2

    x = torch.randn(B, T, D)
    gqa = GroupedQueryAttention(d_model=D, num_heads=num_heads, num_kv_heads=num_kv_heads)
    out = gqa(x)
    print(out.shape)  # 应该是 (B, T, D)