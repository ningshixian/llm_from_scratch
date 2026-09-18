import torch
import torch.nn as nn

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q 独立投影，K/V 共享投影（输出维度为 head_dim）
        self.Wq = nn.Linear(d_model, d_model)  # Q: [num_heads × head_dim]
        self.Wk = nn.Linear(d_model, self.head_dim)  # K: [1 × head_dim]
        self.Wv = nn.Linear(d_model, self.head_dim)  # V: [1 × head_dim]

        # # 注册为模型的缓冲区，用于存储因果掩码
        # self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))  # 下三角
        # self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角

    def forward(self, x, past_kv=None):
        B, L, _ = x.shape
        Q = self.Wq(x).view(B, L, self.num_heads, self.head_dim)
        # 通过 unsqueeze 和隐式广播实现 K/V 的共享。
        K = self.Wk(x).unsqueeze(2)  # 插入组维度，广播到所有头
        V = self.Wv(x).unsqueeze(2)

        # 计算注意力分数（Q 与共享的 K/V 交互）
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, V)
        return output.contiguous().view(B, L, self.d_model)
