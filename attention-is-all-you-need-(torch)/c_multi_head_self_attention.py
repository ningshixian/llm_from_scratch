import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "维度必须能被头数整除"

        # 一次性定义所有头的 Q, K, V 映射
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 1. 线性变换并拆分为多头
        # 维度变化: (B, L, D) -> (B, L, H, D_h) -> (B, H, L, D_h)
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 计算 Scaled Dot-Product
        # scores 维度: (B, H, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # NEW：因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)

        # Softmax 归一化并应用 Dropout
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights = self.dropout(attn_weights)
        
        # 3. 加权求和
        # output 维度: (B, H, L, D_h)
        context = torch.matmul(attn_weights, v)

        # 4. 拼接 (Concatenate)
        # (B, H, L, D_h) -> (B, L, H, D_h) -> (B, L, D)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 5. 最后的线性投影
        return self.out_linear(context)

# 快速测试
mha = MultiHeadAttention(embed_dim=128, num_heads=8)
x = torch.randn(1, 10, 128) # 1个句子，10个词，每个词128维
output = mha(x)
print("多头注意力输出形状:", output.shape) # 应该是 [1, 10, 128]