import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # 每个头的维度

        # 定义线性投影层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # 输出投影层
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 注册因果掩码 (防止当前位置看到未来的 token)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. 投影并拆分为多头形状: (b, tokens, d_out) -> (b, tokens, num_heads, head_dim)
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)

        # 2. 转置以匹配矩阵乘法要求: (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 3. 计算缩放点积注意力分数 (QK^T / sqrt(d_k))
        # (b, heads, tokens, head_dim) @ (b, heads, head_dim, tokens) -> (b, heads, tokens, tokens)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim**0.5)

        # 4. 应用因果掩码 (将未来位置的分数设为负无穷)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 5. Softmax 归一化并应用 Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和得到上下文向量: (b, heads, tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous()
        
        # 7. 合并多个头并进行最终输出投影
        context_vec = context_vec.view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


# torch.manual_seed(123)

# max_len = 1024
# context_length = max_len
# d_in = output_dim = 256
# d_out = d_in

# mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

# batch = input_embeddings
# context_vecs = mha(batch)
# print("context_vecs.shape:", context_vecs.shape)