import torch
import torch.nn as nn
import torch.nn.functional as F

"""高效的多头注意力实现（Multi-Head Attention）"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, qkv_bias=False):
        super().__init__()
        assert d_model % n_heads == 0, "维度必须能被头数整除"

        self.d_model = d_model  # hidden_size
        self.n_heads = n_heads  # 对应 H
        self.d_k = d_model // n_heads # head_dim

        # 定义 Q, K, V 的线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model)
        # self.dropout = nn.Dropout(dropout)
        
        # # 注册为模型的缓冲区，用于存储因果掩码
        # self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))  # 下三角
        self.register_buffer('mask', torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))  # 上三角

    def forward(self, x):  #, mask=None
        # x 维度: [Batch Size, Tokens, d_model]
        B, T, D = x.shape

        # =========1. 线性变换得到 QKV，并拆分为多头
        # 维度变化: (B, T, D) -> (B, T, H, d_k)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k)

        # =========2. 转置以便后面注意力矩阵乘法
        # 维度变化: (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用KV缓存(用于推理加速)

        # 3 缩放点积计算 Attention Scores
        # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # ========== NEW：因果掩码 ============
        mask = self.mask.bool()[:T, :T]
        scores.masked_fill_(mask, -torch.inf)

        # 4.分数归一化（Softmax）
        attn_weights = F.softmax(scores, dim=-1) # (B, H, T, T)
        # attn_weights = self.dropout(attn_weights)
        
        # 5 加权求和
        # (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        context = attn_weights @ v

        # =========6. 合并多个头的结果
        # 先转置回 (B, T, H, d_k)，然后用 contiguous 保证内存连续，最后 view 成 (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.W_o(context)

        return output


if __name__ == "__main__":
    torch.manual_seed(123)
    B, T, D = 2, 10, 512  # Batch=2, Seq_len=10, Dim=512
    H = 8                # 8个头
    
    mha = MultiHeadAttention(d_model=D, n_heads=H, max_seq_len=T)
    x = torch.randn(B, T, D)
    
    output = mha(x)  # 自注意力，Q=K=V=x
    print(f"输入形状: {x.shape}")    # torch.Size([2, 10, 512])
    print(f"输出形状: {output.shape}") # torch.Size([2, 10, 512])
