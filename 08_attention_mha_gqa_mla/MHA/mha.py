import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
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
        self.dropout = nn.Dropout(dropout)
        
        # #提前注册因果掩码
        # self.register_buffer('mask', torch.triu(torch.ones(T, T), diagonal=1))

    def forward(self, x, mask=None):
        # x 维度: [Batch Size, Tokens, d_model]
        B, T, D = x.shape

        # 1. 线性变换得到 QKV，并拆分为多头
        # 维度变化: (B, T, D) -> (B, T, H, d_k)
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k)

        # 2. 转置以便进行矩阵乘法
        # 维度变化: (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用KV缓存(用于推理加速)

        # 3. 计算注意力分数
        # matmul 只对最后两个维度进行矩阵运算
        # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # NEW：因果掩码
        if mask is not None:
            mask = mask.bool()[:T, :T]
            scores = scores.masked_fill(mask, -1e9)

        # 4.分数归一化（Softmax + Dropout）
        attn_weights = F.softmax(scores, dim=-1) # (B, H, T, T)
        attn_weights = self.dropout(attn_weights)
        
        # 5. 加权求和
        # (B, H, T, T) * (B, H, T, d_k) -> (B, H, T, d_k)
        context = torch.matmul(attn_weights, v)

        # 6. 合并多个头的结果
        # 先转置回 (B, T, H, d_k)，然后用 contiguous 保证内存连续，最后 view 成 (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        # 7. 最后的线性投影
        return self.W_o(context)


# --- 测试代码 ---
if __name__ == "__main__":
    B, T, D = 2, 10, 512  # Batch=2, Seq_len=10, Dim=512
    H = 8                # 8个头
    
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    x = torch.randn(B, T, D)
    
    output = mha(x)
    print(f"输入形状: {x.shape}")    # torch.Size([2, 10, 512])
    print(f"输出形状: {output.shape}") # torch.Size([2, 10, 512])
