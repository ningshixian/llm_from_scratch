from dataclasses import dataclass
import math
from transformers import BertTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

"""
https://github.com/datawhalechina/happy-llm/blob/main/docs/chapter2/code/transformer.py
https://www.k-a.in/llm7.html
https://www.k-a.in/llm3.html
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512, is_causal=False):
        super().__init__()
        assert d_model % n_heads == 0, "维度必须能被头数整除"

        self.d_model = d_model  # hidden_size
        self.n_heads = n_heads  # 对应 H
        self.d_k = d_model // n_heads # head_dim

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于因果掩码
        if is_causal:
            # 加性 mask（贴近工程实现）
            mask = torch.full((1, 1, max_len, max_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)
        
            # #布尔 mask（偏手撕）
            # self.register_buffer('mask', torch.triu(torch.ones(max_len, max_len), diagonal=1))
            # mask = self.mask.bool()[:T, :T]
            # scores.masked_fill_(mask, -1e9) # float("-inf")

    def forward(self, q, k, v):
        # x 维度: [Batch Size, Tokens, d_model]
        B, T, D = q.shape

        # 1. 线性变换得到 QKV，并拆分为多头
        # 维度变化: (B, T, D) -> (B, T, H, d_k)
        q = self.W_q(q).view(B, T, self.n_heads, self.d_k)
        k = self.W_k(k).view(B, T, self.n_heads, self.d_k)
        v = self.W_v(v).view(B, T, self.n_heads, self.d_k)

        # 2. 转置以便后面注意力矩阵乘法
        # 维度变化: (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用KV缓存(用于推理加速)

        # 3. 计算注意力分数
        # matmul 只对最后两个维度进行矩阵运算
        # (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # NEW：Apply mask if provided (for padding or future masking)
        if self.is_causal:
            # 这里截取到序列长度，因为有些序列可能比 max_len 短
            scores = scores + self.mask[:, :, :T, :T]

        # 4.分数归一化（Softmax）
        attn_weights = F.softmax(scores, dim=-1) # (B, H, T, T)
        
        # 5. 加权求和
        # (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        context = torch.matmul(attn_weights, v)

        # 6. 合并多个头的结果
        # 先转置回 (B, T, H, d_k)，然后用 contiguous 保证内存连续，最后 view 成 (B, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)

        # 7. 最后的线性投影
        return self.W_o(context)
    

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps  # 防止分母为零的数值稳定项

    def forward(self, x):
        # 在统计每个样本所有维度的值，求均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        out_mean_var = (x - mean) / (std + self.eps)
        out = self.gamma * out_mean_var + self.beta # feature level
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim    # 通常 FFN 的隐藏层维度是输入维度的 4 倍
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        # FFN(x) = W2(ReLU(W1(x)))
        return self.w2(F.relu(self.w1(x)))



class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.attention = MultiHeadAttention(dim, num_heads, is_causal=False)
        self.ffn = FeedForward(dim, hidden_dim)  # hidden_dim 默认为 4 * dim
        self.attention_norm = LayerNorm(dim)
        self.ffn_norm = LayerNorm(dim)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x 维度: (batch_size, seq_len, dim)
        """
        # 1. Self-Attention
        x = self.attention_norm(x)
        h = x + self.attention(x, x, x)         # Pre-Normalization
        # 2. FFN
        out = h + self.ffn(self.ffn_norm(h))    # Pre-Normalization
        return out


class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderBlock(args.dim, args.n_heads) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.dim)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.mask_attention = MultiHeadAttention(dim, num_heads, is_causal=True)
        self.cross_attention = MultiHeadAttention(dim, num_heads, is_causal=False)
        
        self.ffn = FeedForward(dim, hidden_dim)  # hidden_dim 默认为 4 * dim
        self.mask_attention_norm = LayerNorm(dim)
        self.cross_attention_norm = LayerNorm(dim)
        self.ffn_norm = LayerNorm(dim)

    def forward(self, x, encoder_output):
        """
        x 维度: (batch_size, seq_len, dim)
        encoder_output 维度: (batch_size, seq_len_enc, dim)
        """
        # 1. Masked Self-Attention
        norm_x = self.mask_attention_norm(x)
        x = x + self.mask_attention(norm_x, norm_x, norm_x)  # Pre-Normalization
        # 2. Cross-Attention
        norm_x = self.cross_attention_norm(x)
        h2 = x + self.cross_attention(norm_x, encoder_output, encoder_output)
        # 3. FFN
        out = h2 + self.ffn(self.ffn_norm(h2))  # Pre-Normalization
        return out


class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderBlock(args.dim, args.n_heads) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.dim)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    '''sin-cos 位置编码模块'''
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even."
        self.d_model = d_model
        self.max_len = max_len

        # Empty encodings vectors
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2)
        div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))

        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        # 注册成 buffer，默认不可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 嵌入向量 + 绝对位置编码(标准实现)
        # x: (batch_size, seq_len, d_model)
        pe = self.pe[:, :x.size(1), :].to(dtype=x.dtype)
        return x + pe
    
    # def forward(self, x):
    #     # 嵌入向量 + 可学习位置编码
    #     pe = nn.Parameter( torch.randn(1, x.size(1), self.d_model) ) # learnable
    #     return x + pe


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        self.args = args

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args.vocab_size, args.n_embd),
            wpe=PositionalEncoding(args.dim, args.max_len),
            drop=nn.Dropout(args.dropout),
            encoder=Encoder(args),
            decoder=Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, src_idx, target_idx=None):
        """
            src_idx: (batch size, sequence length, 1)
            target_idx 为目标序列，用于计算 loss
        """
        # Apply embedding and positional encoding
        # -> (batch size, sequence length, n_embd)
        tok_emb = self.transformer.wte(src_idx)
        pos_emb = self.transformer.wpe(tok_emb)
        x = self.transformer.drop(pos_emb)

        # Pass through encoder and decoder
        enc_out = self.transformer.encoder(x)
        x = self.transformer.decoder(x, enc_out)

        if target_idx is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_idx.view(-1), ignore_index=-100)
        else:
            # 推理阶段，自回归预测，我们只需要 logits
            logits = self.lm_head(x[:, [-1], :])  # 取最后一个时间步，同时保留时间维度
            loss = None
        return logits, loss


@dataclass
class ModelArgs:
    n_embd: int # 嵌入维度
    n_heads: int # 头数
    dim: int # 模型维度
    dropout: float
    max_len: int
    vocab_size: int
    n_layer: int


def main():
    args = ModelArgs(100, 10, 100, 0.1, 512, 1000, 2)
    text = "我喜欢快乐地学习大模型"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=args.max_len,
        truncation=True,
        padding='max_length'
    )
    args.vocab_size = tokenizer.vocab_size

    transformer = Transformer(args)
    inputs_id = inputs_token['input_ids']
    logits, loss = transformer.forward(inputs_id)
    print(logits)
    predicted_ids = torch.argmax(logits, dim=-1).item()
    output = tokenizer.decode(predicted_ids)
    print(output)

if __name__ == "__main__":
    print("开始")
    main()