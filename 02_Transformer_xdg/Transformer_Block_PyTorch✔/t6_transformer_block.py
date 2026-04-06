import torch
import torch.nn as nn

from t3_mha import MultiHeadAttention
from t4_norm import LayerNorm, RMSNorm
from t5_ffn import FeedForward


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        self.attention = MultiHeadAttention(dim, num_heads, is_causal=False)
        self.ffn = FeedForward(dim, hidden_dim)  # hidden_dim 默认为 4 * dim
        self.attention_norm = LayerNorm(dim)
        self.ffn_norm = LayerNorm(dim)

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


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim

        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(dim, num_heads, is_causal=True)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(dim, num_heads, is_causal=False)
        
        self.ffn = FeedForward(dim, hidden_dim)  # hidden_dim 默认为 4 * dim
        self.mask_attention_norm = LayerNorm(dim)
        self.attention_norm = LayerNorm(dim)
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
        norm_x = self.attention_norm(x)
        h2 = x + self.attention(norm_x, encoder_output, encoder_output)
        # 3. FFN
        out = h2 + self.ffn(self.ffn_norm(h2))  # Pre-Normalization
        return out


if __name__ == "__main__":
    # 参数设置
    dim = 128
    heads = 8
    seq_len = 10
    batch = 2
    
    # 测试 EncoderBlock
    block = EncoderBlock(dim=dim, num_heads=heads)
    
    # 模拟输入
    x = torch.randn(batch, seq_len, dim)
    output = block(x)
    print(f"Transformer Block 输入形状: {x.shape}")
    print(f"Transformer Block 输出形状: {output.shape}") 
    # 输出 torch.Size([2, 10, 128])

    # 测试 DecoderBlock
    decoder_block = DecoderBlock(dim=dim, num_heads=heads)

    # 模拟输入
    encoder_output = torch.randn(batch, seq_len, dim)
    decoder_output = decoder_block(x, encoder_output)
    print(f"Decoder Block 输入形状: {x.shape}, Encoder 输出形状: {encoder_output.shape}")
    