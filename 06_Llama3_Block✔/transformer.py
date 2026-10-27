import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

from typing import Optional
from .norm import RMSNorm
from .rope import _precompute_freqs_cis
from .attention import GroupedQueryAttention
from .ffn import LlamaMLP

"""
位置编码 → RoPE
MHA → GQA / MLA
稠密 MLP → SwiGLU MLP
LayerNorm → RMSNorm

一个块依次执行两次“归一化 → 子层变换 → 残差相加”。
"""


class DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None,
        multiple_of: int,
        norm_eps: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = GroupedQueryAttention(
            d_model=dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads
        )
        self.feed_forward = LlamaMLP(dim, hidden_dim=4 * dim)
        # 定义层的ID
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self, x, freqs_cos, freqs_sin
    ) -> torch.Tensor:
        attn_out, _ = self.attention(self.attention_norm(x), freqs_cos, freqs_sin)  # past_kv
        h = x + attn_out

        ffn_out = self.feed_forward(self.ffn_norm(h))
        h = h + ffn_out
        return h


class LlamaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-6,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size    # 词汇表大小
        self.n_layers = n_layers        # 层数
        self.dim = dim
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        # Decoder层
        self.layers = nn.ModuleList([
            DecoderLayer(
                i,
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                multiple_of=multiple_of,
                norm_eps=norm_eps,
                max_seq_len=max_seq_len,
            )
            for i in range(n_layers)
        ])
        # 归一化层
        self.norm = RMSNorm(dim, eps=norm_eps)
        # 输出层
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight 

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = _precompute_freqs_cis(self.max_seq_len, self.dim // self.n_heads)
        print("预计算 cos 形状:", freqs_cos.shape)  # 必须是 torch.Size([64, 16])
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None

    def forward(self, tokens, targets=None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        # 获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            # 如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        return logits, self.last_loss


    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # 只保留最后一个时间步的输出
            
            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # 只返回生成的token


def main() -> None:
    model = LlamaTransformer(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=8,
        n_kv_heads=2,
        multiple_of=64,
        norm_eps=1e-6,
        # max_batch_size=4,
        max_seq_len=64,
    )

    # 构造随机 token 序列并执行前向
    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, 1000, (batch_size, seq_len))
    logits, _ = model(tokens)  # , start_pos=0

    # 期望: [batch_size, seq_len, vocab_size]
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()