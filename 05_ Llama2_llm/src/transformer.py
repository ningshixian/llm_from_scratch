import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)

from typing import Optional
from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .attention import GroupedQueryAttention
from .ffn import FeedForward


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
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
        )
        # 定义层的ID
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        )
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
        )
        # 定义层的ID
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self, x, freqs_cos, freqs_sin
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


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
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
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


if __name__ == "__main__":
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
    # 实例化LLaMA2Model
    model = LlamaTransformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)

    out = model(x)
    print(out.logits.shape) # [batch_size, 1, vocab_size]