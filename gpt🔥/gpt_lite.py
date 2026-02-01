"""nanochat ->
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from rope import _precompute_rotary_embeddings


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    seq_len: int = 2048
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    drop_rate: float = 0.1

class RMSNorm(nn.Module):
    """现代模型（Llama等）标配，比 LayerNorm 更快且效果相当"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    """RoPE 解决了绝对位置编码无法外推的问题
    实际应用中需结合 cos/sin 缓存
    """
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # 这里的 causal mask 仍是学习重点
        self.register_buffer("bias", torch.tril(torch.ones(config.seq_len, config.seq_len))
                                    .view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        
        # ReShape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # q, k = norm(q), norm(k) # QK norm

        # 核心：Scaled Dot-Product Attention
        # 学习建议：实际项目中请直接使用 F.scaled_dot_product_attention (Flash Attention)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v 
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

# --- 前馈网络 (采用现代 ReLU² 激活) ---
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

# --- 变体 Block ---
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        # self.drop = nn.Dropout(config.drop_rate)

    def forward(self, x, cos_sin):
        # Pre-norm 架构，更有利于深层网络稳定
        x = x + self.attn(self.norm1(x), cos_sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "final_norm": RMSNorm(config.n_embd)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim

        # rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def forward(self, in_idx):
        B, T = in_idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(in_idx)
        for block in self.transformer.h:
            x = block(x, cos_sin)
            
        x = self.transformer.final_norm(x)
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        return logits

@torch.inference_mode()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        logits = model.forward(idx) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size) 只看最后一个时间步的输出
        if top_k is not None:
            # 提取概率最高的前 k 个值，并将除此之外的所有其他 logits 设为负无穷（-Inf）
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        if temperature > 0:
            # 温度采样后重新归一化，再按概率分布随机抽取一个 token
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
        else:  # 贪婪采样
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat((ids, next_ids), dim=1)  # (batch, n_tokens+1)
        token = next_ids.item()
        yield token # 流式
    return ids


if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPT(GPTConfig)
    model.eval()
