"""
LLM 推理解码面试手写题 · 裸代码速查（不含 KV Cache）

覆盖：greedy、temperature、top-k、top-p、重复惩罚、批量 EOS、beam search、
speculative decoding。

约定：B=batch size，T=序列长度，V=词表大小；logits 形状为 (B, V)。
"""

import torch
import torch.nn.functional as F


# 1. Greedy decoding：每步选择概率最大的 token
def greedy(logits):
    return logits.argmax(dim=-1, keepdim=True)  # (B, V) -> (B, 1)


# 2. Temperature sampling：p_i = softmax(logit_i / temperature)
def temperature_sample(logits, temperature=1.0):
    assert temperature > 0
    probs = F.softmax(logits / temperature, dim=-1)  # softmax 沿词表维 V
    return torch.multinomial(probs, num_samples=1)   # (B, 1)


# 3. Top-k：只保留 logit 最大的 k 个 token，再归一化采样
def top_k_sample(logits, k, temperature=1.0):
    assert temperature > 0 and 1 <= k <= logits.size(-1)
    logits = logits / temperature
    threshold = logits.topk(k, dim=-1).values[..., -1, None]  # (B, 1)
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


# 4. Top-p / nucleus：保留累计概率刚好覆盖 p 的最小 token 集合
def top_p_sample(logits, p=0.9, temperature=1.0):
    assert 0 < p <= 1 and temperature > 0
    sorted_logits, sorted_idx = (logits / temperature).sort(descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    remove = sorted_probs.cumsum(dim=-1) > p
    remove[..., 1:] = remove[..., :-1].clone()  # 右移：保留首个使累计概率超过 p 的 token
    remove[..., 0] = False                      # 至少保留一个 token
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    sampled_pos = torch.multinomial(F.softmax(sorted_logits, dim=-1), 1)
    return sorted_idx.gather(-1, sampled_pos)   # 排序后下标 -> 原词表 token id


# 5. Repetition penalty（Hugging Face 常见规则）：正数除 penalty，负数乘 penalty
def apply_repetition_penalty(logits, input_ids, penalty=1.0):
    assert penalty > 0
    logits = logits.clone()
    seen_logits = logits.gather(1, input_ids)   # (B, T)，同一 token 重复出现不影响正确性
    seen_logits = torch.where(seen_logits < 0, seen_logits * penalty,
                              seen_logits / penalty)
    return logits.scatter(1, input_ids, seen_logits)


# 6. 无 KV Cache 的批量生成：已结束样本持续填 pad，直到全部 EOS
@torch.no_grad()
def generate(model, input_ids, max_new_tokens, eos_id, pad_id, sample_fn=greedy):
    finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]      # (B, T, V) -> (B, V)
        next_token = sample_fn(logits).squeeze(-1)
        next_token = torch.where(finished, pad_id, next_token)
        input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)
        finished |= next_token.eq(eos_id)
        if finished.all():
            break
    return input_ids


# 7. Beam search（单样本裸版）：累计 log 概率；长度惩罚只用于最终排序
@torch.no_grad()
def beam_search(model, input_ids, beam_size, max_new_tokens, eos_id, length_penalty=1.0):
    assert input_ids.shape[0] == 1 and beam_size >= 1
    beams = [(input_ids, 0.0, False)]  # (tokens[1,T], sum_log_prob, finished)

    for _ in range(max_new_tokens):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue
            log_probs = F.log_softmax(model(tokens)[:, -1, :], dim=-1)
            values, indices = log_probs.topk(beam_size, dim=-1)
            for log_p, token_id in zip(values[0], indices[0]):
                new_tokens = torch.cat([tokens, token_id.view(1, 1)], dim=1)
                # Beam Search 目标是找到联合概率最大的一条路径, 可以取log转化为求和
                candidates.append((new_tokens, score + log_p.item(), token_id.item() == eos_id))

        # 搜索过程中按未归一化累计分数剪枝；避免长度惩罚破坏可加性
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        # 如果所有beam都已完成，则停止生成
        if all(finished for _, _, finished in beams):
            break

    def normalized_score(beam):
        tokens, score, _ = beam
        generated_len = max(tokens.size(1) - input_ids.size(1), 1)
        return score / (generated_len ** length_penalty)

    return max(beams, key=normalized_score)[0]


# 8. Speculative decoding（greedy 裸版）
# draft 连续提议 gamma 个 token；target 一次验证，接受最长相同前缀，分歧处用 target。
@torch.no_grad()
def speculative_greedy(target, draft, input_ids, max_new_tokens, gamma=4):
    assert input_ids.shape[0] == 1 and gamma >= 1  # batch 内接受长度不同，裸版仅写 B=1
    generated = 0
    while generated < max_new_tokens:
        prefix_len = input_ids.size(1)
        proposal = input_ids
        steps = min(gamma, max_new_tokens - generated)

        for _ in range(steps):
            draft_token = draft(proposal)[:, -1, :].argmax(-1, keepdim=True)
            proposal = torch.cat([proposal, draft_token], dim=1)

        # target 对 proposal 中每个草稿 token 做并行验证。
        target_pred = target(proposal)[:, prefix_len - 1:-1, :].argmax(-1)
        draft_tokens = proposal[:, prefix_len:]
        mismatch = target_pred.ne(draft_tokens)
        accepted = steps if not mismatch.any() else mismatch[0].nonzero()[0].item()

        if accepted:
            input_ids = torch.cat([input_ids, draft_tokens[:, :accepted]], dim=1)
            generated += accepted
        if accepted < steps and generated < max_new_tokens:
            input_ids = torch.cat([input_ids, target_pred[:, accepted:accepted + 1]], dim=1)
            generated += 1

    return input_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 4.0, 2.0, -1.0]])
    assert greedy(logits).item() == 1
    assert top_k_sample(logits, k=1).item() == 1
    assert top_p_sample(logits, p=0.1).item() == 1

    penalized = apply_repetition_penalty(
        torch.tensor([[2.0, -2.0, 1.0]]), torch.tensor([[0, 1]]), penalty=2.0
    )
    assert torch.equal(penalized, torch.tensor([[1.0, -4.0, 1.0]]))

    class NextTokenModel:
        def __init__(self, vocab_size=6):
            self.vocab_size = vocab_size

        def __call__(self, ids):
            B, T = ids.shape
            out = torch.full((B, T, self.vocab_size), -100.0, device=ids.device)
            out.scatter_(2, ((ids + 1) % self.vocab_size).unsqueeze(-1), 100.0)
            return out

    model = NextTokenModel()
    prompt = torch.tensor([[0]])
    assert generate(model, prompt, 5, eos_id=3, pad_id=5).tolist() == [[0, 1, 2, 3]]
    assert beam_search(model, prompt, 2, 3, eos_id=3).tolist() == [[0, 1, 2, 3]]
    assert speculative_greedy(model, model, prompt, 5, gamma=3).tolist() == [[0, 1, 2, 3, 4, 5]]
    print("all decoding checks passed")
