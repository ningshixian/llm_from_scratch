import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax
    """
    logits = logits - np.max(logits)
    exp_x = np.exp(logits)
    return exp_x / np.sum(exp_x)


def top_k_sampling(logits: np.ndarray, top_k: int = None) -> int:
    """
    Top-k sampling with numpy

    Args:
        logits: shape (vocab_size,)
        top_k: keep only top_k tokens for sampling

    Returns:
        sampled token index
    """
    vocab_size = logits.shape[0]
    if top_k is None or top_k <= 0 or top_k >= vocab_size:
        probs = softmax(logits)
        return np.random.choice(vocab_size, p=probs)

    # 取前 k 大的下标（不要求有序）
    topk_indices = np.argsort(probs)[::-1][:top_k]
    topk_logits = logits[topk_indices]

    filtered_probs = np.zeros_like(probs)
    filtered_probs[topk_indices] = probs[topk_indices]
    filtered_probs = filtered_probs / np.sum(filtered_probs)

    # 在 top-k 对应的原始 token id 中采样
    sampled_token = np.random.choice(topk_indices, p=filtered_probs[topk_indices])
    return int(sampled_token)


def top_p_sampling(logits: np.ndarray, p: float) -> int:
    """
    Top-p (nucleus) sampling with numpy

    Args:
        logits: shape (vocab_size,)
        p: cumulative probability threshold, 0 < p <= 1

    Returns:
        sampled token index
    """
    if not (0 < p <= 1):
        raise ValueError(f"p must be in (0, 1], but got {p}")

    # 先转概率
    probs = softmax(logits)

    # 按概率从大到小排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # 累积概率
    cumulative_probs = np.cumsum(sorted_probs)

    # 保留累计概率刚好达到/超过 p 的最小集合
    cutoff = np.searchsorted(cumulative_probs, p, side="left") + 1
    keep_sorted_indices = sorted_indices[:cutoff]

    # 构造完整长度的过滤后概率
    filtered_probs = np.zeros_like(probs)
    filtered_probs[keep_sorted_indices] = probs[keep_sorted_indices]

    # 重新归一化
    filtered_probs = filtered_probs / np.sum(filtered_probs)

    sampled_token = np.random.choice(keep_sorted_indices, p=filtered_probs[keep_sorted_indices])
    return int(sampled_token)


logits = np.array([2.1, 1.3, 0.5, 3.2, 0.1])

token_k = top_k_sampling(logits, top_k=3)
token_p = top_p_sampling(logits, p=0.8)

print("top-k sampled token:", token_k)
print("top-p sampled token:", token_p)