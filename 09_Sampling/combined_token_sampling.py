import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
    """
    Implement a combined token sampling pipeline commonly used in language model text generation. Given a list of raw logits (unnormalized scores for each token in the vocabulary), apply three filtering/scaling stages in sequence to produce a final probability distribution and sample a token from it.
    
    Args:
        logits: Raw unnormalized scores for each token
        temperature: Scaling factor for logit sharpness (0 = greedy)
        top_k: Number of top tokens to keep (0 = disabled)
        top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
        seed: Random seed for reproducible sampling
    
    Returns:
        Dict with 'probabilities' (list of floats rounded to 4 decimals)
        and 'sampled_token' (int index of chosen token)
    """
    logits = np.asarray(logits, dtype=np.float64)
    vocab_size = logits.shape[0]

    if vocab_size == 0:
        raise ValueError("logits must not be empty")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    rng = np.random.default_rng(seed)

    # 1) temperature = 0 -> greedy
    if temperature == 0:
        probs = np.zeros(vocab_size, dtype=np.float64)
        sampled_token = int(np.argmax(logits))
        probs[sampled_token] = 1.0
        return {
            "probabilities": np.round(probs, 4).tolist(),
            "sampled_token": sampled_token
        }

    # 2) temperature scaling
    scaled_logits = logits / temperature

    # 3) softmax
    scaled_logits = scaled_logits - np.max(scaled_logits)  # 数值稳定
    exp_logits = np.exp(scaled_logits)
    probs = exp_logits / np.sum(exp_logits)

    # 4) top-k filtering
    if top_k > 0 and top_k < vocab_size:
        topk_indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros(vocab_size, dtype=bool)
        mask[topk_indices] = True
        probs = np.where(mask, probs, 0.0)

    # 5) top-p filtering
    if top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # 保留达到 top_p 所需的最小集合
        cutoff = np.searchsorted(cumulative_probs, top_p, side="left") + 1
        keep_indices = sorted_indices[:cutoff]

        new_probs = np.zeros_like(probs)
        new_probs[keep_indices] = probs[keep_indices]
        probs = new_probs

    # 6) 重新归一化
    prob_sum = np.sum(probs)
    if prob_sum == 0:
        # 极端情况下兜底：退化成 greedy
        sampled_token = int(np.argmax(logits))
        probs = np.zeros(vocab_size, dtype=np.float64)
        probs[sampled_token] = 1.0
    else:
        probs = probs / prob_sum
        sampled_token = int(rng.choice(vocab_size, p=probs))

    return {
        "probabilities": np.round(probs, 4).tolist(),
        "sampled_token": sampled_token
    }