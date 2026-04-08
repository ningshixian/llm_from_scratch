import numpy as np

def moe_shared_expert_forward(
    X: np.ndarray,
    W_gate: np.ndarray,
    W_shared: np.ndarray,
    W_experts: list,
    top_k: int = 2
) -> dict:
    """
    Forward pass of a Mixture of Experts layer with a shared expert.
    """
    num_tokens = X.shape[0]
    num_experts = len(W_experts)
    d_out = W_shared.shape[1]
    
    # Compute gating logits for routed experts
    logits = X @ W_gate  # (num_tokens, num_experts)
    
    # Softmax over routed experts
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    gate_scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Select top-k experts per token (sorted descending by score)
    top_k_indices = np.argsort(gate_scores, axis=1)[:, -top_k:][:, ::-1]
    
    # Gather top-k scores
    top_k_scores = np.zeros((num_tokens, top_k))
    for i in range(num_tokens):
        top_k_scores[i] = gate_scores[i, top_k_indices[i]]
    
    # Renormalize top-k scores to sum to 1
    top_k_scores = top_k_scores / np.sum(top_k_scores, axis=1, keepdims=True)
    
    # Compute shared expert output (always active for all tokens)
    shared_output = X @ W_shared  # (num_tokens, d_out)
    
    # Compute weighted routed expert outputs
    routed_output = np.zeros((num_tokens, d_out))
    for i in range(num_tokens):
        for j in range(top_k):
            expert_idx = top_k_indices[i, j]
            expert_out = X[i] @ W_experts[expert_idx]
            routed_output[i] += top_k_scores[i, j] * expert_out
    
    # Final output: shared + routed
    output = shared_output + routed_output
    
    return {
        'output': output,
        'shared_output': shared_output,
        'routed_output': routed_output,
        'routing_indices': top_k_indices,
        'routing_weights': top_k_scores
    }