import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def masked_self_attention(Q, K, V):
    """
    Compute masked self-attention.
    """
    seq_len, d_k = Q.shape
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # bias = torch.tril(torch.ones(seq_len, seq_len)
    # scores = scores.masked_fill(bias == 0, float('-inf'))
    mask = np.triu(np.ones((seq_len, seq_len))*(-np.inf), k=1)  # k=1(主对角线向右上方移动 1 格,排除对角线)
    scores = scores + mask
    
    # Apply softmax row-wise to get attention weights
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Numerical stability
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    attention_output = np.matmul(attention_weights, V)
    return attention_output


# X = np.arange(48).reshape(6,8)
# W_q = np.random.randint(0,4,size=(8,8)) 
# W_k = np.random.randint(0,5,size=(8,8)) 
# W_v = np.random.randint(0,6,size=(8,8)) 
# Q, K, V = compute_qkv(X, W_q, W_k, W_v) 
# print(masked_self_attention(Q, K, V))