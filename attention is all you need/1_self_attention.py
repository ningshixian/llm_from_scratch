import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute self-attention output.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # Apply softmax row-wise to get attention weights
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Numerical stability
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Compute weighted sum of values
    attention_output = np.matmul(attention_weights, V)
    
    return attention_output


# X = np.arange(48).reshape(6,8)
# W_q = np.random.randint(0,4,size=(8,8)) 
# W_k = np.random.randint(0,5,size=(8,8)) 
# W_v = np.random.randint(0,6,size=(8,8)) 
# Q, K, V = compute_qkv(X, W_q, W_k, W_v) 
# print(self_attention(Q, K, V))