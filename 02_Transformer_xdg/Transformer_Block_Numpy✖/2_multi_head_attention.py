import numpy as np
from typing import Tuple

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = Q.shape[-1]
    
    # Scaled dot-product attention scores
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    # Numerically stable softmax
    score_max = np.max(scores, axis=-1, keepdims=True)
    attention_weights = np.exp(scores - score_max) / np.sum(np.exp(scores - score_max), axis=-1, keepdims=True)
    
    # Weighted sum of values
    attention_output = np.matmul(attention_weights, V)
    return attention_output

def multi_head_attention(Q, K, V, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0
    d_k = d_model // n_heads
    
    # Reshape to (n_heads, seq_len, d_k)
    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    
    # Compute attention for each head
    head_outputs = []
    for i in range(n_heads):
        head_output = self_attention(Q_heads[i], K_heads[i], V_heads[i])
        head_outputs.append(head_output)
    
    # Concatenate heads
    output = np.concatenate(head_outputs, axis=-1)
    return output