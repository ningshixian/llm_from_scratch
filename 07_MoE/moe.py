
def moe(x: torch.Tensor, We: torch.Tensor, Wg: torch.Tensor, n_experts: int, top_k: int) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape
    
    # Flatten batch and sequence dimensions
    x_flat = x.reshape(-1, d_model)
    n_tokens = x_flat.shape[0]
    
    # Compute gating logits and apply softmax
    gating_logits = torch.matmul(x_flat, Wg)
    gating_weights = torch.softmax(gating_logits, dim=-1)
    
    # Get top-k experts for each token
    topk_weights, topk_idx = torch.topk(gating_weights, top_k, dim=-1)
    
    # Normalize top-k weights
    topk_weights_norm = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    
    # Flatten for indexing
    topk_idx_flat = topk_idx.flatten()
    token_idx_flat = torch.arange(n_tokens, device=x.device).repeat_interleave(top_k)
    topk_weights_norm_flat = topk_weights_norm.flatten()
    
    # Prepare output
    output_flat = torch.zeros_like(x_flat)
    
    # Process each expert
    for i in range(n_experts):
        mask = topk_idx_flat == i
        tokens_expert_i = token_idx_flat[mask]
        
        if tokens_expert_i.numel() > 0:
            x_expert_i = x_flat[tokens_expert_i]
            output_expert_i = torch.matmul(x_expert_i, We[i])
            output_expert_i = output_expert_i * topk_weights_norm_flat[mask].unsqueeze(-1)
            
            # Scatter add to output
            output_flat.index_add_(0, tokens_expert_i, output_expert_i)
    
    return output_flat.reshape(n_batch, l_seq, d_model)
