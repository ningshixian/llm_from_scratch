import torch.nn.functional as F

def causal_lm_loss(
    logits, # [batch_size, seq_len, hidden_dim]
    labels, # [batch_size, seq_len]
    pad_token_id: int = 0, 
    **kwargs
):
    # 取最后一个token前的所有token
    shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1)) # [batch_size * (seq_len-1), hidden_dim]
    # 取第一个token后的所有token
    shift_labels = labels[..., 1:].contiguous().view(-1) # [batch_size * (seq_len-1)]
    
    loss = F.cross_entropy(
        shift_logits, 
        shift_labels, 
        ignore_index=pad_token_id, 
        reduction='mean', 
        **kwargs
    )
    
    return loss
