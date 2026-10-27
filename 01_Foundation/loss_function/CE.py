import torch
import torch.nn.functional as F

def ce_loss(predict, target):
    """
    Args:
        logits: 模型的未归一化输出 (形状: [batch_size, num_classes])
        target: 若为类别索引 (形状: [batch_size])，则为每个样本的类别；
                若为概率分布 (形状: [batch_size, num_classes])，则为每个样本上各类别的分布。
    Returns:
        loss: 交叉熵损失标量
    """
    log_prob=F.log_softmax(predict, dim=-1)

    if target.dim()==1:
        loss=-log_prob.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    else:
        loss=-(target*log_prob).sum(dim=-1)
    return loss.mean()


# pytorch实现
def ce_loss(logits, targets):
    log_sum_exp = torch.logsumexp(logits, dim=1)  # (B,)
    true_logits = logits[torch.arange(logits.size(0)), targets]  # (B,)
    loss = -true_logits + log_sum_exp
    return loss.mean()


logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', ce_loss(logits, targets))
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets))