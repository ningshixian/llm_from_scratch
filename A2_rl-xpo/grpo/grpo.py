# https://github.com/dhcode-cpp/grpo-loss

import torch
import torch.nn.functional as F
torch.manual_seed(42)

def compute_group_advantages(rewards):
    """
    rewards: (B, G)
    return:  (B, G)
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    # keepdim=True 方便和rewards做广播运算，unbiased=False 使用样本标准差的无偏估计
    return (rewards - mean) / (std + 1e-4)

def kl_divergence_estimator(logp_theta, logp_ref):
    # Per-token KL surrogate.
    log_ratio = logp_ref - logp_theta
    return torch.exp(log_ratio) - log_ratio - 1


def grpo_loss(
    logp_theta: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    rewards: torch.Tensor,		    # 序列级别的奖励 (Batch*G,)
    mask: torch.Tensor,   # 掩码 (Batch*G, Seq_Len)，1为有效token，0为padding
    group_size: int,
    beta: float = 0.01,			# KL 散度的系数
    clip_eps: float = 0.2		# PPO 裁剪系数
) -> torch.Tensor:

    # 1) 组内 advantage: (B, G) -> (B*G, 1)
    rewards = rewards.view(-1, group_size)    # (B, G)
    A = compute_group_advantages(rewards)     # (B, G)
    A = A.reshape(-1, 1)                      # (B*G, 1)
    # A = A.flatten().unsqueeze(-1)           # (B*G, 1)

    # 2) PPO-style clipped objective
    ratio = torch.exp(logp_theta - logp_old)  # (B*G, T)
    ratio_clip = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    surr1 = ratio * A
    surr2 = ratio_clip * A
    policy_loss = torch.minimum(surr1, surr2) # (Batch*G, Seq_Len)

    # 3) KL penalty
    kl = kl_divergence_estimator(logp_theta, logp_ref)  # (B*G, Seq_Len)

    # 4) final token-level loss
    token_loss = -policy_loss + beta * kl
    token_loss = token_loss * mask			# 屏蔽 Padding Token
    loss = token_loss.sum() / mask.sum()		# 只对有效 Token 求平均
    return loss # 标量


if __name__ == "__main__":
    B, G, T = 2, 3, 5
    BG = B * G

    pi_theta = torch.randn(BG, T, requires_grad=True)
    pi_old = torch.randn(BG, T)
    pi_ref = torch.randn(BG, T)
    # 获取log prob
    logp_theta = F.log_softmax(pi_theta, dim=-1)
    logp_ref = F.log_softmax(pi_ref, dim=-1)
    logp_old = F.log_softmax(pi_old, dim=-1)

    rewards = torch.randn(BG)
    mask = torch.ones(BG, T)

    print(kl_divergence_estimator(logp_theta, logp_ref))

    loss = grpo_loss(
        logp_theta=logp_theta,
        logp_old=logp_old,
        logp_ref=logp_ref,
        rewards=rewards,
        mask=mask,
        group_size=G
    )

    print("loss:", loss.item())
    
    # 检查 loss 是否正常
    assert loss.dim() == 0, f"loss 应该是标量，但现在 shape={loss.shape}"
    assert torch.isfinite(loss), "loss 出现 NaN 或 Inf"
