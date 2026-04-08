
import torch
import torch.nn.functional as F

# ===============================================================================
# 计算DPO损失
# DPO的核心思想：增大策略模型对优质回答的概率，同时减小对低质量回答的概率
# ===============================================================================
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps, beta=0.1):
    # 策略模型相对于参考模型对好/坏回答的提升
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()  # 对所有样本取平均
    return loss


# Demo
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())