# ===============================================================================
# DPO（Direct Preference Optimization）算法实现
# DPO通过人类偏好数据直接优化语言模型，使其生成更符合人类偏好的输出
# 这里面使用了一个偏好prefer以及两个reject的格式
# ===============================================================================
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy

torch.manual_seed(0)

# 创建简化版的Llama模型作为策略模型（将被优化的模型）
policy_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=12, num_hidden_layers=1, hidden_size=32))

# 创建参考模型（通常是SFT模型，在训练过程中保持不变）
reference_model = deepcopy(policy_model)  # 深度复制确保两个模型初始参数完全相同

# 超参数
beta = 0.1  # DPO的温度系数，控制策略模型与参考模型的偏离程度，值越小允许偏离越大

# 准备训练数据
# 在DPO中，我们需要提示(prompt)、优选回答(chosen/good)和拒绝回答(rejected/bad)
prompt_ids = [1, 2, 3, 4, 5, 6]  # 输入提示的token IDs
good_response_ids = [7, 8, 9, 2]  # 优质回答的token IDs
# 多个低质量回答的示例，每个都是token IDs的列表
bad_response_ids_list = [[1, 2, 0, 0], [4, 5, 6, 0]]

# 构建模型输入：将提示与回答拼接
# 创建包含多个序列的批次：[提示+优质回答, 提示+低质回答1, 提示+低质回答2, ...]
input_ids = torch.LongTensor(
    [prompt_ids + good_response_ids, *[prompt_ids + bad_response_ids for bad_response_ids in bad_response_ids_list]]
)

# 准备用于计算语言模型损失的标签
# 在语言模型训练中，标签是输入向右移动一位（预测下一个token）
# -100表示在计算损失时忽略该位置（这里忽略提示部分）
labels = torch.LongTensor(
    [
        [-100] * len(prompt_ids) + good_response_ids,
        *[[-100] * len(prompt_ids) + bad_response_ids for bad_response_ids in bad_response_ids_list]
    ]
)

# 向右移动一位，因为我们预测的是下一个token
labels = labels[:, 1:]  

# 创建掩码，用于标识哪些位置参与损失计算（即回答部分）
loss_mask = (labels != -100)
print(loss_mask.shape)

# 将-100替换为0，因为在gather操作中-100是无效索引
labels[labels == -100] = 0
print(labels.shape)

output = policy_model(input_ids)
for key, value in output.items():  # 如果是ModelOutput对象，可以用output.__dict__.items()
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")

# ===============================================================================
# 计算策略模型（policy model）的对数概率
# ===============================================================================
# 前向传播，获取每个token位置的预测logits
logits = policy_model(input_ids)["logits"][:, :-1, :]  # 去掉最后一个位置，与label对齐
print(logits.shape)

# 将logits转换为对数概率，并提取每个位置上正确token的对数概率
per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

# 仅对回答部分（loss_mask=True的位置）求和，得到每个序列的总对数概率
all_logps = (per_token_logps * loss_mask).sum(-1)

# 分离优质回答和低质量回答的对数概率
policy_good_logps, policy_bad_logps = all_logps[:1], all_logps[1:]

# ===============================================================================
# 计算参考模型（reference model）的对数概率
# ===============================================================================
with torch.no_grad():  # 不计算梯度，因为参考模型不需要更新
    # 重复与策略模型相同的步骤
    logits = reference_model(input_ids)["logits"][:, :-1, :]
    print(logits.shape)
    print("logits:\n",logits)
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    print(per_token_logps.shape)
    print("per_token_logps:\n",per_token_logps)
    all_logps = (per_token_logps * loss_mask).sum(-1)
    print(all_logps.shape)
    print("all_logps\n",all_logps)
    reference_good_logps, reference_bad_logps = all_logps[:1], all_logps[1:]
    print(reference_good_logps.shape)
    print("reference_good_logps:\n",reference_good_logps)
    print(reference_bad_logps.shape)
    print("reference_bad_logps\n",reference_bad_logps)

# ===============================================================================
# 计算DPO损失
# DPO的核心思想：增大策略模型对优质回答的概率，同时减小对低质量回答的概率
# ===============================================================================
# 计算DPO的logits：(策略模型相对于参考模型对好回答的提升) - (对坏回答的提升)
logits = (policy_good_logps - reference_good_logps) - (policy_bad_logps - reference_bad_logps)
# 应用logsigmoid函数并乘以beta控制优化强度，取负值（因为要最小化损失）
loss = -F.logsigmoid(beta * logits).mean()  # 对所有样本取平均
print(loss)
