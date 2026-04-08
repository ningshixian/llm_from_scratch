import math
import torch
import torch.nn as nn
from functools import partial
from transformers import AutoModelForSequenceClassification

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank   # 关键：LoRA 一般用 alpha / rank

        std_dev = 1 / math.sqrt(rank)
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)    # 随机初始化
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))             # 零初始化

    def forward(self, x):
        x = self.scaling * (x @ self.A @ self.B)
        return x

# 将每个线性层替换为新的 LinearWithLoRA 层
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear

        # 再保险：冻结原始 linear 参数
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


"""
# 示例：如何将 LoRA 层集成到现有模型中
"""

# 1.先加载模型
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2.冻结所有参数
for param in model.parameters():
    param.requires_grad = False
# 然后把分类头打开
for param in model.classifier.parameters():
    param.requires_grad = True

# 3.替换 attention 中的指定线性层为 LoRA 层
lora_r = 8
lora_alpha = 16
assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)
# target_modules=("q_proj", "v_proj")  # LLaMA / Qwen
target_modules = ["q_lin", "v_lin"]   # DistilBERT
for layer in model.distilbert.transformer.layer:
    if "q_lin" in target_modules:
        layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    if "k_lin" in target_modules:
        layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    if "v_lin" in target_modules:
        layer.attention.v_lin = assign_lora(layer.attention.v_lin)
    if "out_lin" in target_modules:
        layer.attention.out_lin = assign_lora(layer.attention.out_lin)

# 4.只训练 LoRA 参数
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

total_params = sum(p.numel() for p in model.parameters())
trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params}")
print(f"Trainable params: {trainable_count}")
print(f"Trainable ratio: {100 * trainable_count / total_params:.4f}%")


"""
使用 peft 实现 LoRA 微调
"""

# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModel
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# from transformers import Trainer

# # 加载基座模型
# MODEL_PATH = "qwen-1.5b"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModel.from_pretrained(
#     MODEL_PATH, trust_remote_code=True
# )

# # 设定 peft 参数：
# peft_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         inference_mode=False,
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.1,
#     )

# # 获取 LoRA 模型
# model = get_peft_model(model, peft_config)

# # 训练
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset= IterableWrapper(train_dataset),
#     tokenizer=tokenizer
# )
# trainer.train()
