import torch
from torch.utils.data import Dataset, DataLoader

"""
构造 SFT 数据集
SFT 训练时，我们只对模型的“回答（Assistant）”部分计算 Loss，而忽略“提问（User/System）”部分。
"""

class TokenSFTDataset(Dataset):
    def __init__(self, messages_list, tokenizer):
        self.tokenizer = tokenizer
        # 预定义特殊 Token
        self.sos_token_id = tokenizer('<|im_start|>').input_ids[0]
        self.eos_token_id = tokenizer('<|im_end|>').input_ids[0]
        self.data = [self.process_item(m) for m in messages_list]
        
    def process_item(self, example):
        input_ids = [self.sos_token_id]
        labels = [-100]  # 初始化 labels，-100 会在 CrossEntropyLoss 中被忽略
        
        for item in example:
            role = item['role']
            content = item['content']
            
            # 构造格式化的文本：例如 "\n#USER:什么是人工智能?"
            prompt_text = f"\n#{role}:{content}"
            token_ids = self.tokenizer(prompt_text).input_ids
            
            input_ids += token_ids
            
            # 只有 ASSISTANT 的内容部分需要计算 Loss (Label = Input_ID)
            if role == 'ASSISTANT':
                # 加上结束符，保证模型学会停止
                input_ids.append(self.eos_token_id)
                labels += token_ids + [self.eos_token_id]
            else:
                # 非回答部分全部填充 -100
                labels += [-100] * len(token_ids)
                
        return torch.tensor(input_ids), torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx][0], "labels": self.data[idx][1]}

"""
数据整理与 Padding
在 Batch 训练中，不同句子的长度不一，需要通过 collate_fn 进行右侧填充（Right Padding）。
"""

def sft_collate_fn(batch, pad_token_id=0):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 获取当前 Batch 的最大长度
    max_len = max([ids.shape[0] for ids in input_ids])
    
    # Padding input_ids 和 labels
    padded_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, lbs) in enumerate(zip(input_ids, labels)):
        curr_len = ids.shape[0]
        padded_input_ids[i, :curr_len] = ids
        padded_labels[i, :curr_len] = lbs
        attention_mask[i, :curr_len] = 1 # 有效 Token 为 1

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }

"""
训练循环（PyTorch 原生版）
这部分展示了如何加载模型并执行梯度更新。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.optim as optim

# 1. 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

# 2. 准备数据 (示例数据见资料)
messages_list = [[
    {'role':'SYSTEM', 'content':'你是智能助手'},
    {'role':'USER', 'content':'计算 1+1'},
    {'role':'ASSISTANT', 'content':'1+1等于2'}
]]
dataset = TokenSFTDataset(messages_list, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=sft_collate_fn)

# 3. 设置优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) # 忽略 -100 的 Loss

# 4. 训练
model.train()
for epoch in range(1):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits # 形状: [batch, seq_len, vocab_size]
        
        # 计算 Loss：需要错开一位（Label 是 Input 的下一位预测）
        # 注意：在简单的实现中，通常使用 CrossEntropyLoss 的 view 处理
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")