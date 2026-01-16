from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import torch

# LoRA 微调 Hugging Face Transformers 示例
# 适用于 OPT-125M 小模型，CPU/GPU 均可运行


def main():
    # 1. 加载预训练模型和分词器
    model_name = "facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # 添加 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 配置 LoRA 并应用 PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # 增加 rank
        lora_alpha=32,  # 增加 alpha
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]  # 增加更多目标模块
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 构造训练数据（增加更多相似样本）
    train_data = [
        {"prompt": "Q: Who are you?\nA:", "response": " I am an AI language model."},
        {"prompt": "Q: Hello, who are you?\nA:", "response": " I am an AI language model."},
        {"prompt": "Q: What are you?\nA:", "response": " I am an AI language model."},
        {"prompt": "Q: Can you introduce yourself?\nA:", "response": " I am an AI language model."},
        {"prompt": "Q: What's the weather today?\nA:", "response": " Sorry, I can't access real-time weather information."},
        {"prompt": "Q: How is the weather?\nA:", "response": " Sorry, I can't access real-time weather information."}
    ]
    dataset = Dataset.from_list(train_data)

    # 4. 数据预处理函数
    def preprocess(example):
        text = example["prompt"] + example["response"]
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=128)
        
        # 修正标签处理
        prompt_tokens = tokenizer(example["prompt"], add_special_tokens=False)
        prompt_len = len(prompt_tokens["input_ids"])
        
        labels = tokens["input_ids"].copy()
        # 将 prompt 部分的标签设为 -100，只计算 response 部分的损失
        labels[:prompt_len] = [-100] * prompt_len
        tokens["labels"] = labels
        return tokens

    train_dataset = dataset.map(preprocess, remove_columns=["prompt", "response"])

    # 5. 训练参数设置
    training_args = TrainingArguments(
        output_dir="lora-opt125m",
        per_device_train_batch_size=1,
        num_train_epochs=10,  # 增加训练轮次
        learning_rate=2e-4,   # 增加学习率
        logging_steps=1,
        save_steps=50,
        save_total_limit=1,
        warmup_steps=2,       # 添加预热
        gradient_accumulation_steps=2,  # 增加梯度累积
        dataloader_pin_memory=False,    # 禁用 pin_memory 避免 MPS 警告
        remove_unused_columns=False     # 避免列移除警告
    )

    # 6. 微调训练
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    model.save_pretrained("lora-opt125m")

    print("训练完成，LoRA 适配器已保存至 lora-opt125m/")

    # 7. 推理示例：加载 LoRA 权重并生成回答
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_model = PeftModel.from_pretrained(base_model, "lora-opt125m")
    lora_model.eval()

    # 使用与训练数据完全一致的 prompt 格式
    prompt = "Q: Hello, who are you?\nA:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # 测试不同的生成设置
    print("\n=== 不同生成设置的对比 ===")
    
    # 1. 贪婪搜索（确定性输出）
    print("\n1. 贪婪搜索 (do_sample=False):")
    with torch.no_grad():
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        output_ids = lora_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
        if '\n' in answer:
            answer = answer.split('\n')[0]
        print(f"输出：{answer}")
    
    # 2. 低温度采样（偏向确定性）
    print("\n2. 低温度采样 (temperature=0.1):")
    with torch.no_grad():
        output_ids = lora_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
        if '\n' in answer:
            answer = answer.split('\n')[0]
        print(f"输出：{answer}")
    
    # 3. 中等温度采样（平衡创造性和一致性）
    print("\n3. 中等温度采样 (temperature=0.7):")
    with torch.no_grad():
        output_ids = lora_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=15,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
        if '\n' in answer:
            answer = answer.split('\n')[0]
        print(f"输出：{answer}")
    
    # 4. 高温度采样（更有创造性）
    print("\n4. 高温度采样 (temperature=1.0):")
    with torch.no_grad():
        output_ids = lora_model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=15,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        answer = tokenizer.decode(output_ids[0][input_ids.size(1):], skip_special_tokens=True).strip()
        if '\n' in answer:
            answer = answer.split('\n')[0]
        print(f"输出：{answer}")
    
    print("\n=== Temperature 参数说明 ===")
    print("• temperature=0.1: 输出更确定、保守")
    print("• temperature=0.7: 平衡确定性和创造性")
    print("• temperature=1.0: 输出更多样化、创造性")
    print("• do_sample=False: 贪婪搜索，完全确定性输出")

if __name__ == "__main__":
    main()