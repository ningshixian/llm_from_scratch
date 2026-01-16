## LLM 算法原理学习和工程实验仓库

[llm_from_scratch](https://github.com/ningshixian/llm_from_scratch)：LLM 算法原理和工程实验仓库，包括 Transformer 的基本组件实现、KV 缓存、**GPT 类主流大模型架构、minimind-LLM 全阶段极简复现、Post-Traning（SFT、DPO、GRPO...）、**

* [bpe](https://github.com/ningshixian/llm_from_scratch/tree/main/bpe)：学习和复现了BPE tokenizer
* [position](https://github.com/ningshixian/llm_from_scratch/tree/main/%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81)：常见的位置编码实现 RoPE、YaRN
* [attention](https://github.com/ningshixian/llm_from_scratch/tree/main/attention)：一些常见的注意力机制 MHA/GQA/MLA
* [kv-cache](https://github.com/ningshixian/llm_from_scratch/tree/main/kv-cache)：学习 KV 缓存是如何用空间换时间滴.
* [moe](https://github.com/ningshixian/llm_from_scratch/tree/main/moe)：了解下 MOE 架构的实现
* [gpt](https://github.com/ningshixian/llm_from_scratch/tree/main/gpt)、[llama](https://github.com/ningshixian/llm_from_scratch/tree/main/llama)、[olmo3](https://github.com/ningshixian/llm_from_scratch/tree/main/olmo3)、[qwen3](https://github.com/ningshixian/llm_from_scratch/tree/main/qwen3)：**复现一些 decoder-only的 GPT 类热门模型**
* [SFT](https://github.com/ningshixian/llm_from_scratch/tree/main/sft)：包括**指令微调、LoRA微调.....**
* [DPO](https://github.com/ningshixian/llm_from_scratch/tree/main/dpo)、[GRPO](https://github.com/ningshixian/llm_from_scratch/tree/main/grpo)、[KTO](https://github.com/ningshixian/llm_from_scratch/tree/main/kto)、[PPO](https://github.com/ningshixian/llm_from_scratch/tree/main/ppo)、[reinforce++](https://github.com/ningshixian/llm_from_scratch/tree/main/reinforce%2B%2B)
* [QwenGRPO.ipynb](https://github.com/ningshixian/llm_from_scratch/blob/main/grpo/QwenGRPO.ipynb)：基于通义千问2.5的0.5B模型，复现DeepSeek R1的顿悟时刻
* [minimind-LLM全阶段极简复现](https://github.com/ningshixian/llm_from_scratch/tree/main/%23minimind-LLM%E5%85%A8%E9%98%B6%E6%AE%B5%E6%9E%81%E7%AE%80%E5%A4%8D%E7%8E%B0%20) **：** **很 nice 的一个项目，实现了一个极简结构的 LLM，以及全阶段的训练过程：数据集清洗、预训练(Pretrain)、监督微调(SFT)、LoRA微调、直接偏好优化(DPO)、强化学习训练(RLAIF: PPO/GRPO等)**
