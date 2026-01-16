"""
极简 BPE (Byte-Pair Encoding) 算法教学代码
核心逻辑：
1. 将文本转换为 UTF-8 字节流 (0-255 的整数列表)
2. 统计相邻字节对的出现频率
3. 将出现频率最高的字节对合并为一个新的 Token ID
4. 重复上述步骤，直到达到预设的词表大小
"""
from collections import OrderedDict

def get_stats(ids, counts=None):
    """
    统计当前 id 列表中，所有相邻“对”(pair) 的出现频率
    例如: [1, 2, 1, 2, 3] -> (1,2):2, (2,3):1, (2,1):1
    """
    counts = {} if counts is None else counts
    # zip(ids, ids[1:]) 巧妙地错位生成相邻对
    for pair in zip(ids, ids[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    执行合并操作：将列表中所有的 pair 替换为新的 idx
    例如: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=256 
    结果 -> [256, 3, 256]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 检查是否匹配到了目标 pair，且不是最后一个元素
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx) # 替换为新 ID
            i += 2             # 跳过已被合并的两个元素
        else:
            newids.append(ids[i]) # 没匹配到，保持原样
            i += 1
    return newids


class BPETokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
    
    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 【关键】将文本转为 UTF-8 字节 (0-255 的整数)
        # 现代 LLM 不直接处理字符，而是处理字节，这样不需要担心未知字符
        # 例如 '中' -> [228, 184, 173]
        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes) 
        print("Raw Bytes:", ids)

        # 【核心】iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        # 初始化基础词表 0-255
        vocab = OrderedDict({idx: bytes([idx]) for idx in range(256)}) # int -> bytes 
        for i in range(num_merges):
            print(f"--- Iteration {i+1}/{num_merges} ---")
            # Step A: 统计频率
            stats = get_stats(ids)
            # Step B: 找出频率最高的对
            pair = max(stats, key=stats.get)
            print(f"Highest frequency pair: {pair} with frequency {stats.get(pair)}")
            # Step C: 分配新的 Token ID (从 256 开始递增)     
            idx = 256 + i
            # Step D: 执行合并
            ids = merge(ids, pair, idx)
            # save the merge，用于后续的 encode/decode
            merges[pair] = idx
            # 递归组合字节，构建解码词表
            # 原来的词不会剔除，而是在基础词表上累加
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            print("New Vocab Map:", list(vocab.items())[256:])
            print("-" * 28 + "\n")
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # 将 ID 还原为字节，再解码回字符串
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # 编码 (Encode: Text -> IDs) 
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255
        while len(ids) >= 2:
            stats = get_stats(ids)  # 统计pair 频率
            # 结果取min，是指merge对应idx越小，出现的频率越高❗️
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 如果找到的最小 ID 是无穷大，说明当前所有对都没学过，无法继续合并
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


# ==========================================
# 运行演示
# ==========================================

# 1. 准备数据 (包含重复模式以便压缩)
data = "aaababb good goodx 中文测试在线中文 "

# 2. 实例化并训练
tokenizer = BPETokenizer()
tokenizer.train(data, vocab_size=265) # 设定只学几个新词，方便观察

print("\n" + "=" * 30)
print("--- Final BPE Vocabulary Map ---")
for token_string, freq in list(tokenizer.vocab.items())[256:]:
    print(f"'{token_string}': {freq}")

print("\n--- Learned Merge Rules (合并规则) ---")
print("New ID | 组成成分 (ID + ID)  | 对应文本")
print("-------|--------------------|---------")
for (p0, p1), new_id in tokenizer.merges.items():
    # 解码对应的文本
    text_val = tokenizer.decode([new_id])
    # 如果是特殊字符或换行，转义一下方便显示
    text_val = repr(text_val) 
    print(f"{new_id:<6} | {p0:<3} + {p1:<3}          | {text_val}")

# 3. 测试 Encode (编码新文本)
# 注意："中文" 在训练数据里出现过，应该会被压缩成新的 ID
text = "中文 is good" 
encoded_ids = tokenizer.encode(text)
token_strs = [tokenizer.decode([i]) for i in encoded_ids]
print("-" * 60)
print(f"\n原文: {text}")
print(f"IDs : {encoded_ids}\n")

# 使用 " | " 作为边界展示
# replace('\n', '\\n') 是为了防止换行符破坏格式
visual_str = " | ".join([s.replace(' ', ' ') for s in token_strs])
print(f"切分: | {visual_str} |")
print(f"统计: 字符数 {len(text)} -> Token数 {len(encoded_ids)} (压缩率: {len(text)/len(encoded_ids):.2f}x)")
print("-" * 60)
