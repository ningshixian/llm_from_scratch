class SimpleBPE:
    def __init__(self):
        self.merges = []

    def _get_stats(self, vocab):
        """统计词表所有相邻符号对 (pair) 的频次"""
        pairs = {}
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def _merge_vocab(self, pair, vocab):
        """将词表中指定的符号对合并为新符号并返回新词表"""
        new_vocab = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, corpus, num_merges):
        """训练 BPE：根据语料构建基础词表并迭代合并最高频符号对"""
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ('</w>',)
            vocab[symbols] = vocab.get(symbols, 0) + 1

        self.merges = []
        for _ in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            vocab = self._merge_vocab(best, vocab)

    def tokenize(self, text):
        """使用训练好的 merge 规则对输入文本进行切词"""
        all_tokens = []
        for word in text.split():
            symbols = list(word) + ['</w>']
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols = symbols[:i] + [a + b] + symbols[i + 2:]
                    else:
                        i += 1
            all_tokens.extend(symbols)
        return all_tokens


# Demo
bpe = SimpleBPE()
bpe.fit(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
print('Merges:', bpe.merges)
print('Tokenize:', bpe.tokenize('low lower newest'))
