# ✅ SOLUTION

class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ('</w>',)
            vocab[symbols] = vocab.get(symbols, 0) + 1
        self.merges = []
        for _ in range(num_merges):
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

    def encode(self, text):
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
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
print('Merges:', bpe.merges)
print('Encode:', bpe.encode('low lower newest'))