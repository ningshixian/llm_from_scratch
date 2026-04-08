import numpy as np
from collections import Counter

def bleu_score(candidate, references, max_n=4):
    def get_ngrams(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    if not candidate or not references:
        return 0.0

    c = len(candidate)
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda x: abs(x - c))  # 选最接近的参考长度
    bp = 1.0 if c > r else np.exp(1 - r / c)

    score = 0.0
    for n in range(1, max_n + 1):
        if len(candidate) < n:
            return 0.0

        cand_ngrams = get_ngrams(candidate, n)
        ref_max = Counter()

        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            for g in ref_ngrams:
                ref_max[g] = max(ref_max[g], ref_ngrams[g])

        match = sum(min(cand_ngrams[g], ref_max[g]) for g in cand_ngrams)
        total = sum(cand_ngrams.values())

        if match == 0:
            return 0.0

        score += np.log(match / total)

    return float(bp * np.exp(score / max_n))