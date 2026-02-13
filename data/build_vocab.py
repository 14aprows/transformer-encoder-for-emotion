import re
from collections import Counter
import torch 

class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens
    
    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        idx = len(self.word2idx)

        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def encode(self, text, max_len):
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(tok, self.word2idx["<unk>"]) for tok in tokens]
        ids = ids[:max_len]
        while len(ids) < max_len:
            ids.append(self.word2idx["<pad>"])
        return ids
    
    def __len__(self):
        return len(self.word2idx)