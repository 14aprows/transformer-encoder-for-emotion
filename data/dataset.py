import torch 
from torch.utils.data import Dataset
from datasets import load_dataset
from .build_vocab import Vocab

class EmotionDataset(Dataset):
    def __init__(self, split, vocab, max_len):
        self.data = load_dataset("emotion")[split]
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]
        
        input_ids = torch.tensor(self.vocab.encode(text, self.max_len))

        attention_mask = (input_ids != 1).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label)
        }