import torch
import torch.nn as nn
from .embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .encoder import EncoderLayer

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=6, embed_dim=256, num_heads=8, num_layers=4, ff_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.pool(x.transpose(1,2)).squeeze(-1)
        x = self.fc(x)
        return x