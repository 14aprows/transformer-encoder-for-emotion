import torch 
import torch.nn as nn
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        q = self.q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.float().masked_fill(
                mask == 0, float('-1e9')  
            ).type_as(scores)

        attn = torch.softmax(scores, dim=-1)
        context = attn @ v

        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(context)