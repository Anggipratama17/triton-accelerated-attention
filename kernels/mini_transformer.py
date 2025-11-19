import torch
import torch.nn as nn
from kernels.triton_attention_layer import TritonAttentionLayer

class MiniTransformer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.attn = TritonAttentionLayer(dim=dim, num_heads=8)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # Attention
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ff_out = self.linear2(self.activation(self.linear1(x)))
        x = self.norm2(x + ff_out)
        return x
