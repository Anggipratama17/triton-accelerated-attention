import torch
from mini_transformer import MiniTransformer

model = MiniTransformer(dim=64, hidden_dim=128).cuda()
x = torch.randn((64, 64), device='cuda')
y = model(x)
print("Output shape:", y.shape)
