import torch
from triton_attention_layer import TritonAttention

torch.manual_seed(0)

B, N, C = 2, 16, 32
H = 4

x = torch.randn(B, N, C, device="cuda")

attn = TritonAttention(dim=C, num_heads=H).cuda()

q = attn.q_proj(x).view(B, H, N, C // H).contiguous()
k = attn.k_proj(x).view(B, H, N, C // H).contiguous()
v = attn.v_proj(x).view(B, H, N, C // H).contiguous()

print("q.shape:", q.shape)
print("q.stride:", q.stride())
print("k.stride:", k.stride())
print("v.stride:", v.stride())