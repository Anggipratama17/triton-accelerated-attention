import torch
import torch.nn.functional as F
from triton_attention_layer import TritonAttention

torch.manual_seed(0)

# small, single-head test
B, N, C = 1, 16, 64
H = 4

x = torch.randn(B, N, C, device="cuda")

attn_triton = TritonAttention(dim=C, num_heads=H).cuda()
attn_torch = torch.nn.MultiheadAttention(C, H, batch_first=True).cuda()

# copy weights into torch MHA
attn_torch.in_proj_weight.data = torch.cat(
    [attn_triton.q_proj.weight,
     attn_triton.k_proj.weight,
     attn_triton.v_proj.weight], dim=0
).detach().clone()

attn_torch.in_proj_bias.data = torch.cat(
    [attn_triton.q_proj.bias,
     attn_triton.k_proj.bias,
     attn_triton.v_proj.bias], dim=0
).detach().clone()

attn_torch.out_proj.weight.data = attn_triton.out_proj.weight.detach().clone()
attn_torch.out_proj.bias.data = attn_triton.out_proj.bias.detach().clone()

# Run both
out_t = attn_torch(x, x, x)[0]
out_tr = attn_triton(x)

print("Torch:", out_t)
print("Triton:", out_tr)

print("Max difference:", (out_t - out_tr).abs().max().item())