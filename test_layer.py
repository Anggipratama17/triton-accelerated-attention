import torch
from triton_attention_layer import TritonAttention

torch.manual_seed(0)

# -------------------------------------------------
# FIX: Use dim=64 so head_dim = 16 (valid for Triton)
# -------------------------------------------------
B, N, C = 2, 16, 64
H = 4  # 64 / 4 = 16 per head (Triton requirement)

x = torch.randn(B, N, C, device="cuda")

attn_triton = TritonAttention(dim=C, num_heads=H).cuda()
attn_torch = torch.nn.MultiheadAttention(C, H, batch_first=True).cuda()

# Copy Q/K/V weights into torch MHA
attn_torch.in_proj_weight.data = torch.cat(
    [
        attn_triton.q_proj.weight,
        attn_triton.k_proj.weight,
        attn_triton.v_proj.weight,
    ],
    dim=0,
).detach().clone()

attn_torch.in_proj_bias.data = torch.cat(
    [
        attn_triton.q_proj.bias,
        attn_triton.k_proj.bias,
        attn_triton.v_proj.bias,
    ],
    dim=0,
).detach().clone()

attn_torch.out_proj.weight.data = attn_triton.out_proj.weight.detach().clone()
attn_torch.out_proj.bias.data = attn_triton.out_proj.bias.detach().clone()

# Run both
out_triton = attn_triton(x)
out_torch, _ = attn_torch(x, x, x)

max_diff = (out_triton - out_torch).abs().max().item()
print("Max difference:", max_diff)

if max_diff < 1e-3:
    print("PASS ✓")
else:
    print("FAIL ✘")
