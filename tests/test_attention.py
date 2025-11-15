import torch
from triton_attention_layer import TritonAttentionLayer

torch.manual_seed(0)

B, N, C = 2, 32, 64
H = 4

device = "cuda"

x = torch.randn(B, N, C, device=device)

# Triton version
attn_triton = TritonAttentionLayer(dim=C, num_heads=H).to(device)

# PyTorch MHA version
attn_torch = torch.nn.MultiheadAttention(
    embed_dim=C,
    num_heads=H,
    batch_first=True,
    bias=False,   # IMPORTANT: match Triton
).to(device)

# Copy weights from Triton → PyTorch
attn_torch.in_proj_weight.data = torch.cat([
    attn_triton.q_proj.weight.detach().clone(),
    attn_triton.k_proj.weight.detach().clone(),
    attn_triton.v_proj.weight.detach().clone(),
], dim=0)

attn_torch.out_proj.weight.data = attn_triton.out_proj.weight.detach().clone()

# Run both
out_triton = attn_triton(x)
out_torch, _ = attn_torch(x, x, x)

max_diff = (out_triton - out_torch).abs().max().item()
print("Max difference:", max_diff)

if max_diff < 1e-3:
    print("PASS ✓")
else:
    print("FAIL ✘")
