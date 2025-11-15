import torch

B, N, C = 2, 16, 32
H = 4
D = C // H

x = torch.randn(B, N, C, device="cuda")

q = x.view(B, N, H, D).transpose(1,2).contiguous()  # B,H,N,D
print("q.shape:", q.shape)
print("q.stride:", q.stride())

k = q.clone()
v = q.clone()

print("\nExpected:")
print("  B =", B)
print("  H =", H)
print("  N =", N)
print("  D =", D)
print("\nStrides of q should be:")
print("  (H*N*D, N*D, D, 1)")
