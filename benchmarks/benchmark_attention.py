import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import time

from kernels.triton_attention_layer import TritonAttentionLayer
from kernels.triton_attention_softmax import attn_softmax_kernel
from kernels.triton_attention_scores import attention_scores_kernel
from kernels.triton_attention_values import attention_values_kernel


def benchmark_attention():
    torch.manual_seed(0)

    B = 1
    N = 4096   # sequence length
    C = 64     # embedding dimension
    H = 4      # number of heads

    x = torch.randn(B, N, C, device="cuda")

    # Instantiate Triton + PyTorch attention layers
    triton_attn = TritonAttentionLayer(dim=C, num_heads=H).cuda()
    torch_attn = torch.nn.MultiheadAttention(C, H, batch_first=True).cuda()

    # Copy projection weights so both implementations match
    with torch.no_grad():
        torch_attn.in_proj_weight.copy_(
            torch.cat([
                triton_attn.q_proj.weight,
                triton_attn.k_proj.weight,
                triton_attn.v_proj.weight,
            ], dim=0)
        )

        torch_attn.in_proj_bias.zero_()
        torch_attn.out_proj.weight.copy_(triton_attn.out_proj.weight)
        torch_attn.out_proj.bias.zero_()

    # Warmup
    for _ in range(5):
        triton_attn(x)
    torch.cuda.synchronize()

    # Triton timing
    start = time.time()
    for _ in range(20):
        triton_attn(x)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / 20

    # PyTorch timing
    start = time.time()
    for _ in range(20):
        torch_attn(x, x, x)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 20

    print(f"Triton attention:   {triton_time*1000:.3f} ms")
    print(f"PyTorch attention:  {torch_time*1000:.3f} ms")

    with open("benchmark_results.txt", "w") as f:
        f.write(f"{triton_time} {torch_time}\n")


if __name__ == "__main__":
    benchmark_attention()
