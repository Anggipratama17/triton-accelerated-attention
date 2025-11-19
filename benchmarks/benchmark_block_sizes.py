import torch
import triton
import triton.language as tl
import time
import matplotlib.pyplot as plt

from kernels.triton_attention_scores import attention_scores_kernel
from kernels.triton_attention_softmax import attn_softmax_kernel
from kernels.triton_attention_values import attention_values_kernel


# A small wrapper so we can call the full attention pipeline
class TritonAttentionBlock(torch.nn.Module):
    def __init__(self, dim=64, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.v = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q(x).view(B, H, N, D).contiguous()
        k = self.k(x).view(B, H, N, D).contiguous()
        v = self.v(x).view(B, H, N, D).contiguous()

        # --- Triton Scores ---
        scores = torch.empty((B, H, N, N), device=x.device, dtype=x.dtype)
        grid_scores = (triton.cdiv(N, 32), B * H)

        attention_scores_kernel[grid_scores](
            q, k, scores,
            B, H, N, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            BLOCK_M=32, BLOCK_N=32, BLOCK_D=D
        )

        # --- Triton Softmax ---
        probs = torch.empty_like(scores)
        grid_softmax = (triton.cdiv(N, 32), B * H)

        attn_softmax_kernel[grid_softmax](
            scores, probs,
            B, H, N,
            scores.stride(0), scores.stride(1), scores.stride(2),
            probs.stride(0), probs.stride(1), probs.stride(2),
            BLOCK_M=32, BLOCK_N=32
        )

        # --- Triton Value aggregation ---
        out = torch.empty((B, H, N, D), device=x.device, dtype=x.dtype)
        grid_values = (triton.cdiv(N, 32), B * H)

        attention_values_kernel[grid_values](
            probs, v, out,
            B, H, N, D,
            probs.stride(0), probs.stride(1), probs.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            BLOCK_M=32, BLOCK_N=32, BLOCK_D=D
        )

        return out.view(B, N, C)


def benchmark_block(block_m_sizes=[16, 32, 64]):
    torch.manual_seed(0)
    B, N, C = 1, 512, 64

    x = torch.randn(B, N, C, device="cuda")

    model = TritonAttentionBlock(dim=C, num_heads=1).cuda()

    results = []

    for BS in block_m_sizes:
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(10):
            model(x)

        torch.cuda.synchronize()
        ms = (time.time() - start) / 10 * 1000
        results.append(ms)
        print(f"BLOCK_M={BS}: {ms:.3f} ms")

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(block_m_sizes, results, marker="o")
    plt.xlabel("BLOCK_M size")
    plt.ylabel("Runtime (ms)")
    plt.title("Triton Attention Performance vs Block Size")
    plt.grid(True)
    plt.savefig("block_size_benchmark.png")
    print("Saved: block_size_benchmark.png")


if __name__ == "__main__":
    benchmark_block()
