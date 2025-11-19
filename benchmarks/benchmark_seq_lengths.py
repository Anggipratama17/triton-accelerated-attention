import torch
import time
import matplotlib.pyplot as plt

from kernels.triton_attention_layer import TritonAttentionLayer

def benchmark_once(N, triton_attn, torch_attn, device):
    x = torch.randn(1, N, triton_attn.dim, device=device)

    # Triton timing
    torch.cuda.synchronize()
    start = time.time()
    _ = triton_attn(x)
    torch.cuda.synchronize()
    triton_ms = (time.time() - start) * 1000

    # PyTorch timing
    torch.cuda.synchronize()
    start = time.time()
    _ = torch_attn(x, x, x)[0]
    torch.cuda.synchronize()
    torch_ms = (time.time() - start) * 1000

    return triton_ms, torch_ms


def main():
    device = "cuda"
    dim = 64
    heads = 4

    triton_attn = TritonAttentionLayer(dim=dim, num_heads=heads).to(device)
    torch_attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True).to(device)

    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    triton_times = []
    torch_times = []

    for N in seq_lengths:
        t_ms, p_ms = benchmark_once(N, triton_attn, torch_attn, device)
        triton_times.append(t_ms)
        torch_times.append(p_ms)
        print(f"N={N}: Triton={t_ms:.3f} ms, Torch={p_ms:.3f} ms")

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, triton_times, label="Triton Attention")
    plt.plot(seq_lengths, torch_times, label="PyTorch Attention")
    plt.xlabel("Sequence length (N)")
    plt.ylabel("Runtime (ms)")
    plt.title("Attention Runtime vs Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.savefig("seq_length_benchmark.png", dpi=200)
    print("Saved: seq_length_benchmark.png")


if __name__ == "__main__":
    main()
